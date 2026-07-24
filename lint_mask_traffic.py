#!/usr/bin/env python
# lint_mask_traffic.py -- ban dense attention-mask materialization on the training path.
#
# a mask_mod is a function evaluated INSIDE the flex_attention kernel, on registers,
# only within partial blocks. it never touches HBM. materializing it into a
# [B, H, Q, KV] tensor -- to discover its block structure, to hand to SDPA, or to index
# from inside the mask_mod itself -- is exactly the O(S^2) traffic flex_attention exists
# to avoid. at B=128, H=12, Q=640, KV=512 that is 503,316,480 bools = 503.3 MB per layer
# per step; the block-level description is 5,120 bools plus a 50,176-byte CSR = 55.3 KB,
# 9,102x smaller. this is not a style rule.
#
# two independent mechanisms, because neither alone is sufficient:
#
#   1. STATIC (scan_source): an AST pass that flags create_block_mask / create_mask and
#      any mask_mod that subscripts a tensor with both a query and a key index. cheap,
#      runs in CI, catches the obvious reintroduction. it cannot see through helper
#      functions.
#   2. DYNAMIC (AllocationAudit): a TorchFunctionMode that watches every tensor produced
#      inside a block and reports the largest. a mask builder run under a block-scale
#      budget fails the moment anything Q*KV-shaped appears, no matter how it was
#      spelled. this one is exact, and it is what tests actually assert with.
#
# both are generic: they take the files / the budget as parameters and know nothing
# about any particular model. a planted-violation test travels with each.
#
# usage:
#   uv run python lint_mask_traffic.py                       # scan the default file set
#   uv run python lint_mask_traffic.py --files a.py b.py --json

import argparse
import ast
import json
import os

import torch

# the attention-carrying source of this repo. flex_masks.py is the mask builder itself;
# pgptlformer.py is where an attention path could grow a dense mask by accident. a fork
# adding its own attention module passes --files rather than editing this tuple.
DEFAULT_FILES = ("flex_masks.py", "pgptlformer.py")

# functions that evaluate a mask over the whole Q x KV grid. legitimate in tests and in
# one-off tooling; never on a training step.
DENSE_MASK_BUILDERS = {"create_block_mask", "create_mask", "_ordered_to_dense"}


# ---------------------------------------------------------------------------------
# 1. static
# ---------------------------------------------------------------------------------

class _Visitor(ast.NodeVisitor):
    """flags dense-mask builders, and mask_mod-shaped functions that subscript a tensor
    with both of their last two arguments (i.e. read a [.., Q, KV] tensor per element)."""

    def __init__(self, path, source):
        self.path = path
        self.lines = source.splitlines()
        self.stack = []
        self.findings = []

    def _record(self, node, kind, detail):
        self.findings.append({
            "file": self.path, "line": node.lineno,
            "scope": ".".join(self.stack) or "<module>",
            "kind": kind, "detail": detail,
            "source": self.lines[node.lineno - 1].strip(),
        })

    def visit_ClassDef(self, node):
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    def visit_FunctionDef(self, node):
        self.stack.append(node.name)
        args = [a.arg for a in node.args.args]
        # a mask_mod is (b, h, q_idx, kv_idx), possibly after self
        if len(args) >= 4 and args[-1].endswith(("idx", "_i", "kv")) and args[-2].endswith(("idx", "_i", "q")):
            q_name, kv_name = args[-2], args[-1]
            for sub in ast.walk(node):
                if isinstance(sub, ast.Subscript):
                    names = {n.id for n in ast.walk(sub.slice) if isinstance(n, ast.Name)}
                    if q_name in names and kv_name in names:
                        self._record(sub, "mask_mod_reads_dense",
                                     f"{q_name} and {kv_name} index the same tensor: this "
                                     f"mask_mod reads an O(Q*KV) tensor from inside the kernel")
        self.generic_visit(node)
        self.stack.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Call(self, node):
        fn = node.func
        name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
        if name in DENSE_MASK_BUILDERS:
            self._record(node, "dense_mask_builder",
                         f"{name}() evaluates the mask over the whole Q x KV grid")
        self.generic_visit(node)


def scan_source(files=DEFAULT_FILES, root="."):
    out = []
    for f in files:
        path = os.path.join(root, f)
        with open(path) as fh:
            src = fh.read()
        v = _Visitor(os.path.basename(path), src)
        v.visit(ast.parse(src))
        out.extend(v.findings)
    return sorted(out, key=lambda d: (d["file"], d["line"]))


def site_key(f):
    return (f["file"], f["scope"], f["kind"])


# ---------------------------------------------------------------------------------
# 2. dynamic
# ---------------------------------------------------------------------------------

class DenseMaskViolation(AssertionError):
    pass


class AllocationAudit(torch.overrides.TorchFunctionMode):
    """watch every tensor produced inside the block and remember the largest.

        with AllocationAudit(max_elems=B*Q) as audit:
            head.block_masks(...)
        assert audit.largest <= B*Q, audit.report()

    with raise_on_violation=True it throws DenseMaskViolation at the offending op
    instead, which points at the line that built the thing.

    this is a TEST instrument. it is not imported by, and costs nothing on, the training
    path -- the whole point of the static half is that CI does not have to run the model
    to notice a regression."""

    def __init__(self, max_elems, raise_on_violation=False):
        super().__init__()
        self.max_elems = int(max_elems)
        self.raise_on_violation = raise_on_violation
        self.largest = 0
        self.worst = None
        self.violations = []

    def __torch_function__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        for t in self._tensors(out):
            n = t.numel()
            if n > self.largest:
                self.largest = n
                self.worst = (getattr(func, "__name__", str(func)), tuple(t.shape))
            if n > self.max_elems:
                rec = (getattr(func, "__name__", str(func)), tuple(t.shape), n)
                self.violations.append(rec)
                if self.raise_on_violation:
                    raise DenseMaskViolation(
                        f"{rec[0]} produced a tensor of shape {rec[1]} = {n} elements, "
                        f"over the {self.max_elems}-element budget. a mask description "
                        f"must be block-scale or key-linear, never O(Q*KV).")
        return out

    @staticmethod
    def _tensors(out):
        if isinstance(out, torch.Tensor):
            return (out,)
        if isinstance(out, (tuple, list)):
            return tuple(x for x in out if isinstance(x, torch.Tensor))
        return ()

    def report(self):
        head = (f"largest tensor: {self.worst[1]} = {self.largest} elements from "
                f"{self.worst[0]}" if self.worst else "no tensors observed")
        if not self.violations:
            return head + f" (budget {self.max_elems})"
        lines = [f"{n} elements {shape} from {fn}" for fn, shape, n in self.violations[:8]]
        return (head + f"\nbudget {self.max_elems}; {len(self.violations)} violation(s):\n  "
                + "\n  ".join(lines))


# ---------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="*", default=list(DEFAULT_FILES))
    ap.add_argument("--root", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    found = scan_source(a.files, a.root)
    if a.json:
        print(json.dumps(found, indent=2))
        return
    print(f"=== dense-mask materialization sites ({len(found)}) ===")
    for f in found:
        print(f"  {f['file']}:{f['line']:<5} {f['scope']}")
        print(f"      {f['kind']}: {f['detail']}")
        print(f"      | {f['source']}")
    if not found:
        print("  none.")


if __name__ == "__main__":
    main()
