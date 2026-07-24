#!/usr/bin/env python
# lint_attention_dtypes.py -- where does this model widen a dtype, and why.
#
# every architecture comparison this repo runs is only meaningful if both arms carry the
# same number of bits per weight and per activation. an `autocast(enabled=False)` region
# or a stray `.float()` on one side of a comparison silently turns an ARCHITECTURE result
# into a PRECISION result, and it does so without changing a single number anyone looks
# at until the wall-clock is already published. the failure mode is not hypothetical: an
# arm that computes QK^T, softmax and PV in fp32 while its counterpart runs bf16 will
# look ~1.8x slower for reasons that have nothing to do with the mechanism under test.
#
# a runtime dtype assertion would catch that -- and would also cost a graph break and a
# device sync on every step of every training run forever, to protect against a mistake
# that is made in the SOURCE, at edit time. so this is a static pass instead. it runs at
# interpreter time, over the source, and costs the training loop exactly zero.
#
# WHY AST AND NOT A TRACED GRAPH: the single most important thing to detect here is
# `torch.autocast(enabled=False)`, and autocast regions do not survive into an FX or
# inductor graph at all -- by the time you have a graph, the region has already decided
# the dtypes and disappeared. dynamo export of the flex_attention path additionally does
# not round-trip on cpu in torch 2.5.1 (create_block_mask's inspect.signature, which is
# also why flex_masks.build_block_mask constructs BlockMask directly). the source is
# where the decision is written, so the source is what gets linted.
#
# usage:
#   uv run python lint_attention_dtypes.py                 # report
#   uv run python lint_attention_dtypes.py --json          # machine-readable
# tests/test_lints.py pins the result against an explicit allowlist, so a NEW widening
# site fails the suite and the deliberate ones pass.

import argparse
import ast
import json
import os

DEFAULT_FILES = ("pgptlformer.py",)

WIDENING_METHODS = {"float", "double"}
NARROWING_METHODS = {"bfloat16", "half"}
FP32_NAMES = {"float32", "float", "double", "float64"}
FP16_NAMES = {"bfloat16", "float16", "half"}


def _dtype_name(node):
    """torch.float32 / torch.bfloat16 -> 'float32' / 'bfloat16'; anything else -> None"""
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) \
            and node.value.id == "torch":
        return node.attr
    return None


class _Visitor(ast.NodeVisitor):
    def __init__(self, path, source):
        self.path = path
        self.lines = source.splitlines()
        self.stack = []
        self.findings = []

    def _record(self, node, kind, detail):
        self.findings.append({
            "file": self.path,
            "line": node.lineno,
            "scope": ".".join(self.stack) or "<module>",
            "kind": kind,
            "detail": detail,
            "source": self.lines[node.lineno - 1].strip(),
        })

    def visit_FunctionDef(self, node):
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node):
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    def visit_With(self, node):
        for item in node.items:
            call = item.context_expr
            if isinstance(call, ast.Call):
                fn = call.func
                name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
                if name == "autocast":
                    enabled = None
                    for kw in call.keywords:
                        if kw.arg == "enabled" and isinstance(kw.value, ast.Constant):
                            enabled = kw.value.value
                    if enabled is False:
                        self._record(call, "autocast_disabled",
                                     "autocast(enabled=False): everything inside runs at "
                                     "full parameter precision")
                    else:
                        self._record(call, "autocast_region", f"autocast(enabled={enabled})")
        self.generic_visit(node)

    def visit_Call(self, node):
        fn = node.func
        if isinstance(fn, ast.Attribute):
            # x.float() / x.double()
            if fn.attr in WIDENING_METHODS and not node.args and not node.keywords:
                self._record(node, "cast_method", f".{fn.attr}()")
            elif fn.attr in NARROWING_METHODS and not node.args and not node.keywords:
                self._record(node, "narrowing_method", f".{fn.attr}()")
            # x.to(torch.float32) / x.to(dtype=torch.float32) / x.type(torch.float32)
            elif fn.attr in ("to", "type"):
                for arg in list(node.args) + [kw.value for kw in node.keywords
                                              if kw.arg == "dtype"]:
                    dn = _dtype_name(arg)
                    if dn in FP32_NAMES:
                        self._record(node, "cast_to", f".{fn.attr}(torch.{dn})")
                    elif dn in FP16_NAMES:
                        self._record(node, "narrowing_to", f".{fn.attr}(torch.{dn})")
        # any dtype=torch.float32 kwarg on any call (tensor factories included)
        for kw in node.keywords:
            if kw.arg == "dtype":
                dn = _dtype_name(kw.value)
                if dn in FP32_NAMES:
                    self._record(node, "dtype_kwarg", f"dtype=torch.{dn}")
        self.generic_visit(node)


def scan_file(path):
    with open(path) as fh:
        src = fh.read()
    v = _Visitor(os.path.basename(path), src)
    v.visit(ast.parse(src))
    return v.findings


def scan(files=DEFAULT_FILES, root="."):
    out = []
    for f in files:
        out.extend(scan_file(os.path.join(root, f)))
    return sorted(out, key=lambda d: (d["file"], d["line"]))


WIDENING_KINDS = {"cast_method", "cast_to", "dtype_kwarg", "autocast_disabled"}


def widening_sites(files=DEFAULT_FILES, root="."):
    """the sites that matter: things that make an activation WIDER than the autocast
    dtype. narrowing casts and plain autocast regions are reported but not gated."""
    return [f for f in scan(files, root) if f["kind"] in WIDENING_KINDS]


def site_key(f):
    return (f["file"], f["scope"], f["detail"])


# the allowlist is DATA, not code: this base file ships with the lint, and a fork that
# adds its own attention module ships an ADDITIONAL json and passes both paths, rather
# than editing this one. that is what keeps the two maintainable (and rebasable)
# separately. load_allowlist() skips a path that does not exist, so extending the tuple
# downstream is safe.
DEFAULT_ALLOWLISTS = ("lint_allowlist_attention_dtypes.json",)


def load_allowlist(paths=DEFAULT_ALLOWLISTS, root="."):
    """-> {(file, scope, detail): justification}, merged in order. a missing file is
    skipped, so a fork's extension is optional and upstream works without it."""
    out = {}
    for p in paths:
        full = os.path.join(root, p)
        if not os.path.exists(full):
            continue
        with open(full) as fh:
            data = json.load(fh)
        for entry in data["entries"]:
            f, scope, detail, why = entry
            out[(f, scope, detail)] = why
    return out


def check(files=DEFAULT_FILES, root=".", allowlists=DEFAULT_ALLOWLISTS):
    """-> (unjustified sites, stale allowlist entries). the whole lint in one call, so a
    fork can point it at a different file set and a different allowlist."""
    found = {site_key(f) for f in widening_sites(files, root)}
    allowed = load_allowlist(allowlists, root)
    return sorted(found - set(allowed)), sorted(set(allowed) - found)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="*", default=list(DEFAULT_FILES))
    ap.add_argument("--allowlists", nargs="*", default=list(DEFAULT_ALLOWLISTS))
    ap.add_argument("--root", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    found = scan(a.files, a.root)
    if a.json:
        print(json.dumps(found, indent=2))
        return
    wide = [f for f in found if f["kind"] in WIDENING_KINDS]
    other = [f for f in found if f["kind"] not in WIDENING_KINDS]
    print(f"=== dtype WIDENING sites ({len(wide)}) "
          f"-- each one must have a numerical justification in a comment ===")
    for f in wide:
        print(f"  {f['file']}:{f['line']:<5} {f['scope']}")
        print(f"      {f['kind']:<18} {f['detail']}")
        print(f"      | {f['source']}")
    print(f"\n=== narrowing / autocast regions ({len(other)}) -- informational ===")
    for f in other:
        print(f"  {f['file']}:{f['line']:<5} {f['scope']:<40} {f['kind']} {f['detail']}")
    unjustified, stale = check(a.files, a.root, a.allowlists)
    print(f"\n=== against the allowlist ({', '.join(a.allowlists)}) ===")
    print(f"  unjustified: {len(unjustified)}   stale entries: {len(stale)}")
    for s in unjustified:
        print(f"  UNJUSTIFIED {s}")
    for s in stale:
        print(f"  STALE       {s}")


if __name__ == "__main__":
    main()
