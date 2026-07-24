# sampler_utils.py
import random

import torch

# ONE autoregressive sampling loop, shared by sample.py, sample_ascii.py and the online
# rollout capture in loader.py. there used to be three byte-similar copies of this, and all
# three did `logits, _, _ = model(...)` against a forward() that returns four things -- so
# all three raised the moment forward_arg grew loss_per_sequence, and stayed broken because
# nothing exercised them. the unpack below is `logits, *_`, which cannot break that way
# again no matter how many research quantities get bolted onto the return tuple.


@torch.no_grad()
def ar_sample(model, idx, max_new_tokens, max_seq=None, temperature=1.0, top_k=None,
              eos_id=None, pad_id=None, stop_check_every=32):
    """autoregressive decode. returns [B, T0 + max_new_tokens].

    device-sync accounting: the naive way to honor eos is

        has_finished |= (idx_next.squeeze() == eos_id)
        if has_finished.all(): break

    which drags a bool off the device EVERY TOKEN -- a full pipeline drain per token, for
    a loop whose entire body is one small forward. instead the finished flags live on device
    and are only read on an interval (`stop_check_every`). the sequences that finished early
    are already being fed pad/eos on device, so a late break costs a few wasted tokens and
    costs nothing in correctness. set stop_check_every=0 to never sync and always run the
    full length.

    (`.squeeze()` with no argument is also a bug at batch_size == 1: it collapses [1,1] to
    a 0-d tensor and the |= silently degenerates. squeeze(-1) or [:, 0], never bare.)
    """
    finished = None
    if eos_id is not None:
        finished = torch.zeros(idx.size(0), dtype=torch.bool, device=idx.device)
        # if no explicit pad was given, keep re-emitting eos: it is always a legal token and
        # sampling code downstream truncates at the first one anyway.
        fill_id = eos_id if pad_id is None else pad_id

    for step in range(max_new_tokens):
        idx_cond = idx if (max_seq is None or idx.size(1) <= max_seq) else idx[:, -max_seq:]
        logits, *_ = model(idx_cond, return_logits=True)
        logits = logits[:, -1, :].float() / temperature
        if top_k is not None and top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits = logits.masked_fill(logits < v[:, [-1]], -float('Inf'))
        probs = torch.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)   # [B, 1]

        if finished is not None:
            idx_next = torch.where(finished.unsqueeze(-1),
                                   torch.full_like(idx_next, fill_id), idx_next)
            # index [:, 0], not .squeeze(): bare squeeze collapses [1,1] -> [] at bs==1.
            finished = finished | (idx_next[:, 0] == eos_id)

        idx = torch.cat((idx, idx_next), dim=1)

        # the ONE deliberate host sync in this loop, and it is interval-gated.
        if (finished is not None and stop_check_every
                and (step + 1) % stop_check_every == 0 and bool(finished.all())):
            break

    return idx


class ParquetSampler:
    """
    Provides memory-efficient random sampling of documents from a large
    Parquet file by reading only a subset of row groups.
    """
    def __init__(self, file_path: str):
        # imported here, not at module scope: pyarrow/pandas are heavy and this module is
        # also the home of ar_sample, which the tests import on machines with neither.
        import pyarrow.parquet as pq
        print(f"Opening Parquet file for efficient sampling: {file_path}")
        self.pq_file = pq.ParquetFile(file_path)
        self.num_row_groups = self.pq_file.num_row_groups
        if self.num_row_groups == 0:
            raise ValueError("Parquet file has no row groups.")
        print(f"File contains {self.pq_file.metadata.num_rows} documents in {self.num_row_groups} row groups.")

    def get_random_documents(self, num_docs: int, column: str = 'text') -> list[str]:
        """
        Efficiently samples N documents from the Parquet file.
        """
        collected_docs = []
        # Create a shuffled list of row group indices to read from
        row_group_indices = list(range(self.num_row_groups))
        random.shuffle(row_group_indices)

        # Read random row groups until we have enough documents
        for group_index in row_group_indices:
            # Read one row group (a small chunk of the file) into a pandas DataFrame
            # This is the only part that uses significant memory, and it's temporary.
            group_df = self.pq_file.read_row_group(group_index, columns=[column]).to_pandas()
            collected_docs.extend(group_df[column].tolist())
            
            # If we've collected enough, stop reading more of the file
            if len(collected_docs) >= num_docs:
                break
        
        # Return a random sample of the exact size requested from our collection
        if len(collected_docs) < num_docs:
            print(f"Warning: Could only collect {len(collected_docs)} documents, requested {num_docs}.")
            return collected_docs
        else:
            return random.sample(collected_docs, num_docs)