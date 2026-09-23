"""Per-example counterfactual-activation cache around ``hooker.cache_cf_activations``.

``cache_cf_activations`` recomputes the CF forward for the SAME example every time it is
resampled across training steps (and, in eval_sva's sweep, once per sparsity for the same
eval batch). The activations are deterministic per example, so that is pure recompute --
~20-25% of a MAttr step. :class:`CFActivationCache` runs the CF forward only on the
UNCACHED examples of each batch (a right-padded sub-batch) and assembles the full batch's
``cf_acts_*`` tensors from cached slices, so the saving scales with the per-example hit
rate rather than requiring a fully-cached batch.

Numerics: activations at real positions are independent of right-padding length (causal
attention; pad keys are either masked or beyond every real query), so sub-batch recompute
and cached slices are mathematically identical to the original full-batch CF forward.
Assembled tensors are zero-padded past each example's length; downstream only ever reads
positions the loss can see (real positions of the base batch), so the loss is unchanged.

Supported for the mlp/attn(/embed) cache slots -- i.e. the node and per-position layouts.
The SAE layouts fall back to plain per-step recomputation.
"""
import torch


class CFActivationCache:
    def __init__(self, hooker, max_gb=4.0, logger=None):
        self.hooker = hooker
        self.max_bytes = int(max_gb * 1e9)
        self.logger = logger
        h = hooker
        self.supported = (h.has_mlp or h.has_attn) and not h.is_sae and self.max_bytes > 0
        self.store = {}          # key -> {"len", "mlp": {li: [p,D]}, "attn": {...}, "embed"}
        self.bytes = 0
        self.full_logged = False
        # per-EXAMPLE stats: fwd_examples / seen_examples = fraction of CF compute paid
        self.seen_examples = 0
        self.fwd_examples = 0

    def prepare(self, keys, src_ids, src_lens):
        """Leave ``hooker.cf_acts_*`` holding the CF activations for this batch.

        ``keys``: one hashable per example (e.g. the corrupted prompt string);
        ``src_ids``: right-padded [B, P] CF input ids; ``src_lens``: real lengths.
        """
        h = self.hooker
        if not self.supported:
            h.cache_cf_activations(src_ids)
            return
        B, P = src_ids.shape
        self.seen_examples += B
        new = []                                  # first occurrence of each uncached key
        seen_here = set()
        for b, k in enumerate(keys):
            if k not in self.store and k not in seen_here:
                new.append(b)
                seen_here.add(k)
        fresh = {}
        if new:
            self.fwd_examples += len(new)
            sub_lens = [src_lens[b] for b in new]
            sub = src_ids[new][:, :max(sub_lens)]   # trim shared right-padding
            h.cache_cf_activations(sub)
            fresh = self._harvest([keys[b] for b in new], sub_lens)
        self._assemble([fresh.get(k) or self.store[k] for k in keys], P)

    def _slots(self):
        h = self.hooker
        if h.has_mlp:
            yield "mlp", h.cf_acts_mlp
        if h.has_attn:
            yield "attn", h.cf_acts_attn
        if h.include_input and getattr(h, "cf_acts_embed", None) is not None:
            yield "embed", {None: h.cf_acts_embed}

    def _harvest(self, keys, lens):
        """Slice the (sub-batch) tensors in ``cf_acts_*`` into per-example entries;
        store them if the byte budget allows. Returns the entries either way."""
        fresh = {}
        for b, (k, p) in enumerate(zip(keys, lens)):
            entry, nbytes = {"len": int(p)}, 0
            for slot, acts in self._slots():
                d = {}
                for li, act in acts.items():
                    t = act[b, :p].clone()
                    d[li] = t
                    nbytes += t.numel() * t.element_size()
                entry[slot] = d
            fresh[k] = entry
            if self.bytes + nbytes <= self.max_bytes:
                self.store[k] = entry
                self.bytes += nbytes
            elif not self.full_logged:
                self.full_logged = True
                if self.logger:
                    self.logger.info(
                        "cf-cache full at %.2f GB (%d examples); further examples are "
                        "recomputed each step", self.bytes / 1e9, len(self.store))
        return fresh

    def _assemble(self, entries, P):
        """Write full-batch [B, P, D] tensors built from per-example entries into the
        hooker's cf_acts_* slots (zero past each example's length)."""
        h = self.hooker
        B = len(entries)
        for slot in ("mlp", "attn", "embed"):
            if slot not in entries[0]:
                continue
            out = {}
            for li in entries[0][slot]:
                ref = entries[0][slot][li]
                buf = ref.new_zeros(B, P, ref.shape[-1])
                for b, e in enumerate(entries):
                    t = e[slot][li]
                    buf[b, : t.shape[0]] = t
                out[li] = buf
            if slot == "mlp":
                h.cf_acts_mlp = out
            elif slot == "attn":
                h.cf_acts_attn = out
            else:
                h.cf_acts_embed = out[None]

    def stats(self):
        if not self.seen_examples:
            return "cf-cache: unused"
        return ("cf-cache: %d/%d example-forwards paid (%.0f%% saved), %d cached, %.2f GB"
                % (self.fwd_examples, self.seen_examples,
                   100 * (1 - self.fwd_examples / self.seen_examples),
                   len(self.store), self.bytes / 1e9))
