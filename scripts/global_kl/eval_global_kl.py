"""Learn a GLOBAL (task-free) circuit: zero-ablate MLP neurons + attention heads on FineWeb-Edu
text and rank them by how much keeping them preserves the unintervened next-token distribution.

Differences from eval_sva.py / eval_mib.py, all of which follow from "global":
  * No counterfactual and no minimal pairs. The ablation is ZERO (--ablation is fixed), so the
    only reference is the model's own unintervened forward on the same tokens.
  * The objective is KL(p_clean || p_masked), averaged over every token position -- not a
    logit diff on one answer token. Lower is better, so it is MINIMIZED directly (no sign flip).
  * Nodes are TIED over token position (`mlp_tied+attn_head_tied`: one score per (layer, neuron)
    and per (layer, head)). A position-indexed node cannot be "globally important" -- it would
    not even be defined on a document of a different length.

Arms (--method):
  mattr   -- MAttr: soft top-k mask (`--variant topk`) + log-k schedule, per CLAUDE.md's
             headline recipe; --optimizer {sgd,adam} is the arm distinction.
  mc_ig   -- Expected Gradients over the ablation->clean INPUT-EMBEDDING path, score =
             mean_alpha grad . (clean_act - base), alpha per EXAMPLE. Now the UNIFIED
             implementation: learning_to_attribute.expected_gradients drives the loop and shares
             MAttr's --steps / --train-batch-size / --k-schedule (alpha ~ k/total; uniform =
             canonical U(0,1)); the hooker's capture_node_acts / override_embed /
             contract_node_grads own the model-specific half (see make_embed_ig_grad_fn).
             One forward+backward per step -> equal --steps is compute-comparable to mattr
             (but NOT draw-matched: per-example alpha is free for IG, while MAttr fixes one
             k per forward -- see expected_gradients's docstring).
             The ALPHA CONVENTION IS FLIPPED vs eval_sva (alpha=1 clean, alpha=0 the
             zero/mean baseline) because the baseline is an ablation, not a patch input.
             Round-1 files (`*_mc_ig_m*`) are the superseded fixed-batch-sweep
             implementation, kept in gradient_scores.
  ig      -- the same integral on a fixed right-endpoint grid alpha in {1/m, ..., 1}
             (legacy loop in gradient_scores; --ig-steps is the grid size).
  random  -- seeded random ranking (the sweep also always reports its own random ordering).

NOTE ON IxG: it is deliberately not offered. IxG is IG at the clean endpoint alone, and the
metric here is KL TO THE CLEAN MODEL, which is exactly 0 with exactly 0 gradient there -- so
IxG scores are identically zero (in bf16, numerical noise). Use --metric ce if you want a
gradient-at-clean baseline that is not degenerate.

    python scripts/global_kl/eval_global_kl.py --method mattr --optimizer sgd --lr 1.0 --steps 1000
"""
import argparse, json, logging, math, random, time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from learning_to_attribute import LlamaAttributionHooks, learn_scores, expected_gradients

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

MODEL_FULLNAMES = {"llama3": "meta-llama/Llama-3.1-8B",
                   "llama3-instruct": "meta-llama/Llama-3.1-8B-Instruct",
                   "llama3.2-1b": "meta-llama/Llama-3.2-1B",     # smoke-test size
                   "qwen2.5": "Qwen/Qwen2.5-0.5B"}


# ---------------------------------------------------------------- data

def load_chunks(path, tok, n_total, seq_len):
    """First ``n_total`` docs of the jsonl that tokenize to at least ``seq_len``, truncated to
    exactly ``seq_len`` (BOS included). Fixed length everywhere means no padding, so every
    position is a real prediction and the KL average is not diluted by pad slots."""
    chunks, skipped = [], 0
    with open(path) as f:
        for line in f:
            if len(chunks) >= n_total:
                break
            ids = tok(json.loads(line)["text"], return_tensors="pt").input_ids[0]
            if ids.shape[0] < seq_len:
                skipped += 1
                continue
            chunks.append(ids[:seq_len])
    if len(chunks) < n_total:
        raise SystemExit(f"{path}: only {len(chunks)} docs reach {seq_len} tokens "
                         f"({skipped} too short) -- need {n_total}; fetch more with "
                         f"scripts/global_kl/fetch_fineweb_edu.py")
    logger.info("Loaded %d chunks of %d tokens (%d docs skipped as too short)",
                len(chunks), seq_len, skipped)
    return torch.stack(chunks)


# ---------------------------------------------------------------- metrics

def masked_metrics(logits, ref_lp, ids):
    """Per-batch (kl, ce, top1_agree) against the unintervened reference.

    kl  = KL(p_clean || p_masked) averaged over all positions -- the training objective.
    ce  = next-token cross-entropy of the MASKED model on the real text (positions 0..L-2).
    top1_agree = fraction of positions where the masked argmax matches the clean argmax.
    """
    lp = logits.float().log_softmax(-1)
    p = ref_lp.exp()
    kl = (p * (ref_lp - lp)).sum(-1).mean()
    ce = -lp[:, :-1].gather(-1, ids[:, 1:, None]).squeeze(-1).mean()
    agree = (lp.argmax(-1) == ref_lp.argmax(-1)).float().mean()
    return kl, ce, agree


def corpus_means(hf, hooker, chunks, device, batch_size):
    """Corpus-mean activation at each masked site, averaged over documents AND positions:
    ``{"mlp": {layer: [1,1,intermediate]}, "attn": {layer: [1,1,hidden]}, "embed": [1,1,d]}``.

    These become the ablation value for --ablation mean. Zero ablation deletes a neuron's
    CONSTANT contribution along with its input-dependent one -- every neuron has a non-zero
    mean activation, so zeroing it perturbs the residual stream even for a neuron that never
    varies. Round 1 measured what that costs: ~0.005 nats of KL per zeroed neuron, additive,
    so 241 of 459,776 nodes is already KL=1. Mean ablation holds the constant part and removes
    only the variation, which is what "is this node doing work here?" is supposed to mean.

    Broadcast shape [1,1,*] on purpose: dropped straight into ``hooker.cf_acts_*``, the same
    slot a cached counterfactual activation would occupy, so no hook code changes.
    """
    sums, n_pos, embed_sum = {}, 0, None
    hooker.mask = None
    handles = []
    def mk(li, kind):
        def hook(mod, args):
            v = args[0].detach().float().sum((0, 1))
            sums[(li, kind)] = v if (li, kind) not in sums else sums[(li, kind)] + v
        return hook
    for li in range(len(hf.model.layers)):
        handles.append(hf.model.layers[li].mlp.down_proj.register_forward_pre_hook(mk(li, "mlp")))
        handles.append(hf.model.layers[li].self_attn.o_proj.register_forward_pre_hook(mk(li, "attn")))
    box = {}
    handles.append(hf.model.embed_tokens.register_forward_hook(
        lambda m, i, o: box.__setitem__("e", box.get("e", 0) + o.detach().float().sum((0, 1)))))
    with torch.no_grad():
        for bi in range(math.ceil(chunks.shape[0] / batch_size)):
            ids = chunks[bi * batch_size:(bi + 1) * batch_size].to(device)
            hf(ids)
            n_pos += ids.shape[0] * ids.shape[1]
    for h in handles:
        h.remove()
    dt = next(hf.parameters()).dtype
    out = {"mlp": {}, "attn": {}}
    for (li, kind), v in sums.items():
        out[kind][li] = (v / n_pos).view(1, 1, -1).to(dt)
    out["embed"] = (box["e"] / n_pos).view(1, 1, -1).to(dt)
    return out


# ---------------------------------------------------------------- gradient attribution

def gradient_scores(hf, hooker, chunks, total, device, *, batch_size, draws, seed,
                    metric="kl", mc=True, means=None):
    """IG / Expected Gradients over the zero->clean INPUT-EMBEDDING path, scored at the node acts.

    NOTE: the mc=True (Expected Gradients) path is SUPERSEDED by learning_to_attribute.expected_gradients +
    make_embed_ig_grad_fn (same per-example-alpha estimator; alpha now comes from the shared
    --k-schedule and batches are sampled for --steps steps instead of swept once). Kept for
    --method ig (the fixed grid) and to reproduce the round-1 ``*_mc_ig_m*`` files.

    Path: emb(alpha) = base + alpha * (emb_clean - base), so alpha=0 is the baseline and
    alpha=1 is clean. The baseline is the input analogue of the ablation these scores will be
    EVALUATED under: the zero embedding for --ablation zero, the corpus-mean embedding for
    --ablation mean (``means``). Same for the node delta below -- (clean - 0) vs (clean - mean).
    Mismatching the two would score one intervention and evaluate another.
    ``mc=True`` draws alpha ~ U(0,1) per example (unbiased at any number of draws); ``mc=False``
    walks the right-endpoint grid {1/m, ..., 1}, which contains the clean end and omits the
    degenerate all-zero-embedding end.

    Score of node j = mean_alpha [ d(-metric)/d a_j ] . (a_j^clean - 0), summed over batch and
    token positions (the tie over positions IS the node, so the sum is the node's total effect).
    Accumulated in float64 across batches: with ~5e5 nodes and bf16 grads the per-node sums are
    small and differ by orders of magnitude across layers.
    """
    layers = hf.model.layers
    L, N, nh, Hd = len(layers), hooker.intermediate_size, hooker.num_heads, hooker.head_dim
    is_node = hooker.mask_type == "node"
    hooker.mask = None                       # masking hooks inert; this is a clean-path forward
    gen = torch.Generator(device="cpu"); gen.manual_seed(seed)
    scores = torch.zeros(total, dtype=torch.float64)

    def capture(ids, want_grad, embed_override=None):
        store, handles = {}, []
        def mk(li, kind):
            def hook(mod, args):
                x = args[0]
                if want_grad:
                    x.requires_grad_(True); x.retain_grad()
                store[(li, kind)] = x
                return (x,) + tuple(args[1:])
            return hook
        for li in range(L):
            handles.append(layers[li].mlp.down_proj.register_forward_pre_hook(mk(li, "mlp")))
            handles.append(layers[li].self_attn.o_proj.register_forward_pre_hook(mk(li, "attn")))
        if embed_override is not None:
            handles.append(hf.model.embed_tokens.register_forward_hook(
                lambda mod, inp, out: embed_override))
        logits = hf(ids).logits
        for h in handles:
            h.remove()
        return store, logits

    n_batches = math.ceil(chunks.shape[0] / batch_size)
    for bi in range(n_batches):
        ids = chunks[bi * batch_size:(bi + 1) * batch_size].to(device)
        with torch.no_grad():
            clean_acts, ref_logits = capture(ids, False)
            clean_acts = {k: v.detach().float() for k, v in clean_acts.items()}
            ref_lp = ref_logits.float().log_softmax(-1)
            cap = {}
            h = hf.model.embed_tokens.register_forward_hook(
                lambda m, i, o: cap.__setitem__("e", o.detach()))
            hf(ids); h.remove()
            e_clean = cap["e"]
            e_base = (means["embed"] if means is not None
                      else torch.zeros_like(e_clean[:1, :1]))

        g_acc = {k: torch.zeros_like(v) for k, v in clean_acts.items()}
        for d in range(draws):
            if mc:
                alpha = torch.rand(ids.shape[0], 1, 1, generator=gen).to(e_clean)
            else:
                alpha = torch.full((ids.shape[0], 1, 1), (d + 1) / draws).to(e_clean)
            store, logits = capture(ids, True, embed_override=e_base + alpha * (e_clean - e_base))
            kl, ce, _ = masked_metrics(logits, ref_lp, ids)
            (-(ce if metric == "ce" else kl)).backward()      # goodness = -loss
            for k in g_acc:
                g_acc[k] += store[k].grad.float()
        for k in g_acc:
            g_acc[k] /= draws

        for li in range(L):
            dm = clean_acts[(li, "mlp")]
            da = clean_acts[(li, "attn")]
            if means is not None:
                dm = dm - means["mlp"][li].float()
                da = da - means["attn"][li].float()
            cm = (g_acc[(li, "mlp")] * dm)                                   # [B, P, N]
            ga = g_acc[(li, "attn")]; ca = da
            B, P = ga.shape[0], ga.shape[1]
            ch = (ga.view(B, P, nh, Hd) * ca.view(B, P, nh, Hd)).sum(-1)     # [B, P, nh]
            if is_node:   # layout [attn: L*nh][mlp: L]
                off0 = hooker._node_offset
                scores[off0 + L * nh + li] += cm.sum().double().cpu()
                scores[off0 + li * nh:off0 + (li + 1) * nh] += ch.sum((0, 1)).double().cpu()
            else:         # layout [mlp: L*N][attn: L*nh]
                scores[li * N:(li + 1) * N] += cm.sum((0, 1)).double().cpu()
                off = hooker.mlp_tied_total + li * nh
                scores[off:off + nh] += ch.sum((0, 1)).double().cpu()
        logger.info("  grad batch %d/%d", bi + 1, n_batches)
    return scores.float()


def make_embed_ig_grad_fn(hf, hooker, chunks, *, batch_size, metric="kl", means=None,
                          device="cuda", alpha_per_example=True):
    """``grad_fn(alpha)`` for ``trainer.expected_gradients`` -- the unified mc_ig arm's environment.

    Mirrors the MAttr ``loss_fn``: sample ONE batch from ``chunks``, compute the clean
    reference (capturing the node activations along the way), then run a forward from the
    alpha-interpolated embedding ``emb = base + alpha * (emb_clean - base)``, backward the
    metric, and return ``(hooker.contract_node_grads(dL/da, clean - base), loss)``. The
    embedding base and the node delta both come from the ablation (zero or corpus mean);
    see gradient_scores for why mismatching them would be wrong. Numerics per draw are
    identical to gradient_scores' -- only the alpha/batch loop moved into the trainer.

    ``alpha_per_example=True`` (default, the round-1 / eval_sva convention) gives every
    example its own alpha -- free variance reduction. ``False`` shares ONE alpha across the
    batch, which is draw-matched to MAttr's one-k-per-forward (see expected_gradients's docstring
    on the asymmetry) at the cost of that reduction.
    """
    n = chunks.shape[0]

    def grad_fn(draw_alphas):
        idx = torch.randint(0, n, (batch_size,))
        ids = chunks[idx].to(device)
        hooker.mask = None                     # masking hooks inert; both forwards clean-path
        with torch.no_grad():
            store, handles = hooker.capture_node_acts(want_grad=False)
            ref_lp = hf(ids).logits.float().log_softmax(-1)
            for h in handles:
                h.remove()
            clean_acts = {k: v.float() for k, v in store.items()}
            e_clean = hooker._get_embed_module()(ids)      # a lookup, not a full forward
            e_base = (means["embed"] if means is not None
                      else torch.zeros_like(e_clean[:1, :1]))
        # one alpha per EXAMPLE ([B,1,1]) by default, as in round-1 / eval_sva -- or one
        # shared alpha ([1,1,1], broadcast) for the MAttr-draw-matched ablation; .to(e_clean)
        # matches gradient_scores' cast of alpha to the embedding dtype
        alphas = draw_alphas(ids.shape[0] if alpha_per_example else 1).view(-1, 1, 1).to(e_clean)
        store, handles = hooker.capture_node_acts(want_grad=True)
        handles.append(hooker.override_embed(e_base + alphas * (e_clean - e_base)))
        logits = hf(ids).logits
        for h in handles:
            h.remove()
        kl, ce, _ = masked_metrics(logits, ref_lp, ids)
        loss = ce if metric == "ce" else kl
        loss.backward()
        grads = {k: v.grad.float() for k, v in store.items()}
        deltas = (clean_acts if means is None else
                  {(li, kind): clean_acts[(li, kind)] - means[kind][li].float()
                   for (li, kind) in clean_acts})
        return hooker.contract_node_grads(grads, deltas), loss.item()

    return grad_fn


# ---------------------------------------------------------------- magnitude null

@torch.no_grad()
def magnitude_scores(hf, hooker, chunks, total, device, *, batch_size, means=None):
    """RMS L2 norm of what each node WRITES into the residual stream. No gradients, no
    objective -- the null hypothesis for this whole experiment.

    Under zero ablation the sweep's damage is roughly additive in the activation mass deleted
    (see docs/global_kl_circuits.md), so "important" may be nothing more than "large". If this
    ranking matches the learned ones on the sweep, the learned ones are not finding a circuit;
    they are finding the big nodes. It is the control that makes the other arms falsifiable.

    Exact, not approximated:
      * MLP neuron j of layer l writes ``a_j * W_down[:, j]``, so its per-position write norm is
        ``|a_j| * ||W_down[:,j]||``. Accumulate ``sum_p a_j^2`` -> RMS, times the column norm.
      * Head h writes ``W_o[:, slice] @ x_slice``. Summed over positions its squared norm is
        ``<G_h, S_h>`` with ``G_h = W_slice^T W_slice`` and ``S_h = sum_p x_p x_p^T`` -- both
        [head_dim, head_dim], so this costs a 128x128 accumulator per head instead of
        materializing the per-position writes.

    With --ablation mean the deleted quantity is the DEVIATION from the mean, so ``means`` is
    subtracted first; that is the same quantity the mask actually removes.
    """
    layers = hf.model.layers
    Lh, N, nh, Hd = len(layers), hooker.intermediate_size, hooker.num_heads, hooker.head_dim
    hooker.mask = None
    a2 = {li: torch.zeros(N, dtype=torch.float64, device=device) for li in range(Lh)}
    S = {li: torch.zeros(nh, Hd, Hd, dtype=torch.float64, device=device) for li in range(Lh)}
    n_pos = 0

    handles = []
    def mk(li, kind):
        def hook(mod, args):
            x = args[0].detach().float()
            if means is not None:
                x = x - means[kind][li].float()
            if kind == "mlp":
                a2[li] += x.pow(2).sum((0, 1)).double()
            else:
                xh = x.view(-1, nh, Hd).permute(1, 0, 2)          # [nh, B*P, Hd]
                S[li] += torch.bmm(xh.transpose(1, 2), xh).double()
        return hook
    for li in range(Lh):
        handles.append(layers[li].mlp.down_proj.register_forward_pre_hook(mk(li, "mlp")))
        handles.append(layers[li].self_attn.o_proj.register_forward_pre_hook(mk(li, "attn")))
    for bi in range(math.ceil(chunks.shape[0] / batch_size)):
        ids = chunks[bi * batch_size:(bi + 1) * batch_size].to(device)
        hf(ids)
        n_pos += ids.shape[0] * ids.shape[1]
    for h in handles:
        h.remove()

    scores = torch.zeros(total, dtype=torch.float64)
    for li in range(Lh):
        Wd = layers[li].mlp.down_proj.weight.float()              # [d_model, N]
        col = Wd.pow(2).sum(0).sqrt().double()                    # ||W_down[:, j]||
        scores[li * N:(li + 1) * N] = ((a2[li] / n_pos).sqrt() * col).cpu()
        Wo = layers[li].self_attn.o_proj.weight.float()           # [d_model, nh*Hd]
        Wh = Wo.view(-1, nh, Hd).permute(1, 0, 2)                 # [nh, d_model, Hd]
        G = torch.bmm(Wh.transpose(1, 2), Wh).double()            # [nh, Hd, Hd]
        q = (G * S[li]).sum((1, 2)).clamp_min(0) / n_pos          # mean squared write norm
        off = hooker.mlp_tied_total + li * nh
        scores[off:off + nh] = q.sqrt().cpu()
    return scores.float()


# ---------------------------------------------------------------- sparsity sweep

@torch.no_grad()
def sweep(hf, hooker, chunks, orderings, sparsities, total, device, batch_size):
    """Mean (kl, ce, top1_agree) at each sparsity for each ranking, plus the clean/all-ablated
    endpoints. Loops BATCHES outermost so each batch's reference forward is paid once for all
    (ranking, sparsity) cells instead of once per cell."""
    masks = {}
    for name, order in orderings.items():
        for frac in sparsities:
            k = max(1, int(round(frac * total)))
            m = torch.zeros(total, device=device)
            m[order[:k]] = 1.0
            masks[(name, frac)] = m
    masks[("_endpoint", 1.0)] = torch.ones(total, device=device)     # keep everything
    masks[("_endpoint", 0.0)] = torch.zeros(total, device=device)    # ablate everything

    acc = {key: [0.0, 0.0, 0.0] for key in masks}
    n_batches = math.ceil(chunks.shape[0] / batch_size)
    for bi in range(n_batches):
        ids = chunks[bi * batch_size:(bi + 1) * batch_size].to(device)
        hooker.mask = None
        ref_lp = hf(ids).logits.float().log_softmax(-1)
        for key, m in masks.items():
            hooker.mask = m
            kl, ce, agree = masked_metrics(hf(ids).logits, ref_lp, ids)
            a = acc[key]
            a[0] += kl.item(); a[1] += ce.item(); a[2] += agree.item()
        logger.info("  sweep batch %d/%d", bi + 1, n_batches)
    hooker.mask = None
    for a in acc.values():
        for i in range(3):
            a[i] /= n_batches

    out = {"sparsities": list(sparsities),
           "clean": dict(zip(("kl", "ce", "top1_agree"), acc[("_endpoint", 1.0)])),
           "ablated": dict(zip(("kl", "ce", "top1_agree"), acc[("_endpoint", 0.0)]))}
    for name in orderings:
        out[name] = {mn: [acc[(name, f)][i] for f in sparsities]
                     for i, mn in enumerate(("kl", "ce", "top1_agree"))}
    return out


def make_grid(total, n, kind):
    """Sparsity grid (fractions of nodes KEPT).

    ``log``      -- log-spaced from 1/total to 1, the usual circuit-discovery grid.
    ``log_both`` -- half those points on the sparse end, half MIRRORED at the dense end
                    (keep = 1 - frac, i.e. ablate a log-spaced few). DEFAULT, and not a
                    cosmetic choice: under ZERO ablation with a global objective the model is
                    destroyed at every budget the one-sided grid can see -- the first round
                    measured KL 12.5 at 42% kept against 11.0 with EVERYTHING ablated. The
                    regime where a circuit actually reproduces the model is at the top, so the
                    grid has to resolve 1 - 1e-3 as finely as it resolves 1e-3.
    """
    lo = np.logspace(math.log10(1.0 / total), 0.0, n)
    if kind == "log":
        return sorted(set(float(x) for x in lo))
    half = np.logspace(math.log10(1.0 / total), math.log10(0.5), n // 2)
    grid = set(float(x) for x in half) | {float(1.0 - x) for x in half}
    return sorted(grid)


def thresholds(sparsities, curve, total, levels, direction):
    """Smallest kept-node count reaching each level. ``direction`` is "below" (KL) or
    "above" (agreement). None when the curve never gets there on this grid."""
    out = {}
    for lv in levels:
        hit = next((s for s, y in zip(sparsities, curve)
                    if (y <= lv if direction == "below" else y >= lv)), None)
        out[str(lv)] = None if hit is None else int(round(hit * total))
    return out


def log_auc(xs, ys):
    """Trapezoid mean of ``ys`` against log10(sparsity) -- one summary number per curve."""
    lx = np.log10(np.asarray(xs, float)); ya = np.asarray(ys, float)
    return float(np.sum((lx[1:] - lx[:-1]) * (ya[1:] + ya[:-1]) / 2) / (lx[-1] - lx[0]))


# ---------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="llama3", choices=list(MODEL_FULLNAMES))
    p.add_argument("--data", default="data/fineweb_edu_200.jsonl")
    p.add_argument("--method", default="mattr",
                   choices=["mattr", "mc_ig", "ig", "magnitude", "random"],
                   help="magnitude = the objective-free write-norm null; see magnitude_scores. "
                        "mc_ig = unified embedding-path Expected Gradients (shares mattr's --steps/"
                        "--train-batch-size/--k-schedule; canonical IG is --k-schedule "
                        "uniform); ig = the legacy fixed-grid variant")
    p.add_argument("--nodes", default="mlp_tied+attn_head_tied",
                   choices=["mlp_tied+attn_head_tied", "mlp_tied", "node"],
                   help="node granularity; all are position-tied (a global circuit has no pos)")
    p.add_argument("--metric", default="kl", choices=["kl", "ce"],
                   help="training / attribution objective. kl = KL(clean||masked) (the point of "
                        "the experiment); ce = next-token CE, the non-degenerate-at-clean option")
    p.add_argument("--n-train", type=int, default=100)
    p.add_argument("--n-eval", type=int, default=50)
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--variant", default="topk", choices=["topk", "hard_topk", "hard_topk_identity"],
                   help="mask forward; topk = soft (the CLAUDE.md headline MAttr recipe)")
    p.add_argument("--ablation", default="zero", choices=["zero", "mean"],
                   help="value the COMPLEMENT of the circuit is set to. zero = delete the node "
                        "outright; mean = hold it at its corpus-mean activation (see corpus_means)")
    p.add_argument("--k-schedule", default="log", choices=["log", "log_both", "uniform",
                                                           "logit"],
                   help="log_both splits draws between small k (supervises which nodes to KEEP "
                        "FIRST) and small complement (which to DROP LAST). Plain log only ever "
                        "supervises the head of the ranking -- and round 1's dense end is where "
                        "the learned orderings lost to a random one. Also mc_ig's alpha "
                        "distribution (alpha = k/total): uniform = canonical Expected Gradients, "
                        "anything else = a p(alpha)-weighted path integral.")
    p.add_argument("--optimizer", default="sgd", choices=["sgd", "adam"])
    p.add_argument("--lr", type=float, default=None,
                   help="default: 1.0 for sgd, 0.05 for adam (the node-level MIB optima)")
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--T", type=float, default=0.5)
    p.add_argument("--n-iters", type=int, default=30)
    p.add_argument("--train-batch-size", type=int, default=4)
    p.add_argument("--eval-batch-size", type=int, default=4)
    p.add_argument("--ig-steps", type=int, default=1,
                   help="grid points for ig (mc_ig is driven by --steps now)")
    p.add_argument("--alpha-per-batch", action="store_true",
                   help="mc_ig only: ONE shared alpha per step instead of one per example. "
                        "Draw-matched to MAttr's one-k-per-forward, at the cost of the free "
                        "per-example variance reduction. Default (off) is the round-1 / "
                        "eval_sva per-example convention.")
    p.add_argument("--grad-examples", type=int, default=None,
                   help="docs used by the gradient arms (default: --n-train)")
    p.add_argument("--n-sparsities", type=int, default=20)
    p.add_argument("--sparsity-grid", default="log_both", choices=["log_both", "log"],
                   help="log_both also resolves the DENSE end (1 - log-spaced); see make_grid")
    p.add_argument("--scores-from", default=None, metavar="PATH",
                   help="re-sweep a saved *_scores.pt instead of scoring again (no training)")
    p.add_argument("--reuse-scores", action="store_true",
                   help="if THIS run's own *_scores.pt already exists, load it and only re-run "
                        "the sweep. Re-running the submit script is then a cheap re-eval "
                        "(a changed --sparsity-grid costs a sweep, not a retrain).")
    p.add_argument("--top-report", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default="results/global_kl")
    p.add_argument("--tag", default=None)
    args = p.parse_args()

    torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)
    if args.lr is None:
        args.lr = 1.0 if args.optimizer == "sgd" else 0.05
    tag = args.tag or (f"{args.method}"
                       + (f"_{args.optimizer}_lr{args.lr:g}_s{args.steps}"
                          if args.method == "mattr" else "")
                       # mc_ig ALWAYS names its alpha distribution (= the k-schedule): the
                       # flag default "log" is a mattr convention, and an untagged mc_ig run
                       # would silently read as canonical (uniform-alpha) Expected Gradients. Also
                       # keeps the new stems distinct from the round-1 `mc_ig_m*` files.
                       + (f"_{args.k_schedule}_s{args.steps}"
                          + ("_abatch" if args.alpha_per_batch else "")
                          if args.method == "mc_ig" else "")
                       + (f"_m{args.ig_steps}" if args.method == "ig" else "")
                       + (f"_{args.metric}" if args.metric != "kl" else "")
                       + (f"_{args.ablation}" if args.ablation != "zero" else "")
                       + (f"_{args.k_schedule}"
                          if args.k_schedule != "log" and args.method != "mc_ig" else "")
                       + f"_seed{args.seed}")
    out_dir = Path(args.output); out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.model}_{args.nodes.replace('+', '-')}_{tag}"
    logger.info("Run: %s", stem)

    if args.reuse_scores and not args.scores_from:
        cached = out_dir / f"{stem}_scores.pt"
        if cached.exists():
            args.scores_from = str(cached)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    name = MODEL_FULLNAMES[args.model]
    tok = AutoTokenizer.from_pretrained(name)
    hf = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16).to(device)
    hf.eval().requires_grad_(False)

    chunks = load_chunks(args.data, tok, args.n_train + args.n_eval, args.seq_len)
    train_chunks, eval_chunks = chunks[:args.n_train], chunks[args.n_train:]

    hooker = LlamaAttributionHooks(hf, args.nodes, seq_len=args.seq_len,
                                   sufficient=False,        # legacy flag: False = top-k CLEAN
                                   zero_ablation=(args.ablation == "zero"))
    total = hooker.total
    logger.info("Nodes (%s): %s", args.nodes, hooker.describe())

    means = None
    if args.ablation == "mean":
        if args.nodes == "node":
            raise SystemExit("--ablation mean is defined at the mlp_tied+attn_head_tied layout; "
                             "the `node` layout masks a whole MLP block, whose mean is a "
                             "different object (mean of the block output, not of each neuron)")
        t_m = time.time()
        means = corpus_means(hf, hooker, train_chunks, device, args.eval_batch_size)
        # Installed in the slot a cached counterfactual would occupy; _interpolate then mixes
        # clean and mean under the mask exactly as it mixes clean and patch elsewhere.
        hooker.cf_acts_mlp = dict(means["mlp"])
        hooker.cf_acts_attn = dict(means["attn"])
        logger.info("corpus means over %d docs in %.1fs", train_chunks.shape[0], time.time() - t_m)
    hooker.register_hooks()

    # ---- score ----
    t0 = time.time()
    train_log = None
    if args.scores_from:
        scores = torch.load(args.scores_from, map_location="cpu").float()
        if scores.shape[0] != total:
            raise SystemExit(f"{args.scores_from} has {scores.shape[0]} scores, but "
                             f"--nodes {args.nodes} on {args.model} has {total}")
        logger.info("Re-sweeping %s (no scoring)", args.scores_from)
    elif args.method == "magnitude":
        scores = magnitude_scores(hf, hooker, train_chunks, total, device,
                                  batch_size=args.eval_batch_size, means=means).cpu()
    elif args.method == "random":
        scores = torch.randn(total)
    elif args.method == "mc_ig":
        gchunks = train_chunks[:(args.grad_examples or args.n_train)]
        grad_fn = make_embed_ig_grad_fn(hf, hooker, gchunks,
                                        batch_size=args.train_batch_size,
                                        metric=args.metric, means=means, device=device,
                                        alpha_per_example=not args.alpha_per_batch)
        logger.info("Expected Gradients (embedding path): %d steps, alpha ~ %s (%s) over %d nodes",
                    args.steps, args.k_schedule,
                    "per batch" if args.alpha_per_batch else "per example", total)
        res = expected_gradients(total, grad_fn, steps=args.steps, k_schedule=args.k_schedule,
                          logger=logger, log_every=50)
        scores = res.scores
        train_log = res.train_log
    elif args.method == "ig":
        gchunks = train_chunks[:(args.grad_examples or args.n_train)]
        scores = gradient_scores(hf, hooker, gchunks, total, device,
                                 batch_size=args.train_batch_size, draws=args.ig_steps,
                                 seed=args.seed, metric=args.metric,
                                 mc=False, means=means).cpu()
    else:
        n_tr = train_chunks.shape[0]

        def loss_fn(mask):
            idx = torch.randint(0, n_tr, (args.train_batch_size,))
            ids = train_chunks[idx].to(device)
            hooker.mask = None
            with torch.no_grad():
                ref_lp = hf(ids).logits.float().log_softmax(-1)
            hooker.mask = mask
            kl, ce, _ = masked_metrics(hf(ids).logits, ref_lp, ids)
            return ce if args.metric == "ce" else kl

        logger.info("MAttr: %d steps, %s lr=%g, variant=%s, k-schedule=%s over %d nodes",
                    args.steps, args.optimizer, args.lr, args.variant, args.k_schedule, total)
        res = learn_scores(total, loss_fn, steps=args.steps, variant=args.variant,
                           k_schedule=args.k_schedule, T=args.T, n_iters=args.n_iters,
                           lr=args.lr, optimizer=args.optimizer, device=device,
                           logger=logger, log_every=50)
        scores = res.scores.cpu()
        train_log = res.train_log
    score_time = time.time() - t0
    if args.scores_from:
        # A re-sweep must NOT overwrite the recorded cost of the run that produced the scores
        # with its own reload time (~0.0s) -- that number is read later as "what did this arm
        # cost", and silently replacing it makes the training cost unrecoverable from the json.
        # Carry the original forward from the sibling json when there is one.
        prior = Path(str(args.scores_from).replace("_scores.pt", ".json"))
        if prior.exists():
            try:
                score_time = json.load(open(prior)).get("score_time_s", score_time)
            except (json.JSONDecodeError, OSError):
                pass
    logger.info("scores done in %.1fs%s", score_time,
                " (carried over; this run only re-swept)" if args.scores_from else "")

    # ---- eval ----
    sparsities = make_grid(total, args.n_sparsities, args.sparsity_grid)
    g = torch.Generator().manual_seed(0)
    orderings = {"learned": scores.argsort(descending=True).to(device),
                 "random": torch.randperm(total, generator=g).to(device)}
    sw = sweep(hf, hooker, eval_chunks, orderings, sparsities, total, device,
               args.eval_batch_size)
    xs = [s * total for s in sparsities]
    aucs = {f"{name}_{m}_auc": log_auc(xs, sw[name][m])
            for name in ("learned", "random") for m in ("kl", "ce", "top1_agree")}
    logger.info("KL-AUC learned=%.4f random=%.4f   (lower is better)   "
                "top1-agree-AUC learned=%.4f random=%.4f",
                aucs["learned_kl_auc"], aucs["random_kl_auc"],
                aucs["learned_top1_agree_auc"], aucs["random_top1_agree_auc"])
    logger.info("endpoints: clean kl=%.4f ce=%.4f | all-ablated kl=%.4f ce=%.4f",
                sw["clean"]["kl"], sw["clean"]["ce"], sw["ablated"]["kl"], sw["ablated"]["ce"])
    # The AUCs above integrate over a grid whose sparse half is a destroyed model under zero
    # ablation, so they compress the arms together. These are the grid-robust readouts, and
    # the direct answer to "how big is the global circuit": the smallest budget that reproduces
    # the model to a given tolerance.
    thr = {"kl": thresholds(sparsities, sw["learned"]["kl"], total, [1.0, 0.5, 0.1], "below"),
           "top1_agree": thresholds(sparsities, sw["learned"]["top1_agree"], total,
                                    [0.9, 0.95, 0.99], "above")}
    logger.info("nodes needed: KL<=1.0:%s <=0.5:%s <=0.1:%s | agree>=0.9:%s >=0.95:%s >=0.99:%s",
                thr["kl"]["1.0"], thr["kl"]["0.5"], thr["kl"]["0.1"],
                thr["top1_agree"]["0.9"], thr["top1_agree"]["0.95"], thr["top1_agree"]["0.99"])

    # ---- what the top of the ranking actually is ----
    top_idx = scores.argsort(descending=True)[:args.top_report]
    top = [{"rank": r, "score": float(scores[i]), **hooker.decode_index(int(i))}
           for r, i in enumerate(top_idx.tolist())]
    n_attn_top = sum(1 for t in top if t["component"] == "attn")
    logger.info("top-%d: %d attn heads, %d mlp neurons; layers %s",
                args.top_report, n_attn_top, args.top_report - n_attn_top,
                sorted({t["layer"] for t in top}))

    torch.save(scores, out_dir / f"{stem}_scores.pt")
    with open(out_dir / f"{stem}.json", "w") as f:
        json.dump({"args": vars(args), "total": total, "node_layout": hooker.describe(),
                   "score_time_s": score_time, "scores_reused": bool(args.scores_from),
                   "sweep": sw, "aucs": aucs, "thresholds": thr,
                   "top_nodes": top, "train_log": train_log}, f, indent=2)
    logger.info("wrote %s", out_dir / f"{stem}.json")


if __name__ == "__main__":
    main()
