"""SAE-feature attribution on CausalGym (sufficient / denoising), gemma-2-2b base.

Mirrors the DAS-64 *sufficient* experiment but replaces the learned DAS rotation with a
frozen pretrained JumpReLU residual-stream SAE (Gemma Scope) at a single layer, and learns
a score per SAE feature via sigmoid-top-k masking.

Intervention (denoising; analog of DAS sufficient else-branch  out = cf + ((rot_base-rot_cf)*mask)@W^T):
    out = a_cf + (mask ⊙ (f_base − f_cf)) @ W_dec
  where f = SAE.encode(a). The SAE reconstruction *error* cancels — it is held at the
  counterfactual example's residual (a_cf − decode(f_cf)), analogous to DAS leaving the
  null-space at cf. So top-k features are held CLEAN, the complement set to CF, and the
  SAE-unexplained residual stays corrupted. We train CE(logits, base_label): "which features,
  kept clean, are sufficient to recover clean behavior?"
"""
import argparse, json, math, os, random, subprocess, time
import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from learning_to_attribute.data.causalgym import CausalGymDataset
from learning_to_attribute.sigmoid_topk import sigmoid_topk_hard    # evaluation mask (all variants)
from learning_to_attribute.masks import build_mask          # canonical mask registry
from learning_to_attribute.schedules import sample_k        # canonical k-sampler
from learning_to_attribute.losses import attribution_loss   # canonical loss definitions


class JumpReLUSAE:
    """Gemma Scope JumpReLU SAE loaded from raw params.npz (no b_dec pre-subtraction)."""
    def __init__(self, repo, sae_id, device, dtype=torch.float32):
        p = np.load(hf_hub_download(repo, f"{sae_id}/params.npz"))
        t = lambda k: torch.tensor(p[k], device=device, dtype=dtype)
        self.W_enc, self.W_dec = t("W_enc"), t("W_dec")     # (d_model,d_sae),(d_sae,d_model)
        self.b_enc, self.b_dec = t("b_enc"), t("b_dec")
        self.threshold = t("threshold")
        self.d_sae = self.W_enc.shape[1]

    def encode(self, x):                                    # x: [..., d_model] fp32
        pre = x @ self.W_enc + self.b_enc
        return (pre > self.threshold) * torch.relu(pre)     # JumpReLU -> [..., d_sae]

    def decode(self, f):
        return f @ self.W_dec + self.b_dec


def strip_bos(pair):
    """Template prepends literal '<|endoftext|>' (pythia BOS); gemma adds its own <bos>."""
    pair.base_spans[0] = pair.base_spans[0].replace("<|endoftext|>", "")
    pair.src_spans[0] = pair.src_spans[0].replace("<|endoftext|>", "")
    return pair


def pos_map(tok):
    """Aligned (base_pos, cf_pos, span_idx) over all spans (skip BOS at pos 0).
    span_idx[k] = which span position k belongs to (for per-span, untied masks)."""
    base_pos, cf_pos, span_idx = [], [], []
    for i in range(tok.num_spans):
        ba, sa = tok.base_alignment[i], tok.src_alignment[i]
        if not ba:
            continue
        for j, bp in enumerate(ba):
            sp = sa[min(j, len(sa) - 1)] if sa else bp
            if bp == 0:
                continue
            base_pos.append(bp); cf_pos.append(sp); span_idx.append(i)
    return base_pos, cf_pos, span_idx


def _key(pair):
    """Identity of an example: rendered base/source text plus both labels."""
    return ("".join(pair.base_spans), "".join(pair.src_spans), pair.base_label, pair.src_label)


def build_eval_set(ds, n_eval, max_draws=200_000):
    """Hold out ``n_eval`` DISTINCT examples drawn from the generator.

    Previously train drew from ``seed=42`` and eval from ``seed=123`` -- the same infinite
    generator, so eval examples recurred in training (measured 0-100% overlap by task).
    Here the eval keys are fixed up front and then *excluded* from training by rejection, so
    the training distribution stays the generator's own (see :func:`sample_train_pair`).
    """
    seen, pairs, draws = set(), [], 0
    while len(pairs) < n_eval and draws < max_draws:
        p = strip_bos(ds.sample_pair())
        draws += 1
        k = _key(p)
        if k not in seen:
            seen.add(k)
            pairs.append(p)
    assert len(pairs) == n_eval, f"only {len(pairs)} distinct examples in {draws} draws"
    return pairs, seen, draws


def sample_train_pair(ds, eval_keys, max_tries=10_000):
    """One training example from the ORIGINAL generator, rejecting held-out examples.

    Rejection sampling keeps the training distribution exactly the generator's own
    distribution conditioned on ``key not in eval_keys`` -- unlike sampling uniformly from a
    materialised pool, which would flatten the generator's non-uniform example probabilities
    (correlated variables, repeated label draws). Returns ``(pair, n_rejected)``.
    """
    for i in range(max_tries):
        p = strip_bos(ds.sample_pair())
        if _key(p) not in eval_keys:
            return p, i
    raise RuntimeError(f"could not draw a non-eval example in {max_tries} tries; "
                       f"the generator's support may be smaller than n_eval")


def sae_grad_scores(model, ds, tokenizer, hook, eval_keys, device, S, width,
                    n_examples, grad_loss="ce"):
    """I x G over the per-span SAE variable set (latents + reconstruction-error nodes).

    Paper eq. 21:  s_H = (h(b) - h(s)) . dl/dH |_b.  Our intervention

        new = a_cf + (m (.) (f_b - f_cf)) @ W_dec  [+ m_err (err_b - err_cf)]

    is exactly AFFINE in the mask m, with m=0 -> a_cf and m=1 -> a_base, so

        dl/dm_ji = (f_b,ji - f_cf,ji) . dl/df_ji

    which IS the eq.-21 score: the (h(b) - h(s)) factor is already inside the mask derivative,
    so no separate (clean - patch) multiplication is needed. The gradient is taken at m = 1,
    matching eq. 21's ``|_b`` and ``eval_sva.gradient_scores``'s ig_steps=1 behaviour. The
    reconstruction-error coordinate gets eq. 21 with H = the error term for free, in the same
    units as the feature scores -- it is deliberately NOT special-cased.

    Returns (scores [S*width] detached, info dict). Score sign follows the repo convention
    (``eval_sva.metric_of``): we differentiate GOODNESS = -attribution_loss(..., corrupt_topk=
    False), so a larger score means "restoring this unit to its base value helps preserve the
    base behaviour".
    """
    acc = torch.zeros(S, width, device=device)
    used, rejected, skipped, keys = 0, 0, 0, set()
    for _ in range(n_examples):
        pair, rej = sample_train_pair(ds, eval_keys)     # same stream semantics as MAttr
        rejected += rej
        keys.add(_key(pair))
        tok = ds.tokenize_pair(pair, tokenizer, device)
        bp, cp, si = pos_map(tok)
        if not bp:
            skipped += 1
            continue
        m = torch.ones(S, width, device=device, requires_grad=True)   # m = 1  ->  base point
        logits = run_intervened(model, tok, hook, m, bp, cp, si)
        goodness = -attribution_loss(
            grad_loss, logits.unsqueeze(0),
            torch.tensor([tok.base_label_id], device=device),
            torch.tensor([tok.src_label_id], device=device),
            corrupt_topk=False)                           # iso direction, as in MAttr training
        goodness.backward()
        acc += m.grad
        used += 1
    return acc.div_(max(used, 1)).flatten().detach(), {
        "grad_examples_used": used, "grad_examples_skipped": skipped,
        "n_rejected_eval_draws": rejected, "train_keys": keys}


def curve_metrics(curve, tag="learned", thr=0.9):
    """Aggregates of the SAME IIA-vs-k curve the evaluator already produces (nothing is
    re-measured): log-weighted AUC (paper eq. 8/12), linear-in-k AUC, plateau, k*."""
    ks, ys = curve["k"], curve[f"{tag}_acc"]
    if len(ks) < 2:
        return {}
    lin = sum((ks[i + 1] - ks[i]) * (ys[i] + ys[i + 1]) / 2 for i in range(len(ks) - 1)) / (ks[-1] - ks[0])
    lg = sum((math.log(ks[i + 1]) - math.log(ks[i])) * (ys[i] + ys[i + 1]) / 2
             for i in range(len(ks) - 1)) / (math.log(ks[-1]) - math.log(ks[0]))
    # plateau_acc is the curve's ENDPOINT (largest k on the grid), not its maximum: the curve is
    # not monotone, so max_acc can be reached mid-curve and would overstate the large-k limit.
    # kstar_grid is the smallest k ON THE EVALUATION GRID reaching thr -- not a true crossing.
    kstar_grid = next((k for k, y in zip(ks, ys) if y >= thr), None)
    return {"log_auc": lg, "linear_auc": lin, "plateau_acc": ys[-1], "max_acc": max(ys),
            "kstar_grid": kstar_grid, "kstar_thr": thr}


class SAEIntervention:
    """Forward hook on gemma layer L: caches cf activation, then denoising-intervenes."""
    def __init__(self, sae):
        self.sae = sae
        self.mode = "off"        # "cache" | "intervene"
        self.cf_act = None
        self.base_pos = self.cf_pos = None
        self.mask = None         # [num_spans, d_sae(+1)] per-span (untied) feature mask
        self.span_idx = None     # [P] long: span of each intervened position
        self.error_mode = "cf"   # "cf" (error corrupted) | "clean" (error restored)

    def __call__(self, module, inp, out):
        hs = out[0] if isinstance(out, tuple) else out      # [1, seq, d_model]
        if self.mode == "cache":
            self.cf_act = hs.detach()
            return out
        if self.mode == "intervene":
            bp = torch.tensor(self.base_pos, device=hs.device)
            cp = torch.tensor(self.cf_pos, device=hs.device)
            base_sel = hs[0, bp].float().detach()           # [P, d_model]
            cf_sel = self.cf_act[0, cp].float().detach()
            f_base = self.sae.encode(base_sel)              # [P, d_sae]
            f_cf = self.sae.encode(cf_sel)
            d = self.sae.d_sae
            mfeat = self.mask[self.span_idx, :d]                 # per-span feature mask [P, d_sae]
            delta = (mfeat * (f_base - f_cf)) @ self.sae.W_dec   # [P, d_model]
            new = cf_sel + delta                                 # err held at cf (default)
            if self.error_mode == "clean":
                # error always restored to the clean example
                new = new + ((base_sel - self.sae.decode(f_base)) - (cf_sel - self.sae.decode(f_cf)))
            elif self.error_mode == "node":
                # error is an extra per-span scored node: clean iff its score is in the top-k
                m_err = self.mask[self.span_idx, d].unsqueeze(1)     # [P, 1]
                new = new + m_err * ((base_sel - self.sae.decode(f_base)) - (cf_sel - self.sae.decode(f_cf)))
            new = new.to(hs.dtype)
            hs = hs.clone()
            hs[0, bp] = new
            return (hs,) + tuple(out[1:]) if isinstance(out, tuple) else hs
        return out


def run_intervened(model, tok, hook, mask, base_pos, cf_pos, span_idx):
    """Cache cf, then forward base with denoising intervention; return final-pos logits.
    mask: [num_spans, d_sae(+1)] per-span; span_idx: span of each position."""
    hook.mode = "cache"
    with torch.no_grad():
        model(tok.src_input_ids)
    dev = tok.base_input_ids.device
    hook.mode, hook.mask, hook.base_pos, hook.cf_pos = "intervene", mask, base_pos, cf_pos
    hook.span_idx = torch.as_tensor(span_idx, device=dev, dtype=torch.long)
    logits = model(tok.base_input_ids).logits[0, -1].float()
    hook.mode = "off"
    return logits


def evaluate(model, ds, tokenizer, hook, scores, device, ks, eval_pairs, T=0.5):
    """Sufficiency curve: prob-diff & accuracy vs #features-kept-clean (HARD top-k).

    Evaluation is deliberately identical for every training variant: the learned ranking is
    always read out with a hard top-k mask, so A/B/C differ only in how the ranking was
    trained. ``eval_pairs`` is the held-out set from :func:`build_eval_set`.
    """
    total = scores.numel()
    width = hook.sae.d_sae + (1 if hook.error_mode == "node" else 0)
    S = total // width
    rand_scores = torch.randn_like(scores)
    out = {"k": [], "learned_probdiff": [], "learned_acc": [], "random_probdiff": [], "random_acc": []}
    pairs = []
    for p in eval_pairs:
        tk = ds.tokenize_pair(p, tokenizer, device)
        bp, cp, si = pos_map(tk)
        if bp:
            pairs.append((tk, bp, cp, si))
    with torch.no_grad():
        for k in ks:
            for tag, sc in [("learned", scores), ("random", rand_scores)]:
                mask = sigmoid_topk_hard(sc, k=float(k), T=T).view(S, width)
                pds, accs = [], []
                for tk, bp, cp, si in pairs:
                    lg = run_intervened(model, tk, hook, mask, bp, cp, si)
                    pb = F.log_softmax(lg, -1)
                    pd = (pb[tk.base_label_id] - pb[tk.src_label_id]).item()
                    pds.append(pd); accs.append(float(lg[tk.base_label_id] > lg[tk.src_label_id]))
                out[f"{tag}_probdiff"].append(float(np.mean(pds)))
                out[f"{tag}_acc"].append(float(np.mean(accs)))
            out["k"].append(k)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-2-2b")
    ap.add_argument("--task", default="syntaxgym/npi_ever_subj-relc")
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--sae-repo", default="google/gemma-scope-2b-pt-res")
    ap.add_argument("--sae-id", default="layer_12/width_16k/average_l0_82")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--T", type=float, default=0.5)
    ap.add_argument("--k-schedule", default="log", choices=["uniform", "log"],
                    help="log (default) matches Algorithm 1 / the MAttr headline recipe")
    ap.add_argument("--variant", default="topk", choices=["topk", "hard_topk"],
                    help="training mask, via learning_to_attribute.masks.build_mask. "
                         "topk = differentiable sigmoid top-k in forward AND backward (canonical); "
                         "hard_topk = hard top-k forward, sigmoid-STE backward (the legacy path). "
                         "Evaluation always uses a hard top-k mask regardless of this flag.")
    ap.add_argument("--loss", default="ce", choices=["ce", "logit_diff", "acc"],
                    help="--method mattr: training objective, via learning_to_attribute.losses. "
                         "ce (default) is bit-identical to the previous hardcoded behaviour")
    ap.add_argument("--method", default="mattr", choices=["mattr", "ixg"],
                    help="mattr = learn scores by sigmoid-top-k masking (default); "
                         "ixg = closed-form I x G over the same SAE variable set (no training)")
    ap.add_argument("--grad-examples", type=int, default=4000,
                    help="--method ixg: #examples in the attribution average (matches MAttr's --steps)")
    ap.add_argument("--grad-loss", default="ce", choices=["ce", "logit_diff", "acc"],
                    help="--method ixg: metric differentiated for I x G. ce matches the MAttr "
                         "training objective; logit_diff is the repo's canonical logit difference")
    ap.add_argument("--n-eval", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto", help="auto -> cuda, else mps, else cpu")
    ap.add_argument("--error-mode", default="node", choices=["cf", "clean", "node"],
                    help="default node: error term is always a scored node (consistent w/ arith SAE)")
    ap.add_argument("--output", default="results/sae_npi_subj_relc")
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)

    if args.device == "auto":
        device = ("cuda" if torch.cuda.is_available()
                  else "mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = args.device
    os.makedirs(args.output, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16).to(device).eval()
    model.requires_grad_(False)
    sae = JumpReLUSAE(args.sae_repo, args.sae_id, device)
    # ONE seeded generator stream: the first n_eval distinct draws are held out, everything
    # afterwards is training (rejecting held-out keys), so the split is fully determined by --seed.
    ds = CausalGymDataset(args.task, seed=args.seed)
    eval_pairs, eval_keys, eval_draws = build_eval_set(ds, args.n_eval)
    split_info = {"n_eval": len(eval_pairs), "n_eval_distinct_keys": len(eval_keys),
                  "eval_collection_draws": eval_draws, "train_sampling": "generator, rejecting eval keys"}
    print(f"task={args.task} num_spans={ds.num_spans} d_sae={sae.d_sae} layer={args.layer} "
          f"device={device}", flush=True)
    print(f"split: {split_info}", flush=True)

    hook = SAEIntervention(sae)
    hook.error_mode = args.error_mode
    handle = model.model.layers[args.layer].register_forward_hook(hook)

    S = ds.num_spans
    width = sae.d_sae + (1 if args.error_mode == "node" else 0)    # per-span width (+1 error node)
    total = S * width                                              # per-span (untied) scores
    scores = torch.zeros(total, device=device, requires_grad=True)
    opt = torch.optim.Adam([scores], lr=args.lr)
    print(f"per-span scores: {S} spans x {width} = {total}", flush=True)

    losses, mask_frac_binary = [], []
    n_rejected, train_keys = 0, set()
    t0 = time.time()
    if args.method == "ixg":
        # closed-form I x G over the same variable set: no optimiser, no k-schedule, no mask
        # variant -- just d(goodness)/dm at m=1, averaged over --grad-examples draws.
        scores, ginfo = sae_grad_scores(model, ds, tokenizer, hook, eval_keys, device, S, width,
                                        args.grad_examples, grad_loss=args.grad_loss)
        train_keys = ginfo.pop("train_keys")
        n_rejected = ginfo["n_rejected_eval_draws"]
        split_info.update(ginfo)
        print(f"I x G ({args.grad_loss}): averaged {ginfo['grad_examples_used']} examples, "
              f"score |.|: mean {scores.abs().mean():.3e} max {scores.abs().max():.3e}", flush=True)

    n_train_steps = args.steps if args.method == "mattr" else 0   # ixg is closed-form: no training
    for step in range(n_train_steps):
        pair, rej = sample_train_pair(ds, eval_keys)      # generator draw, eval keys rejected
        n_rejected += rej
        train_keys.add(_key(pair))
        tok = ds.tokenize_pair(pair, tokenizer, device)
        bp, cp, si = pos_map(tok)
        if not bp:
            continue
        k = sample_k(total, args.k_schedule)
        # canonical mask registry: "topk" = soft fwd+bwd, "hard_topk" = hard fwd + sigmoid-STE bwd
        mask = build_mask(scores, k, args.variant, T=args.T).mask.view(S, width)
        logits = run_intervened(model, tok, hook, mask, bp, cp, si)
        # denoising / sufficient (Iso): non-top-k patched to source, target is the CLEAN (base) label
        # canonical losses; corrupt_topk=False is the Iso direction. "ce" is bit-identical to the
        # previous hardcoded F.cross_entropy(logits, base_label).
        loss = attribution_loss(args.loss, logits.unsqueeze(0),
                                torch.tensor([tok.base_label_id], device=device),
                                torch.tensor([tok.src_label_id], device=device),
                                corrupt_topk=False)
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
        if step % 100 == 0:
            with torch.no_grad():                          # binary-ness probe: soft must be < 1.0
                m = mask.detach()
                mask_frac_binary.append(float(((m == 0) | (m == 1)).float().mean()))
            print(f"step {step}  loss {np.mean(losses[-100:]):.4f}  k~{k:.0f}  "
                  f"binary_frac={mask_frac_binary[-1]:.3f}", flush=True)
    train_time = time.time() - t0
    # hard guarantee, checked against what training ACTUALLY consumed (not just the design)
    overlap = train_keys & eval_keys
    assert not overlap, f"train/eval overlap = {len(overlap)} examples"
    split_info.update({"n_train_distinct": len(train_keys),
                       "n_rejected_eval_draws": n_rejected, "train_eval_overlap": 0})
    if args.method == "mattr":
        split_info["n_train_draws"] = len(losses)

    torch.save(scores.detach().cpu(), os.path.join(args.output, "scores.pt"))
    ks = [1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 2048, 4096]
    print("evaluating sufficiency curve...", flush=True)
    t1 = time.time()
    curve = evaluate(model, ds, tokenizer, hook, scores.detach(), device, ks, eval_pairs, T=args.T)
    eval_time = time.time() - t1
    sc = scores.detach()
    sc2 = sc.view(S, width)                                # [num_spans, d_sae(+1)]
    # top-50 (span, feature) pairs by score
    flat_top = torch.argsort(sc2[:, :sae.d_sae].flatten(), descending=True)[:50].cpu().tolist()
    top = [[i // sae.d_sae, i % sae.d_sae] for i in flat_top]   # [span, feature]
    err_info = None
    if args.error_mode == "node":
        err = sc2[:, sae.d_sae]                            # per-span error-node scores [S]
        err_info = {"error_node_scores_per_span": err.cpu().tolist(),
                    "max_feature_score": float(sc2[:, :sae.d_sae].max())}
        print(f"ERROR NODES per span: {[round(float(x), 3) for x in err]} "
              f"(max feat {err_info['max_feature_score']:.3f})", flush=True)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                         cwd=os.path.dirname(os.path.abspath(__file__)),
                                         stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        commit = None
    peak_mem = (torch.cuda.max_memory_allocated() / 2**30 if device == "cuda"
                else torch.mps.current_allocated_memory() / 2**30 if device == "mps" else None)
    metrics = {"learned": curve_metrics(curve, "learned"), "random": curve_metrics(curve, "random")}
    provenance = {
        "task": args.task, "method": args.method,
        "grad_loss": args.grad_loss if args.method == "ixg" else None,
        "grad_examples": args.grad_examples if args.method == "ixg" else None,
        "variant": args.variant if args.method == "mattr" else None,
        "k_schedule": args.k_schedule if args.method == "mattr" else None,
        "seed": args.seed, "model": args.model, "sae_repo": args.sae_repo, "sae_id": args.sae_id,
        "layer": args.layer, "d_sae": sae.d_sae, "num_spans": S, "width": width,
        "total_scores": total, "steps": args.steps, "lr": args.lr, "T": args.T,
        "error_mode": args.error_mode,
        "loss": f"iso/sufficient ({args.loss if args.method == 'mattr' else args.grad_loss})",
        "objective": args.loss if args.method == "mattr" else args.grad_loss,
        "eval_mask": "hard top-k (sigmoid_topk_hard), identical for every variant",
        "device": device, "git_commit": commit, "peak_mem_gib": peak_mem,
        "train_time_s": train_time, "eval_time_s": eval_time, **split_info,
    }
    json.dump({"args": vars(args), "provenance": provenance, "metrics": metrics,
               "losses": losses, "curve": curve, "top50_features": top,
               "mask_binary_frac": mask_frac_binary,
               "total_features": total, "error_node": err_info},
              open(os.path.join(args.output, "results.json"), "w"), indent=2)
    print("=== SUFFICIENCY CURVE (prob-diff base-src; higher=more sufficient) ===")
    for i, k in enumerate(curve["k"]):
        print(f"k={k:5d}  learned acc={curve['learned_acc'][i]:.3f} pd={curve['learned_probdiff'][i]:+.3f}"
              f"   random acc={curve['random_acc'][i]:.3f} pd={curve['random_probdiff'][i]:+.3f}", flush=True)
    m = metrics["learned"]
    tag = (f"{args.variant}/{args.k_schedule}/{args.loss}" if args.method == "mattr"
           else f"ixg/{args.grad_loss}/n{args.grad_examples}")
    print(f"=== {tag}/s{args.seed}  log-AUC={m['log_auc']:.4f} "
          f"linear-AUC={m['linear_auc']:.4f} plateau={m['plateau_acc']:.3f} "
          f"max={m['max_acc']:.3f} kstar_grid={m['kstar_grid']} "
          f"train={train_time:.0f}s eval={eval_time:.0f}s", flush=True)
    handle.remove()


if __name__ == "__main__":
    main()
