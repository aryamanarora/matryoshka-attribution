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
import argparse, json, math, os
import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from learning_to_attribute.data.causalgym import CausalGymDataset
from learning_to_attribute.sigmoid_topk import sigmoid_topk, sigmoid_topk_hard


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


def sample_k(total, schedule="log"):
    if schedule == "log":
        return math.exp(math.log(total) * torch.rand(1).item())
    return 1 + (total - 1) * torch.rand(1).item()


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


def evaluate(model, ds, tokenizer, hook, scores, device, ks, n_eval=50, T=0.5, seed=123):
    """Sufficiency curve: prob-diff & accuracy vs #features-kept-clean (hard top-k)."""
    ds_eval = CausalGymDataset(ds.task_name, seed=seed)
    total = scores.numel()
    width = hook.sae.d_sae + (1 if hook.error_mode == "node" else 0)
    S = total // width
    rand_scores = torch.randn_like(scores)
    out = {"k": [], "learned_probdiff": [], "learned_acc": [], "random_probdiff": [], "random_acc": []}
    pairs = []
    for _ in range(n_eval):
        p = strip_bos(ds_eval.sample_pair())
        tk = ds_eval.tokenize_pair(p, tokenizer, device)
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
    ap.add_argument("--k-schedule", default="uniform")  # uniform > log in our results
    ap.add_argument("--n-eval", type=int, default=50)
    ap.add_argument("--error-mode", default="node", choices=["cf", "clean", "node"],
                    help="default node: error term is always a scored node (consistent w/ arith SAE)")
    ap.add_argument("--output", default="results/sae_npi_subj_relc")
    args = ap.parse_args()

    device = "cuda"
    os.makedirs(args.output, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16).to(device).eval()
    model.requires_grad_(False)
    sae = JumpReLUSAE(args.sae_repo, args.sae_id, device)
    ds = CausalGymDataset(args.task)
    print(f"task={args.task} num_spans={ds.num_spans} d_sae={sae.d_sae} layer={args.layer}", flush=True)

    hook = SAEIntervention(sae)
    hook.error_mode = args.error_mode
    handle = model.model.layers[args.layer].register_forward_hook(hook)

    S = ds.num_spans
    width = sae.d_sae + (1 if args.error_mode == "node" else 0)    # per-span width (+1 error node)
    total = S * width                                              # per-span (untied) scores
    scores = torch.zeros(total, device=device, requires_grad=True)
    opt = torch.optim.Adam([scores], lr=args.lr)
    print(f"per-span scores: {S} spans x {width} = {total}", flush=True)

    losses = []
    for step in range(args.steps):
        pair = strip_bos(ds.sample_pair())
        tok = ds.tokenize_pair(pair, tokenizer, device)
        bp, cp, si = pos_map(tok)
        if not bp:
            continue
        k = sample_k(total, args.k_schedule)
        mask = sigmoid_topk_hard(scores, k=k, T=args.T).view(S, width)   # hard fwd, soft bwd
        logits = run_intervened(model, tok, hook, mask, bp, cp, si)
        # denoising / sufficient: target is the CLEAN (base) label
        loss = F.cross_entropy(logits.unsqueeze(0), torch.tensor([tok.base_label_id], device=device))
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
        if step % 100 == 0:
            print(f"step {step}  loss {np.mean(losses[-100:]):.4f}  k~{k:.0f}", flush=True)

    torch.save(scores.detach().cpu(), os.path.join(args.output, "scores.pt"))
    ks = [1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 2048, 4096]
    print("evaluating sufficiency curve...", flush=True)
    curve = evaluate(model, ds, tokenizer, hook, scores.detach(), device, ks, n_eval=args.n_eval, T=args.T)
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
    json.dump({"args": vars(args), "losses": losses, "curve": curve, "top50_features": top,
               "total_features": total, "error_node": err_info},
              open(os.path.join(args.output, "results.json"), "w"), indent=2)
    print("=== SUFFICIENCY CURVE (prob-diff base-src; higher=more sufficient) ===")
    for i, k in enumerate(curve["k"]):
        print(f"k={k:5d}  learned acc={curve['learned_acc'][i]:.3f} pd={curve['learned_probdiff'][i]:+.3f}"
              f"   random acc={curve['random_acc'][i]:.3f} pd={curve['random_probdiff'][i]:+.3f}", flush=True)
    handle.remove()


if __name__ == "__main__":
    main()
