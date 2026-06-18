"""Single-layer DAS-64 *sufficient* baseline on gemma-2-2b, for apples-to-apples comparison
with the SAE-feature attribution (same layer, same loop, same denoising intervention).

Denoising (sufficient) intervention at layer L, mirroring RotateLayer.intervene(sufficient=False):
    out = a_cf + ((base@W − cf@W) ⊙ mask) @ Wᵀ        (W: [d_model, das_dim], orthogonal, learned)
top-k subspace dims held clean, complement + null-space set to cf. Train CE(logits, base_label).
"""
import argparse, json, math, os, sys
import numpy as np, torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from learning_to_attribute.data.causalgym import CausalGymDataset
from learning_to_attribute.sigmoid_topk import sigmoid_topk_hard
from learning_to_attribute.sigmoid_das import make_rotate_layer


def sample_k(total, schedule="log"):
    if schedule == "log":
        return math.exp(math.log(total) * torch.rand(1).item())
    return 1 + (total - 1) * torch.rand(1).item()

def strip_bos(p):
    p.base_spans[0] = p.base_spans[0].replace("<|endoftext|>", "")
    p.src_spans[0] = p.src_spans[0].replace("<|endoftext|>", "")
    return p

def pos_map(tok):
    bp, cp = [], []
    for i in range(tok.num_spans):
        ba, sa = tok.base_alignment[i], tok.src_alignment[i]
        for j, b in enumerate(ba):
            s = sa[min(j, len(sa) - 1)] if sa else b
            if b == 0:
                continue
            bp.append(b); cp.append(s)
    return bp, cp


class DASHook:
    def __init__(self, R):
        self.R = R; self.mode = "off"; self.cf_act = None
        self.base_pos = self.cf_pos = None; self.mask = None

    def __call__(self, module, inp, out):
        hs = out[0] if isinstance(out, tuple) else out
        if self.mode == "cache":
            self.cf_act = hs.detach(); return out
        if self.mode == "intervene":
            bp = torch.tensor(self.base_pos, device=hs.device)
            cp = torch.tensor(self.cf_pos, device=hs.device)
            base_sel = hs[0, bp].float().detach()
            cf_sel = self.cf_act[0, cp].float().detach()
            new = self.R.intervene(base_sel, cf_sel, self.mask.unsqueeze(0), sufficient=False).to(hs.dtype)
            hs = hs.clone(); hs[0, bp] = new
            return (hs,) + tuple(out[1:]) if isinstance(out, tuple) else hs
        return out


def run_intervened(model, tok, hook, mask, bp, cp):
    hook.mode = "cache"
    with torch.no_grad():
        model(tok.src_input_ids)
    hook.mode, hook.mask, hook.base_pos, hook.cf_pos = "intervene", mask, bp, cp
    lg = model(tok.base_input_ids).logits[0, -1].float()
    hook.mode = "off"
    return lg


def evaluate(model, ds, tokenizer, hook, scores, device, ks, n_eval, T, seed=123):
    de = CausalGymDataset(ds.task_name, seed=seed)
    rand = torch.randn_like(scores)
    out = {"k": [], "learned_acc": [], "learned_probdiff": [], "random_acc": []}
    pairs = []
    for _ in range(n_eval):
        tk = de.tokenize_pair(strip_bos(de.sample_pair()), tokenizer, device)
        bp, cp = pos_map(tk)
        if bp: pairs.append((tk, bp, cp))
    with torch.no_grad():
        for k in ks:
            for tag, sc in [("learned", scores), ("random", rand)]:
                mask = sigmoid_topk_hard(sc, k=float(k), T=T)
                accs, pds = [], []
                for tk, bp, cp in pairs:
                    lg = run_intervened(model, tk, hook, mask, bp, cp)
                    accs.append(float(lg[tk.base_label_id] > lg[tk.src_label_id]))
                    lp = F.log_softmax(lg, -1); pds.append((lp[tk.base_label_id] - lp[tk.src_label_id]).item())
                out[f"{tag}_acc"].append(float(np.mean(accs)))
                if tag == "learned": out["learned_probdiff"].append(float(np.mean(pds)))
            out["k"].append(k)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-2-2b")
    ap.add_argument("--task", default="syntaxgym/npi_ever_subj-relc")
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--das-dim", type=int, default=64)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--lr-rotation", type=float, default=0.001)
    ap.add_argument("--T", type=float, default=0.5)
    ap.add_argument("--n-eval", type=int, default=80)
    ap.add_argument("--output", default="results/das64_npi_subj_relc")
    args = ap.parse_args()
    device = "cuda"; os.makedirs(args.output, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16).to(device).eval()
    model.requires_grad_(False)
    d_model = model.config.hidden_size
    R = make_rotate_layer(d_model, args.das_dim).to(device)
    ds = CausalGymDataset(args.task)
    hook = DASHook(R); handle = model.model.layers[args.layer].register_forward_hook(hook)
    total = args.das_dim
    scores = torch.zeros(total, device=device, requires_grad=True)
    opt = torch.optim.Adam([{"params": [scores], "lr": args.lr},
                            {"params": list(R.parameters()), "lr": args.lr_rotation}])
    print(f"DAS-{total} single-layer L{args.layer} task={args.task}", flush=True)
    losses = []
    for step in range(args.steps):
        tok = ds.tokenize_pair(strip_bos(ds.sample_pair()), tokenizer, device)
        bp, cp = pos_map(tok)
        if not bp: continue
        mask = sigmoid_topk_hard(scores, k=sample_k(total, "uniform"), T=args.T)  # uniform > log
        lg = run_intervened(model, tok, hook, mask, bp, cp)
        loss = F.cross_entropy(lg.unsqueeze(0), torch.tensor([tok.base_label_id], device=device))
        opt.zero_grad(); loss.backward(); opt.step(); losses.append(loss.item())
        if step % 100 == 0: print(f"step {step} loss {np.mean(losses[-100:]):.4f}", flush=True)
    ks = [1, 2, 4, 8, 12, 16, 24, 32, 48, 64]
    ks = [k for k in ks if k <= total]
    curve = evaluate(model, ds, tokenizer, hook, scores.detach(), device, ks, args.n_eval, args.T)
    json.dump({"args": vars(args), "losses": losses, "curve": curve}, open(f"{args.output}/results.json", "w"), indent=2)
    print("=== DAS-%d SUFFICIENCY ===" % total)
    for i, k in enumerate(curve["k"]):
        print(f"k={k:3d} learned acc={curve['learned_acc'][i]:.3f} pd={curve['learned_probdiff'][i]:+.3f} random acc={curve['random_acc'][i]:.3f}", flush=True)
    handle.remove()


if __name__ == "__main__":
    main()
