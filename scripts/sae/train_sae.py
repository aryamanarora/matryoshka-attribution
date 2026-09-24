"""Train a TopK SAE on one residual-stream site, with a hard or sigmoid top-k forward and a
fixed or random k. The only loss is reconstruction MSE.

The runs this exists for (scripts/sae/launch/submit_sae_pythia_sc.sh, RUN=A..E):
    A  --forward hard --k-schedule fixed --k 20      does our pipeline reproduce SAEBench's TopK k=20?
    B  --forward soft --k-schedule fixed --k 20      the sigmoid top-k operator alone
    C  --forward hard --k-schedule log --k-max 640   random k alone (one SAE for every L0)
    D  --forward soft --k-schedule log --k-max 640   both: sigmoid top-k + random k
    E  --forward ste  --k-schedule log --k-max 640   C with the sigmoid top-k as backward only
B and D showed the soft forward is a loose relaxation in an SAE (see sae.TopKSAE.soft_encode);
E keeps C's exact forward and changes only the gradient.

Data/optimizer follow SAEBench's TopK baselines (released 0108 configs of
adamkarvonen/saebench_pythia-160m-deduped_width-2pow12_date-0108; adamkarvonen/dictionary_learning_demo):
monology/pile-uncopyrighted docs >= 4*ctx chars, ctx 1024 with the BOS position dropped,
244-context shuffled activation buffer, batch 2048, Adam lr 3e-4 betas (0.9, 0.999), 1000 warmup
steps, linear decay over the last 20%, b_dec initialised at the first batch's geometric median,
unit-norm decoder rows. Left out on purpose: AuxK, dead-latent tracking, gradient clipping,
activation normalization, bf16 autocast. Activations come from TransformerLens'
``from_pretrained_no_processing`` (what SAEBench's core eval loads) at blocks.{layer}.hook_resid_post.

k is one draw per step, shared by the whole batch (as MAttr draws one k per step); ``log`` is
``schedules.sample_k(k_max, "log")``, i.e. log-uniform on [1, k_max]. The hard and ste forwards
use round(k). The soft forward (and ste's backward) is ``sigmoid_topk`` at T=0.5 with 50 bisection iterations, MAttr's
defaults (trainer.learn_scores).

  UV_PROJECT_ENVIRONMENT=.venv-sae uv run --no-default-groups --group sae \
      python scripts/sae/train_sae.py --forward hard --k-schedule fixed --k 20 --out results/sae/<tag>
"""

import argparse
import json
import logging
import time
from pathlib import Path

import torch
from datasets import load_dataset
from transformer_lens import HookedTransformer

from matryoshka_attribution import sample_k, wandb_util
from matryoshka_attribution.sae import TopKSAE, fve, geometric_median

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("train_sae")


def windows(tokenizer, dataset, ctx_len, docs_per_call=256):
    """Yield BOS + (ctx_len - 1) token windows, one per document long enough to fill it."""
    bos = tokenizer.bos_token_id
    texts = []
    for row in load_dataset(dataset, split="train", streaming=True):
        text = row["text"]
        if len(text) < 4 * ctx_len:                       # dictionary_learning_demo's min_chars
            continue
        texts.append(text[: 8 * ctx_len])                 # plenty for ctx_len tokens; skip the tail
        if len(texts) == docs_per_call:
            for ids in tokenizer(texts, add_special_tokens=False)["input_ids"]:
                if len(ids) >= ctx_len - 1:
                    yield [bos] + ids[: ctx_len - 1]
            texts = []


class ActivationBuffer:
    """Shuffled pool of ``n_ctxs`` contexts' activations; refilled once half has been served."""

    def __init__(self, model, hook, layer, window_iter, n_ctxs, ctx_len, llm_bs, out_bs):
        self.model, self.hook, self.layer, self.win = model, hook, layer, window_iter
        self.cap, self.llm_bs, self.out_bs = n_ctxs * (ctx_len - 1), llm_bs, out_bs
        self.acts = torch.empty(0, model.cfg.d_model, device=model.cfg.device)

    @torch.no_grad()
    def refill(self):
        parts, n = [self.acts], len(self.acts)
        while n < self.cap:
            toks = torch.tensor([next(self.win) for _ in range(self.llm_bs)], device=self.acts.device)
            _, cache = self.model.run_with_cache(toks, names_filter=self.hook,
                                                 stop_at_layer=self.layer + 1)
            a = cache[self.hook][:, 1:].reshape(-1, self.acts.shape[1])   # drop the BOS position
            parts.append(a)
            n += len(a)
        acts = torch.cat(parts)
        self.acts = acts[torch.randperm(len(acts), device=acts.device)]

    def next(self):
        if len(self.acts) < self.cap // 2:
            self.refill()
        out, self.acts = self.acts[: self.out_bs], self.acts[self.out_bs:]
        return out


def lr_lambda(steps, warmup, decay_start):
    def f(step):
        if step < warmup:
            return step / warmup
        if step >= decay_start:
            return (steps - step) / (steps - decay_start)
        return 1.0
    return f


def save(sae, cfg, out, step):
    out.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.detach().cpu() for k, v in sae.state_dict().items()}, out / "ae.pt")
    (out / "config.json").write_text(json.dumps({**cfg, "saved_at_step": step}, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--forward", required=True, choices=["hard", "soft", "ste"],
                   help="hard top-k / sigmoid top-k / hard forward with sigmoid top-k backward")
    p.add_argument("--k-schedule", default="log", choices=["fixed", "log"])
    p.add_argument("--k", type=int, default=20, help="k for --k-schedule fixed")
    p.add_argument("--k-max", type=int, default=640, help="upper end of the log-uniform k range")
    p.add_argument("--T", type=float, default=0.5)
    p.add_argument("--n-iters", type=int, default=50)
    p.add_argument("--model", default="pythia-160m-deduped")
    p.add_argument("--layer", type=int, default=8)
    p.add_argument("--d-sae", type=int, default=4096)
    p.add_argument("--tokens", type=int, default=100_000_000)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--ctx-len", type=int, default=1024)
    p.add_argument("--llm-batch-size", type=int, default=32)
    p.add_argument("--buffer-ctxs", type=int, default=244)
    p.add_argument("--dataset", default="monology/pile-uncopyrighted")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup-steps", type=int, default=1000)
    p.add_argument("--decay-frac", type=float, default=0.8, help="linear decay starts here")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--diag-every", type=int, default=1000,
                   help="hard- and soft-top-k FVE at fixed k on the current batch")
    p.add_argument("--diag-ks", type=int, nargs="+", default=[20, 80, 320])
    p.add_argument("--ckpt-every", type=int, default=10000)
    p.add_argument("--out", required=True)
    wandb_util.add_args(p)
    p.add_argument("--wandb-name", default=None)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda"
    out = Path(args.out)
    steps = args.tokens // args.batch_size
    decay_start = int(args.decay_frac * steps)

    model = HookedTransformer.from_pretrained_no_processing(args.model, device=device,
                                                            dtype=torch.float32)
    model.eval()
    hook = f"blocks.{args.layer}.hook_resid_post"
    d_in = model.cfg.d_model
    buf = ActivationBuffer(model, hook, args.layer,
                           windows(model.tokenizer, args.dataset, args.ctx_len),
                           args.buffer_ctxs, args.ctx_len, args.llm_batch_size, args.batch_size)

    sae = TopKSAE(d_in, args.d_sae).to(device)
    opt = torch.optim.Adam(sae.parameters(), lr=args.lr, betas=(0.9, 0.999))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(steps, args.warmup_steps, decay_start))

    cfg = {"trainer": {"trainer_class": "TopKSAETrainer", "lm_name": args.model,
                       "layer": args.layer, "hook_name": hook, "activation_dim": d_in,
                       "dict_size": args.d_sae, "steps": steps, "decay_start": decay_start,
                       **vars(args)}}
    run = wandb_util.init("sae", args.wandb_name or out.name, cfg["trainer"],
                          project=args.wandb_project, entity=args.wandb_entity,
                          enabled=args.wandb, group=f"{args.model}/L{args.layer}/{args.d_sae}")

    t0 = time.time()
    for step in range(steps):
        x = buf.next()
        if step == 0:
            sae.b_dec.data = geometric_median(x)
        k = args.k if args.k_schedule == "fixed" else sample_k(args.k_max, "log")
        if args.forward == "hard":
            f = sae.hard_encode(x, max(1, round(k)))
        elif args.forward == "soft":
            f, mask = sae.soft_encode(x, k, T=args.T, n_iters=args.n_iters)
        else:
            f, mask = sae.ste_encode(x, k, T=args.T, n_iters=args.n_iters)
        x_hat = sae.decode(f)
        loss = (x - x_hat).pow(2).sum(dim=-1).mean()

        loss.backward()
        sae.remove_parallel_decoder_grad()
        opt.step()
        opt.zero_grad(set_to_none=True)
        sched.step()
        sae.normalize_decoder()

        if run is not None and (step % args.log_every == 0 or step == steps - 1):
            with torch.no_grad():
                a = sae.acts(x)
                row = {"loss": loss.item(), "k": k, "lr": sched.get_last_lr()[0],
                       "fve_train": fve(x, x_hat),
                       "relu_l0": float((a > 0).sum(-1).float().mean()),
                       "tokens": (step + 1) * args.batch_size,
                       "steps_per_s": (step + 1) / (time.time() - t0)}
                if args.forward != "hard":             # the sigmoid mask (ste: backward only)
                    row["mask_gt_half_l0"] = float((mask > 0.5).sum(-1).float().mean())
                if step % args.diag_every == 0 or step == steps - 1:
                    for dk in args.diag_ks:
                        fh = sae.hard_encode(x, dk)
                        row[f"fve_hard_k{dk}"] = fve(x, sae.decode(fh))
                        row[f"fve_soft_k{dk}"] = fve(x, sae.decode(
                            sae.soft_encode(x, float(dk), T=args.T, n_iters=args.n_iters)[0]))
                        # distinct latents used by the hard top-k over this batch (of d_sae)
                        row[f"used_latents_k{dk}"] = int((fh > 0).any(dim=0).sum())
            run.log(row, step=step)
        if step % 1000 == 0:
            log.info("step %d/%d loss %.4f k %.0f (%.1f step/s)", step, steps, loss.item(), k,
                     (step + 1) / (time.time() - t0))
        if step and step % args.ckpt_every == 0:
            save(sae, cfg, out, step)

    save(sae, cfg, out, steps)
    log.info("done in %.1f h -> %s", (time.time() - t0) / 3600, out)
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
