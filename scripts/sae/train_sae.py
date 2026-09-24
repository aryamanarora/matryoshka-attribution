"""Train a sigmoid top-k SAE (MAttr's mask + k schedule) on one residual-stream site.

Recipe = SAEBench's TopK baselines (read off the released 0108 configs,
adamkarvonen/saebench_pythia-160m-deduped_width-2pow12_date-0108 TopK trainer_*/config.json, and
adamkarvonen/dictionary_learning_demo): monology/pile-uncopyrighted, docs >= 4*ctx chars, ctx 1024
with the BOS position dropped, 244-context shuffled activation buffer, batch 2048, 500M tokens
(244140 steps), Adam lr 3e-4 betas (0.9, 0.999), 1000 warmup steps, linear decay over the last
20%, AuxK alpha 1/32 on latents dead for 10M tokens, grad clip 1.0, unit-norm decoder.

Changes vs that recipe, all deliberate:
  * THE METHOD: soft forward through ``sigmoid_topk`` (T=0.5, 50 bisection iterations -- MAttr's
    defaults in trainer.learn_scores) with k ~ sample_k(d_sae, "uniform") redrawn every step.
  * Activations are normalized to mean squared norm d_in (sae_lens' convention), not 1
    (dictionary_learning's): latents are then O(1), the scale MAttr's T=0.5 was tuned against. At
    unit norm every latent sits well inside one temperature of tau and the mask is flat.
    Adam makes the weights' updates scale-free; checkpoints are saved in raw space either way.
  * fp32 throughout (dictionary_learning autocasts to bf16): a bf16 bisection over 4096 terms
    would not resolve tau.
  * "Fired" (for AuxK's dead set) = in the hard top-round(k) AND positive. TopKTrainer counts the
    top-k indices alone, which at MAttr's large k would mark zero latents as firing.

Activations come from TransformerLens' ``from_pretrained_no_processing`` -- the model SAEBench's
core eval loads -- at ``blocks.{layer}.hook_resid_post``.

  UV_PROJECT_ENVIRONMENT=.venv-sae uv run --no-default-groups --group sae \
      python scripts/sae/train_sae.py --out results/sae/pythia160m_l8_4k_sigtopk_uniform
"""

import argparse
import json
import logging
import math
import time
from pathlib import Path

import torch
from datasets import load_dataset
from transformer_lens import HookedTransformer

from matryoshka_attribution import sample_k, wandb_util
from matryoshka_attribution.sae import SigmoidTopKSAE, auxk_loss, fve, geometric_median

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


def save(sae, scale, cfg, out, step):
    out.mkdir(parents=True, exist_ok=True)
    torch.save(sae.state_dict_raw_space(scale), out / "ae.pt")
    (out / "config.json").write_text(json.dumps({**cfg, "saved_at_step": step}, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="pythia-160m-deduped")
    p.add_argument("--layer", type=int, default=8)
    p.add_argument("--d-sae", type=int, default=4096)
    p.add_argument("--tokens", type=int, default=500_000_000)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--ctx-len", type=int, default=1024)
    p.add_argument("--llm-batch-size", type=int, default=32)
    p.add_argument("--buffer-ctxs", type=int, default=244)
    p.add_argument("--dataset", default="monology/pile-uncopyrighted")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup-steps", type=int, default=1000)
    p.add_argument("--decay-frac", type=float, default=0.8, help="linear decay starts here")
    p.add_argument("--auxk-alpha", type=float, default=1 / 32)
    p.add_argument("--dead-tokens", type=int, default=10_000_000)
    p.add_argument("--T", type=float, default=0.5)
    p.add_argument("--n-iters", type=int, default=50)
    p.add_argument("--k-schedule", default="uniform", choices=["uniform", "log", "log_both"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--diag-every", type=int, default=2000,
                   help="hard- vs soft-top-k FVE at fixed k on the current batch")
    p.add_argument("--diag-ks", type=int, nargs="+", default=[20, 80, 320])
    p.add_argument("--ckpt-every", type=int, default=20000)
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

    # Fixed normalizer from 100 batches (dictionary_learning's get_norm_factor, which also spends
    # them): x_train = x / scale has mean squared norm d_in.
    msn = torch.stack([buf.next().pow(2).sum(-1).mean() for _ in range(100)]).mean().item()
    scale = math.sqrt(msn / d_in)
    log.info("mean squared norm %.4g -> scale %.4g", msn, scale)

    sae = SigmoidTopKSAE(d_in, args.d_sae).to(device)
    opt = torch.optim.Adam(sae.parameters(), lr=args.lr, betas=(0.9, 0.999))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(steps, args.warmup_steps, decay_start))
    since_fired = torch.zeros(args.d_sae, dtype=torch.long, device=device)
    k_aux = d_in // 2                                   # TopKTrainer's top_k_aux heuristic

    cfg = {"trainer": {"trainer_class": "SigmoidTopKTrainer", "dict_class": "SigmoidTopKSAE",
                       "lm_name": args.model, "layer": args.layer, "hook_name": hook,
                       "activation_dim": d_in, "dict_size": args.d_sae, "steps": steps,
                       "decay_start": decay_start, "norm_scale": scale, **vars(args)}}
    run = wandb_util.init("sae", args.wandb_name or out.name, cfg["trainer"],
                          project=args.wandb_project, entity=args.wandb_entity,
                          enabled=args.wandb, group=f"{args.model}/L{args.layer}/{args.d_sae}")

    t0 = time.time()
    for step in range(steps):
        x = buf.next() / scale
        if step == 0:
            sae.b_dec.data = geometric_median(x)
        k = sample_k(args.d_sae, args.k_schedule)
        f, a, mask = sae.soft_encode(x, k, T=args.T, n_iters=args.n_iters)
        x_hat = sae.decode(f)
        e = x - x_hat
        l2 = e.pow(2).sum(dim=-1).mean()

        with torch.no_grad():
            top = a.topk(max(1, round(k)), dim=-1, sorted=False)
            fired = torch.zeros(args.d_sae, dtype=torch.bool, device=device)
            fired[top.indices[top.values > 0]] = True
            since_fired += len(x)
            since_fired[fired] = 0
            dead = since_fired >= args.dead_tokens
        aux = auxk_loss(sae, e.detach(), a, dead, k_aux) if args.auxk_alpha > 0 else l2.new_zeros(())
        loss = l2 + args.auxk_alpha * aux

        loss.backward()
        sae.remove_parallel_decoder_grad()
        torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        sched.step()
        sae.normalize_decoder()

        if run is not None and (step % args.log_every == 0 or step == steps - 1):
            with torch.no_grad():
                row = {"loss": loss.item(), "l2": l2.item(), "auxk": float(aux), "k": k,
                       "lr": sched.get_last_lr()[0], "fve_soft": fve(x, x_hat),
                       "dead": int(dead.sum()), "relu_l0": float((a > 0).sum(-1).float().mean()),
                       "mask_gt_half_l0": float((mask > 0.5).sum(-1).float().mean()),
                       "tokens": (step + 1) * args.batch_size,
                       "steps_per_s": (step + 1) / (time.time() - t0)}
                if step % args.diag_every == 0 or step == steps - 1:
                    for dk in args.diag_ks:
                        row[f"fve_hard_k{dk}"] = fve(x, sae.decode(sae.hard_encode(x, dk)))
                        fs, _, _ = sae.soft_encode(x, float(dk), T=args.T, n_iters=args.n_iters)
                        row[f"fve_soft_k{dk}"] = fve(x, sae.decode(fs))
            run.log(row, step=step)
        if step % 1000 == 0:
            log.info("step %d/%d loss %.4f l2 %.4f k %.0f dead %d (%.1f step/s)", step, steps,
                     loss.item(), l2.item(), k, int(dead.sum()), (step + 1) / (time.time() - t0))
        if step and step % args.ckpt_every == 0:
            save(sae, scale, cfg, out, step)

    save(sae, scale, cfg, out, steps)
    log.info("done in %.1f h -> %s", (time.time() - t0) / 3600, out)
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
