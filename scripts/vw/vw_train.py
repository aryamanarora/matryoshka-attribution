"""Train the note's 1L transformer (Turner, Wu & Batson 2026, Appendix > Training details).

    sbatch -J vw scripts/vw/launch/vw.sbatch scripts/vw/vw_train.py --lr 1e-3 --out results/vw/base

Everything quoted below is stated in the note; the LEARNING RATE AND SCHEDULE ARE NOT, and
are the only fitted knobs here (see `--lr`, and scripts/vw_lr_sweep.sh):

  "Training used Adam with β = (0.9, 0.95) and no weight decay, under bfloat16 autocast,
   with gradients globally clipped to 1.5x an exponential moving average (decay 0.95) of
   recent gradient norms.
   The model trained on ~9.8x10^7 unique tokens (95,464 unique sequences) over its 11,933
   steps in a single pass with no data repetition. Training loss fell from 8.93 (~ln 4096
   at initialization) to ~3.33 at the final step and was still decreasing when the step
   budget was exhausted."

95,464 / 11,933 = 8.0 exactly, so the batch is 8 sequences (8,192 tokens) and "single pass
with no data repetition" fixes the sampler: one shuffled permutation, consumed once.

The two published losses are the acceptance test, not decoration: `--lr` is chosen as the
value whose final train loss lands on ~3.33 with held-out loss ~3.38, and the run's
`log.json` records both so the choice is auditable.
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from vw_model import CTX, TinyLM, loss_from_logits  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "vw"


def load(tag):
    a = np.fromfile(DATA / f"{tag}.bin", dtype=np.uint16)
    return torch.from_numpy(a.reshape(-1, CTX).astype(np.int64))


@torch.no_grad()
def evaluate(model, val, batch, device, cap=None):
    model.eval()
    tot, n = 0.0, 0
    v = val[:cap] if cap else val
    for i in range(0, len(v), batch):
        b = v[i:i + batch].to(device)
        with torch.autocast("cuda", torch.bfloat16, enabled=device == "cuda"):
            tot += loss_from_logits(model(b), b).item() * len(b)
        n += len(b)
    model.train()
    return tot / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--steps", type=int, default=11933, help="the note's step count")
    ap.add_argument("--batch", type=int, default=8, help="95464 sequences / 11933 steps")
    ap.add_argument("--warmup", type=int, default=0, help="not in the note; 0 = as stated")
    ap.add_argument("--cosine", action="store_true", help="not in the note; off = as stated")
    # The note states no schedule for the TRANSFORMER, but states one exactly for the
    # transcoder in the very next paragraph -- "linearly decayed to zero over the final 20%".
    # `--decay-frac 0.2` applies that same house-style schedule here; 0 = as stated.
    ap.add_argument("--decay-frac", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--eval-seqs", type=int, default=512)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train, val = load("train"), load("val")
    model = TinyLM(seed=args.seed).to(device)
    tot, core = model.n_params()
    print(f"params {tot:,} ({core:,} excl. embed/unembed) | train {tuple(train.shape)} "
          f"val {tuple(val.shape)} | device {device}", flush=True)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)
    order = torch.randperm(len(train), generator=torch.Generator().manual_seed(args.seed))
    ema = None                       # EMA(0.95) of recent grad norms; clip at 1.5x it
    log, t0 = [], time.time()

    for step in range(args.steps):
        idx = order[(step * args.batch) % len(train):][:args.batch]
        if len(idx) < args.batch:    # single pass; the tail is short only if steps*batch > N
            idx = order[:args.batch]
        b = train[idx].to(device)
        if args.warmup or args.cosine or args.decay_frac:
            f = min(1.0, (step + 1) / max(args.warmup, 1))
            if args.cosine:
                f *= 0.5 * (1 + math.cos(math.pi * step / args.steps))
            if args.decay_frac:
                start = (1 - args.decay_frac) * args.steps
                f *= min(1.0, max(0.0, (args.steps - step) / (args.steps - start)))
            for g in opt.param_groups:
                g["lr"] = args.lr * f
        with torch.autocast("cuda", torch.bfloat16, enabled=device == "cuda"):
            loss = loss_from_logits(model(b), b)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(),
                                            1.5 * ema if ema is not None else float("inf"))
        ema = float(gn) if ema is None else 0.95 * ema + 0.05 * float(gn)
        opt.step()

        if step % 200 == 0 or step == args.steps - 1:
            rec = {"step": step, "train_loss": loss.item(), "gnorm": float(gn), "clip": 1.5 * ema}
            if step % args.eval_every == 0 or step == args.steps - 1:
                rec["val_loss"] = evaluate(model, val, 16, device, args.eval_seqs)
            log.append(rec)
            print(f"  {step:6d} loss {rec['train_loss']:.4f}"
                  + (f" val {rec['val_loss']:.4f}" if "val_loss" in rec else "")
                  + f" | {time.time() - t0:.0f}s", flush=True)

    final_val = evaluate(model, val, 16, device)
    # last 20 logged train losses, to compare with the note's "~3.33 at the final step"
    tail = float(np.mean([r["train_loss"] for r in log[-20:]]))
    torch.save({"model": model.state_dict(), "args": vars(args)}, args.out / "model.pt")
    (args.out / "log.json").write_text(json.dumps(
        {"args": {k: str(v) for k, v in vars(args).items()}, "log": log,
         "train_loss_tail20": tail, "val_loss_full": final_val,
         "note_train": 3.33, "note_test": 3.38, "params": [tot, core]}, indent=2))
    print(f"FINAL train(tail20) {tail:.4f} [note 3.33] | val {final_val:.4f} [note 3.38] "
          f"| {time.time() - t0:.0f}s -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
