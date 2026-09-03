"""Optimizer x learning-rate x Adam-eps grid for MAttr on the ViT teaser, against Stepless IG.

The teaser (`vit_teaser_attr.py`) runs ONE MAttr configuration -- soft top-k, log-k, Adam at
lr 0.05 with the library-default eps 1e-8, the pre-2026-08-24 headline. Since then the paper's
headline optimizer became SGD, and at neuron scale Adam's eps was found to decide the ranking
outright (`scripts/submit_adam_eps_followup.sh`: at eps=1e-8 the update degenerates to
sign(g), the score becomes a signed step COUNT, and every trace of effect magnitude is divided
out). The ViT substrate is 196 units, three to four orders of magnitude smaller, so whether
either finding transfers is an empirical question this script answers. One process, model
loaded once, every arm on the same image / corruption family / explained scalar:

  adam    lr in --adam-lrs  x  eps in --adam-eps       (the paper's LR grid, the eps bracket)
  sgd     lr in --sgd-lrs                              (zero init, no momentum: LR-invariant
                                                        ranking up to the soft top-k's T)
  ig      Stepless IG, alpha ~ U(0,1) per draw, along the patch-embedding path from a freshly
          sampled corruption to the clean image; compute-matched at one forward+backward per
          step, same step count as MAttr. `learning_to_attribute.trainer.stepless_ig`, called
          in chunks so the running mean can be probed at the same cadence as MAttr's scores.

Every arm is probed every `--probe-every` steps with the hard top-k sufficiency AUC of its
current ranking (`make_auc_probe`, `--probe-draws` corruption draws), and scored at the end on
a held-out corruption seed with `--eval-draws` draws -- the protocol of `vit_teaser_faith.py`,
so the reference rankings in `--reference` (the teaser's KernelSHAP / AttnLRP / ...) are
re-scored here on the same footing rather than copied from `faithfulness.json`.

Output: `<out>/grid.npz` (per-arm `scores` [14,14], `curve` [10], `auc_trace` [n,2],
`loss_log`) and `<out>/grid.json` (per-arm AUC, timing, score-distribution diagnostics, the
resolved arguments). `plots/plot_vit_optim.py` reads them.

    sbatch -J vito42 vit_optim.sbatch --seed 42 --out results/vit_optim/seed42
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from learning_to_attribute.losses import attribution_loss   # noqa: E402
from learning_to_attribute.trainer import learn_scores, stepless_ig   # noqa: E402
from vit_teaser_attr import (CAT_IDX, DOG_IDX, FRACS, GRID, N_PATCH, _LOG,   # noqa: E402
                             forward_from_patches, load_model, make_auc_probe,
                             make_corrupt_image_sampler, make_target_fn)

REFERENCE_METHODS = ["mattr", "attnlrp", "smoothgrad", "gradattnroll", "kernelshap"]


def parse_floats(s):
    return [float(x) for x in s.replace(",", " ").split()]


def tag(x):
    """Learning rates as written: `0.05`, `1`, `100`."""
    return f"{x:g}"


def etag(x):
    """`1e-2` for 0.01 -- eps always in normalised scientific notation, one spelling per value
    (the convention of eval_sva.eps_tag and the results/adamsgd_mlp/A_eps dir names)."""
    m, e = f"{x:.0e}".split("e")
    return f"{m}e{int(e)}"


def score_stats(s):
    a = np.abs(np.asarray(s, dtype=np.float64))
    med = float(np.median(a))
    return {"abs_median": med,
            "abs_p99_over_p50": float(np.percentile(a, 99) / med) if med > 0 else float("nan"),
            "abs_max": float(a.max()), "n_zero": int((a == 0).sum())}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", default="assets/cat_dog.jpg")
    ap.add_argument("--out", default="results/vit_optim/seed42")
    ap.add_argument("--reference", default="results/vit_teaser_pixelate",
                    help="teaser run whose non-MAttr rankings are re-scored on this footing "
                         "('' to skip)")
    ap.add_argument("--target", default="logit_diff", choices=["logit_diff", "single"])
    ap.add_argument("--explain", default="dog", choices=["dog", "cat"])
    ap.add_argument("--baseline", default="pixelate",
                    choices=["pixelate", "shuffle", "solid", "blur", "mean", "black", "white"])
    ap.add_argument("--blur-sigma", type=float, default=48.0)
    ap.add_argument("--pixel-size", type=int, default=16)
    ap.add_argument("--fixed-corrupt", action="store_true")
    # the grid
    ap.add_argument("--arms", default="adam,sgd,ig",
                    help="comma-separated subset of {adam, sgd, ig}")
    ap.add_argument("--adam-lrs", default="0.005 0.01 0.05 0.1 0.3 1.0 3.0",
                    help="the paper's node-level LR grid (tabs/lr_sweep.tex) extended upward, "
                         "because at eps >= 1e-1 Adam is ~SGD at effective lr/eps and needs "
                         "the larger rates to be bracketed")
    ap.add_argument("--adam-eps", default="1e-8 1e-6 1e-4 1e-2 1e-1 1e0",
                    help="submit_adam_vs_sgd_mlp.sh arm A's bracket plus the follow-up's two "
                         "top-end points; 1e-8 is torch's default")
    ap.add_argument("--sgd-lrs", default="0.01 0.03 0.1 0.3 1.0 3.0 10 30 100",
                    help="brackets the node optimum (1.0), the edge one (3.0) and the "
                         "global-KL one (100)")
    ap.add_argument("--ig-schedules", default="uniform",
                    help="alpha schedules for stepless IG; 'uniform' is canonical")
    # MAttr knobs shared by every arm (the teaser's)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--T", type=float, default=0.5)
    ap.add_argument("--n-iters", type=int, default=30)
    ap.add_argument("--k-schedule", default="log")
    # probing / scoring
    ap.add_argument("--probe-every", type=int, default=25)
    ap.add_argument("--probe-draws", type=int, default=8)
    ap.add_argument("--eval-draws", type=int, default=16)
    ap.add_argument("--random-seeds", type=int, default=20,
                    help="random rankings averaged for the floor")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log-every", type=int, default=500)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    model, weights = load_model(device)
    categories = weights.meta["categories"]
    tf = weights.transforms()
    image = Image.open(args.image).convert("RGB")
    x = tf(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)[0]
    dog = max(DOG_IDX, key=lambda i: logits[i])
    cat = max(CAT_IDX, key=lambda i: logits[i])
    pos, neg = (dog, cat) if args.explain == "dog" else (cat, dog)
    print(f"explaining {categories[pos]} ({logits[pos]:.3f}) against "
          f"{categories[neg]} ({logits[neg]:.3f})", flush=True)
    target_fn = make_target_fn(args.target, pos, neg)
    loss_name = "logit" if args.target == "single" else "logit_diff"
    base_id = torch.tensor([pos], device=device)
    source_id = torch.tensor([neg], device=device)
    E_clean = model._process_input(x).detach()
    resample = args.baseline in ("shuffle", "pixelate", "solid") and not args.fixed_corrupt
    kw = dict(resample=resample, blur_sigma=args.blur_sigma, pixel_size=args.pixel_size)

    # one corruption stream for training (per arm, re-seeded so every arm sees the same
    # sequence), a different one for scoring -- vit_teaser_faith's seed + 1 convention
    def train_sampler():
        return make_corrupt_image_sampler(x, image, tf, args.baseline, seed=args.seed, **kw)

    eval_image = make_corrupt_image_sampler(x, image, tf, args.baseline, seed=args.seed + 1, **kw)
    total = N_PATCH

    def make_apply(sample_image):
        def apply_mask(m):
            mm = m.view(-1, N_PATCH, 1)
            E_corr = model._process_input(sample_image()).detach()
            return forward_from_patches(model, mm * E_clean + (1 - mm) * E_corr)
        return apply_mask

    eval_apply = make_apply(eval_image)
    logf = np.log(FRACS)
    auc_of = lambda y: float(np.trapezoid(y, logf) / (logf[-1] - logf[0]))   # noqa: E731

    def final_curve(scores):
        """Held-out sufficiency curve, `--eval-draws` draws, the faith script's protocol."""
        order = torch.as_tensor(np.asarray(scores).ravel(), device=device).argsort(descending=True)
        masks = torch.zeros(len(FRACS), total, device=device)
        for i, f in enumerate(FRACS):
            masks[i, order[:max(1, round(f * total))]] = 1.0
        with torch.no_grad():
            y = torch.stack([target_fn(eval_apply(masks)) for _ in range(args.eval_draws)])
        return y.mean(0).float().cpu().numpy()

    arrays, records = {}, {}

    def record(name, scores, trace, loss_log, seconds, **meta):
        curve = final_curve(scores)
        arrays[f"{name}/scores"] = np.asarray(scores, dtype=np.float32).reshape(GRID, GRID)
        arrays[f"{name}/curve"] = curve.astype(np.float32)
        arrays[f"{name}/auc_trace"] = np.asarray(trace, dtype=np.float32).reshape(-1, 2)
        arrays[f"{name}/loss_log"] = np.asarray(loss_log, dtype=np.float32)
        records[name] = {"auc": auc_of(curve), "seconds": seconds,
                         "final_probe_auc": float(trace[-1][1]) if len(trace) else None,
                         **score_stats(scores), **meta}
        print(f"[{name}] held-out AUC = {records[name]['auc']:.3f}  ({seconds:.1f}s)", flush=True)

    # ---------------------------------------------------------------- MAttr arms
    def run_mattr(name, optimizer, lr, eps):
        torch.manual_seed(args.seed)
        sample_image = train_sampler()
        apply_mask = make_apply(sample_image)

        def loss_fn(mask):
            return attribution_loss(loss_name, apply_mask(mask.unsqueeze(0)),
                                    base_id, source_id, corrupt_topk=False)

        torch.manual_seed(args.seed + 1000)      # probe draws; distinct from the k/corruption stream
        probe = make_auc_probe(lambda masks: target_fn(apply_mask(masks)), total, args.probe_draws)
        torch.manual_seed(args.seed)
        trace = []

        def on_step(step, k, loss, scores):
            if step % args.probe_every == 0 or step == args.steps - 1:
                trace.append((step, probe(scores)[0]))

        t0 = time.time()
        res = learn_scores(total, loss_fn, steps=args.steps, variant="topk",
                           k_schedule=args.k_schedule, lr=lr, optimizer=optimizer,
                           adam_eps=eps, T=args.T, n_iters=args.n_iters, device=device,
                           log_every=args.log_every, logger=_LOG, on_step=on_step)
        record(name, res.scores.numpy(), trace, res.loss_log, time.time() - t0,
               arm=optimizer, optimizer=optimizer, lr=lr, eps=eps if optimizer == "adam" else None)

    # ---------------------------------------------------------------- Stepless IG
    def run_ig(name, schedule):
        torch.manual_seed(args.seed)
        sample_image = train_sampler()
        losses = []

        def grad_fn(draw_alphas):
            a = draw_alphas(1).to(device).view(1, 1, 1)
            E_corr = model._process_input(sample_image()).detach()
            delta = E_clean - E_corr
            E = (E_corr + a * delta).requires_grad_(True)
            loss = attribution_loss(loss_name, forward_from_patches(model, E),
                                    base_id, source_id, corrupt_topk=False)
            (g,) = torch.autograd.grad(loss, E)
            lv = float(loss.detach())
            losses.append(lv)
            return (g * delta).sum(-1)[0], lv

        torch.manual_seed(args.seed + 1000)
        probe = make_auc_probe(lambda masks: target_fn(make_apply(sample_image)(masks)),
                               total, args.probe_draws)
        torch.manual_seed(args.seed)
        acc = torch.zeros(total, dtype=torch.float64)
        n, trace = 0, []
        t0 = time.time()
        # chunked so the running mean can be probed; each chunk is an independent MC estimate
        # of the same integral, so the size-weighted average of chunks IS the full-run mean
        for start in range(0, args.steps, args.probe_every):
            n_chunk = min(args.probe_every, args.steps - start)
            r = stepless_ig(total, grad_fn, steps=n_chunk, k_schedule=schedule)
            acc += r.scores.double() * n_chunk
            n += n_chunk
            trace.append((n - 1, probe((acc / n).float().to(device))[0]))
            if (n // args.probe_every) % (args.log_every // args.probe_every) == 0:
                print(f"IG {n:5d}/{args.steps}  loss={np.mean(losses[-n_chunk:]):.4f}  "
                      f"probe AUC={trace[-1][1]:.3f}", flush=True)
        record(name, (acc / n).numpy(), trace, losses, time.time() - t0,
               arm="ig", optimizer=None, lr=None, eps=None, schedule=schedule)

    arms = args.arms.split(",")
    if "adam" in arms:
        for eps in parse_floats(args.adam_eps):
            for lr in parse_floats(args.adam_lrs):
                run_mattr(f"adam_lr{tag(lr)}_eps{etag(eps)}", "adam", lr, eps)
    if "sgd" in arms:
        for lr in parse_floats(args.sgd_lrs):
            run_mattr(f"sgd_lr{tag(lr)}", "sgd", lr, None)
    if "ig" in arms:
        for sched in args.ig_schedules.split(","):
            run_ig("ig" if sched == "uniform" else f"ig_{sched}", sched)

    # ---------------------------------------------------------------- references + floor
    if args.reference:
        ref = np.load(Path(args.reference) / "attributions.npz")
        ref_meta = json.loads((Path(args.reference) / "meta.json").read_text())
        assert ref_meta["pos"]["index"] == pos and ref_meta["baseline"] == args.baseline, \
            "reference run explains a different contrast or corruption"
        for m in REFERENCE_METHODS:
            if f"{m}/patch" in ref.files:
                record(f"ref_{m}", ref[f"{m}/patch"], [], [], 0.0, arm="reference",
                       optimizer=None, lr=None, eps=None, method=m)
    rng = np.random.default_rng(0)
    rand = np.mean([final_curve(rng.standard_normal(total)) for _ in range(args.random_seeds)], 0)
    arrays["random/curve"] = rand.astype(np.float32)
    records["random"] = {"auc": auc_of(rand), "arm": "random"}
    with torch.no_grad():
        y_clean = float(target_fn(forward_from_patches(model, E_clean)))
        y_null = float(np.mean([float(target_fn(model(eval_image())))
                                for _ in range(args.eval_draws)]))

    np.savez_compressed(out / "grid.npz", **arrays)
    (out / "grid.json").write_text(json.dumps({
        "pos": {"index": pos, "label": categories[pos], "logit": float(logits[pos])},
        "neg": {"index": neg, "label": categories[neg], "logit": float(logits[neg])},
        "clean": y_clean, "corrupted": y_null, "fracs": FRACS, "total": total,
        "device": device, "args": vars(args), "arms": records,
    }, indent=2))
    print(f"wrote {out}/grid.npz and grid.json ({len(records)} arms)")
    for name, r in sorted(records.items(), key=lambda kv: -kv[1]["auc"]):
        print(f"  {name:<24} AUC {r['auc']:6.3f}")


if __name__ == "__main__":
    main()
