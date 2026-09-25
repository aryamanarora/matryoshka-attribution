""""Stepless IG" vs IxG at matched compute, read against the MC estimator's own noise floor.

THE CONTRAST. EAP-IG-inputs integrates the input-path gradient with a right-endpoint Riemann
sum over alpha = k/m, k=1..m. At m=1 that grid collapses to the single point alpha=1 -- the
clean input -- so one-step EAP-IG-inputs IS input x gradient. EAP-IG-inputs-mc draws
alpha ~ U(0,1) per example instead, which is unbiased for the integral at every m including
m=1, at identical cost: one forward+backward per batch either way.

    results/ig1_eval + ig1_accauc      m=1 grid, alpha=1        IxG            COMPUTE-MATCHED
    results/napig_mc_eval              m=1 mc,   alpha~U(0,1)   stepless IG    <- the arm
    results/napig_ref_eval/_accauc     m=5 grid                 5x cost        headroom marker
    results/napig10_eval               m=10 grid                10x cost
    results/napig30_eval               m=30 grid                30x cost

WHY THE SEED COLUMN IS NOT OPTIONAL. A one-draw-per-example estimator has real variance, so
"mc beats ig1 by 0.2 CPR-AUC" is meaningless until you know what two seeds of mc differ by.
Seeds 1 and 2 were run on the three cheapest cells for exactly this; every gap in the main
table is printed next to that floor, and a gap smaller than it is NOT a result. If the floor
is large, the honest conclusion is "more seeds", not "mc wins".

WHY THE FLOOR IS SMALLER THAN PER-BATCH SAMPLING WOULD GIVE. alpha is drawn per EXAMPLE
(shape [batch,1,1] broadcasting over n_pos and d_model), and scores sum over examples before
normalising, so a 1000-example cell averages 1000 independent alphas rather than 50 batches'
worth -- and the batch-size-1 llama3 cells are not penalised relative to batch-size-20 gpt2.

Both reported metrics, per the project rule: acc_auc (log-normalised, reads the sparse end)
and area_under (linear trapezoid, ~90% of it is k>=20% where methods tie). They can disagree
in sign; when they do, say so rather than picking the flattering one.

Run:  .venv/bin/python compare_stepless_ig.py
"""
import os
import pickle
import statistics

CELLS = [
    ("gpt2", "ioi"), ("qwen2.5", "ioi"), ("qwen2.5", "mcqa"),
    ("gemma2", "ioi"), ("gemma2", "mcqa"), ("gemma2", "arc_easy"),
    ("llama3", "ioi"), ("llama3", "mcqa"), ("llama3", "arithmetic_addition"),
    ("llama3", "arithmetic_subtraction"), ("llama3", "arc_easy"), ("llama3", "arc_challenge"),
]
# Cells with seed replicates. The first three attribute over 1000 examples; gemma2/arc_easy over
# 100. Since alpha is drawn per example the MC error scales like 1/sqrt(n_examples), so these two
# groups pin the two ends of the floor and the nine unreplicated cells are read against whichever
# end matches their own n. Quoting the 1000-example floor at a 100-example cell understates it by
# about sqrt(10) = 3.2x -- which is the difference between "MC wins" and "MC is noise".
CHEAP = [("gpt2", "ioi"), ("qwen2.5", "ioi"), ("qwen2.5", "mcqa"), ("gemma2", "arc_easy"),
         # Not cheap (43 min) and not in run_napig_mc.sh's CHEAP list -- replicated on purpose.
         # It is the one cell where MC exceeds even m=30 on area_under (1.80 v 1.52) while
         # sitting BELOW it on acc_auc, and it is the only llama3 floor we have: 6 of the 12
         # cells are llama3 at batch-size 1 with --head 200 eval, a regime none of the other
         # four replicates covers.
         ("llama3", "ioi")]
NEX = {("gpt2", "ioi"): 1000, ("qwen2.5", "ioi"): 1000, ("qwen2.5", "mcqa"): "full",
       ("gemma2", "arc_easy"): 100, ("llama3", "ioi"): 1000}
MC_DIR = "EAP-IG-inputs-mc_patching_node"
GRID_DIR = "EAP-IG-inputs_patching_node"
# dirs listed PRIMARY FIRST. The _accauc twins are pre-acc_auc-change reruns of the same
# circuits; their area_under matches the _eval pkl bit for bit, so falling back to one for
# acc_auc introduces no eval-sizing confound (verified on ioi/gpt2 for ig1 and napig_ref).
ARMS = [
    ("IxG (m=1 grid)",  ["ig1_eval", "ig1_accauc"],           GRID_DIR),
    ("MC (m=1 sample)", ["napig_mc_eval"],                    MC_DIR),
    ("IG m=5",          ["napig_ref_eval", "napig_ref_accauc"], GRID_DIR),
    ("IG m=10",         ["napig10_eval"],                     GRID_DIR),
    ("IG m=30",         ["napig30_eval"],                     GRID_DIR),
]
METRICS = ["acc_auc", "area_under"]


def load(odirs, mdir, model, task, metric):
    """First listed dir carrying this cell WITH the metric set; None if none does."""
    stask = task.replace("_", "-")
    for odir in odirs:
        p = f"results/{odir}/{mdir}/{stask}_{model}_validation_abs-False.pkl"
        if not os.path.exists(p):
            continue
        try:
            d = pickle.load(open(p, "rb"))
        except Exception:
            continue
        if d.get(metric) is not None:
            return d[metric]
    return None


def seed_spread(model, task, metric):
    """(values, spread) across mc seeds 0/1/2. Spread is max-min, not stdev: with three
    points stdev understates the range a single future seed could land in, and the range is
    what a one-seed gap has to clear."""
    vals = []
    for s in (0, 1, 2):
        d = "napig_mc_eval" if s == 0 else f"napig_mc_s{s}_eval"
        v = load([d], MC_DIR, model, task, metric)
        if v is not None:
            vals.append(v)
    return vals, (max(vals) - min(vals) if len(vals) > 1 else None)


def floors_by_model(metric):
    """{model: seed spread} from that model's replicated cell.

    PER MODEL, NOT ONE GLOBAL WORST CASE. The spreads are not the same order: on area_under
    llama3/ioi is 0.78 while gpt2/qwen2.5/gemma2 sit at 0.02-0.05, ~20x smaller. Taking the max
    and applying it everywhere would declare the gpt2 and qwen2.5 wins unresolved when their own
    replicates settle them 30x over; taking the min and applying it everywhere would call llama3
    noise a result. A cell is only ever read against a floor measured in ITS OWN regime -- same
    model, hence same batch size and same --head 200-vs-full eval.
    """
    out = {}
    for m, t in CHEAP:
        sp = seed_spread(m, t, metric)[1]
        if sp is not None:
            out[m] = max(out.get(m, 0.0), sp)
    return out


def main():
    for metric in METRICS:
        fl = floors_by_model(metric)
        # Guard the recovered-fraction denominator with the LARGEST per-model floor. This one
        # statistic is pooled across cells, so it has to survive the noisiest of them.
        floor = max(fl.values()) if fl else 0.0
        print("=" * 96)
        print(f"{metric}   (higher is better)     "
              + "  ".join(f"[{m} floor {v:.4f}]" for m, v in sorted(fl.items())))
        hdr = (f"{'cell':30}" + "".join(f"{n:>17}" for n, _, _ in ARMS)
               + f"{'MC - IxG':>12}" + f"{'vs floor':>10}")
        print(hdr)
        print("-" * len(hdr))
        gaps, recovered, cleared = [], [], []
        for model, task in CELLS:
            vals = [load(od, md, model, task, metric) for _, od, md in ARMS]
            ixg, mc, m5 = vals[0], vals[1], vals[2]
            gap = None if (ixg is None or mc is None) else mc - ixg
            if gap is not None:
                gaps.append(gap)
            # Fraction of the 5x-cost arm's advantage that the free arm buys back. The
            # denominator must clear the SEED FLOOR, not merely 1e-9: on the cells where m=5
            # collapses to IxG (ioi/qwen2.5, mcqa/qwen2.5, both 0.0502 vs 0.0502) the headroom
            # is a rounding difference and the ratio blows up to 4 digits, dragging a mean that
            # then reads as "MC recovers 420,000% of IG's advantage". Those cells have no
            # headroom to express a fraction OF -- report them as m=5-failures instead.
            if None not in (ixg, mc, m5) and m5 - ixg > floor:
                recovered.append((mc - ixg) / (m5 - ixg))
            row = f"{task + '/' + model:30}"
            row += "".join((f"{v:>17.4f}" if v is not None else f"{'--':>17}") for v in vals)
            row += (f"{gap:>+12.4f}" if gap is not None else f"{'--':>12}")
            # Each cell against ITS OWN model's floor. "ok" means the gap is bigger than what
            # two seeds of this estimator differ by in this regime; "NOISE" means it is not a
            # result no matter how large it looks next to another model's floor.
            mf = fl.get(model)
            if gap is None or mf is None:
                row += f"{'--':>10}"
            else:
                ok = abs(gap) > mf
                cleared.append(ok)
                row += f"{('ok' if ok else 'NOISE'):>10}"
            print(row)
        print("-" * len(hdr))
        # Two means per arm. The per-arm one answers "what does this arm score", the MATCHED one
        # is the only one that may be compared ACROSS arms: while an MC cell is still running,
        # its own mean silently omits that cell from itself but not from IG's, and if the missing
        # cell is a hard one (ioi/gemma2 is IG m=30's second-worst) the incomplete arm is
        # flattered by exactly the amount that matters.
        shared = [(m, t) for m, t in CELLS
                  if all(load(od, md, m, t, metric) is not None for _, od, md in ARMS)]
        for name, od, md in ARMS:
            done = [v for m, t in CELLS if (v := load(od, md, m, t, metric)) is not None]
            mean = statistics.fmean(done) if done else float("nan")
            sh = statistics.fmean([load(od, md, m, t, metric) for m, t in shared]) if shared \
                else float("nan")
            print(f"  {name:18} mean over {len(done):2}/12 = {mean:.4f}"
                  f"   | matched ({len(shared)} cells) = {sh:.4f}")
        if gaps:
            print(f"  MC - IxG: mean {statistics.fmean(gaps):+.4f}, "
                  f"wins {sum(g > 0 for g in gaps)}/{len(gaps)} cells")
        if recovered:
            print(f"  fraction of the m=5 (5x cost) advantage recovered for free: "
                  f"mean {statistics.fmean(recovered):+.3f} over {len(recovered)} cells")

        if cleared:
            print(f"  cells whose |MC - IxG| clears their OWN model's floor: "
                  f"{sum(cleared)}/{len(cleared)}")

        print(f"\n  MC SEED SPREAD (each model's floor, applied to that model's cells):")
        for model, task in CHEAP:
            vals, sp = seed_spread(model, task, metric)
            shown = " ".join(f"{v:.4f}" for v in vals) if vals else "--"
            n_cells = sum(1 for m, _ in CELLS if m == model)
            print(f"    {task + '/' + model:24} n={str(NEX[(model, task)]):5} seeds: {shown:28}"
                  + (f"spread {sp:.4f}" if sp is not None else "spread --")
                  + f"   -> floor for {n_cells} {model} cell(s)")
        print()


if __name__ == "__main__":
    main()
