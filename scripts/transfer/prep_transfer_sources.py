"""Normalise every method's 13 llama3 score files into raw [1057] tensors for transfer evals.

The transfer pipeline (eval_transfer_mib.py, eval_sva.py --scores-from) accepts raw tensors in
the node+input layout (index 0 = input, then 32x32 heads (layer, head)-major, then 32 MLPs --
plot_task_corr_heatmap.names()). MAttr's files already are that (or a dict carrying it), but
the extra methods come in three shapes, so this script converts everything once, asserting the
recipe/config as it goes, and writes results/transfer_src/<method>/<task>.pt. Fail-loud: a
missing input score or a mismatched config kills the run rather than shipping a wrong ranking.

Methods (see METHODS): the paper-comparison set with full 13-cell llama3 coverage at
node+input. `adam`, `mc_ig`, `attnlrp` feed scripts/transfer/launch/submit_transfer.sh METHOD=...; `mattr`
(the headline, whose transfer round ran off the raw files) is normalised too so score
readers can treat all methods alike.

Run: python scripts/transfer/prep_transfer_sources.py
"""
import json
import sys
from pathlib import Path

import torch

from matryoshka_attribution.deps import mib_results_dir

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "plots"))
from plot_task_corr_heatmap import names   # canonical node order for the json -> vec map

R = Path("results")
R_MIB = mib_results_dir()
OUT = R / "transfer_src"
MIB_TASKS = ["ioi", "arithmetic_subtraction", "mcqa", "arc_easy", "arc_challenge"]
SVA_TASKS = ["simple", "nounpp", "rc", "within_rc", "addition", "months", "weekdays", "hours"]

# method -> (MIB-side kind+path template, SVA-side tag, config asserts for the SVA json)
METHODS = {
    # the headline itself, normalised here too so downstream readers (the faceted
    # transfer-vs-corr scatter) can treat every method's vectors uniformly. Its transfer
    # round ran off the raw files; results are identical (same tensors, same layout).
    "mattr": dict(mib=("ours", R / "softlog_sgd_lr_1.0" / "{task}_llama3_scores.pt"),
                  sva_tag="sufficient_topk_sgd_bs1",
                  want={"method": "mattr", "variant": "topk", "optimizer": "sgd",
                        "k_schedule": "log", "mode": "sufficient", "include_input": True}),
    # the "+ Adam" ablation: same recipe as the headline, optimizer swapped (CLAUDE.md table)
    "adam": dict(mib=("ours", R / "topklog_lr_0.05" / "{task}_llama3_scores.pt"),
                 sva_tag="sufficient_topk_adam_bs1",
                 want={"method": "mattr", "variant": "topk", "optimizer": "adam",
                       "k_schedule": "log", "mode": "sufficient", "include_input": True}),
    "mc_ig": dict(mib=("json", R_MIB / "napig_mc" / "EAP-IG-inputs-mc_patching_node"
                       / "{stask}_llama3" / "importances.json"),
                  sva_tag="mc_ig_m1_s42",
                  want={"method": "mc_ig", "ig_steps": 1, "seed": 42, "include_input": True}),
    "attnlrp": dict(mib=("json", R_MIB / "attnlrp" / "AttnLRP_patching_node"
                         / "{stask}_llama3" / "importances.json"),
                    sva_tag="attnlrp",
                    want={"method": "attnlrp", "include_input": True}),
    # MIB side of IxG is the `ig1` run: EAP-IG-inputs at m=1 degenerates to plain attribution
    # patching, i.e. grad x (delta act) = I x G at node level -- the same identification the
    # method heatmap (plot_method_corr_heatmap.py "I$\\times$G" -> ig1/...) and the test
    # table's I x G row already make. See make_mib_test_table.py's m=1 ladder note.
    "ixg": dict(mib=("json", R_MIB / "ig1" / "EAP-IG-inputs_patching_node"
                     / "{stask}_llama3" / "importances.json"),
                sva_tag="ixg", want={"method": "ixg", "include_input": True}),
}


def from_json(p):
    """MIB run_evaluation importances.json -> vector in names() order. The json must carry an
    input score (all three MIB-side methods here do); logits is a sink and is dropped."""
    d = json.load(open(p))
    nodes = d.get("nodes", d)
    sc = {n: i["score"] for n, i in nodes.items() if n != "logits" and "score" in i}
    nm = names()
    missing = [n for n in nm if n not in sc]
    if missing:
        raise SystemExit(f"{p}: missing scores for {len(missing)} nodes (e.g. {missing[:3]})")
    return torch.tensor([sc[n] for n in nm], dtype=torch.float32)


def main():
    for method, cfg in METHODS.items():
        outdir = OUT / method
        outdir.mkdir(parents=True, exist_ok=True)
        kind, tmpl = cfg["mib"] or (None, None)
        for task in MIB_TASKS if kind else []:
            p = Path(str(tmpl).format(task=task, stask=task.replace("_", "-")))
            if not p.exists():
                raise SystemExit(f"{method}/{task}: missing {p}")
            if kind == "ours":
                sd = torch.load(p, map_location="cpu")
                want_opt = cfg["want"]["optimizer"]
                assert sd["args"]["optimizer"] == want_opt and sd["args"]["k_schedule"] == "log", \
                    f"{p}: not the +{want_opt} recipe"
                vec = sd["scores"].float().flatten()
            else:
                vec = from_json(p)
            assert vec.numel() == 1057, (p, vec.numel())
            torch.save(vec, outdir / f"{task}.pt")
        for task in SVA_TASKS:
            j = R / "sva_sweep_input" / f"{task}_llama3_node_{cfg['sva_tag']}.json"
            if not j.exists():
                raise SystemExit(f"{method}/{task}: missing {j}")
            d = json.load(open(j))
            # Older runs predate the out["config"] record; their top level still carries
            # variant/mode/optimizer/k_schedule (and total, which is 1057 only at node+input).
            # Check each wanted key wherever it exists; a key present NOWHERE is only a
            # failure if the top level knows the concept (method/include_input aren't there --
            # for those the filename tag + total check are the guarantee, as in
            # make_fingerprint_tables).
            c = {**d, **d.get("config", {})}
            bad = {k: (c[k], v) for k, v in cfg["want"].items() if k in c and c[k] != v}
            if bad:
                raise SystemExit(f"{j}: config mismatch {bad}")
            if d.get("total") != 1057:
                raise SystemExit(f"{j}: total={d.get('total')}, not the node+input substrate")
            vec = torch.load(j.with_suffix(".scores.pt")).float().flatten()
            assert vec.numel() == 1057, (j, vec.numel())
            torch.save(vec, outdir / f"{task}.pt")
        print(f"{method}: {len(list(outdir.glob('*.pt')))} sources -> {outdir}")


if __name__ == "__main__":
    main()
