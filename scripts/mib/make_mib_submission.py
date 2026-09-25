"""Build a MIB circuit-track leaderboard submission from a node dir and an edge dir.

    uv run python scripts/mib/make_mib_submission.py \
        --node-dir results/test_node_topk_uniform_lr05 \
        --edge-dir results/test_edge_topk_uniform_lr05 \
        --out results/mib_submit

Output layout (what the leaderboard's validator and backend expect -- one folder per
task/model, ONE .json/.pt file inside, task names HYPHENATED):

    <out>/node/<task>_<model>/importances.json
    <out>/edge/<task>_<model>/importances.pt

Then:
  1. scripts/mib/launch/verify_mib_submission.sh <out> results/<node dir>   # CPU, ioi/gpt2
  2. hf upload <hf-user>/<repo> <out> . --repo-type model        # must be PUBLIC
  3. https://huggingface.co/spaces/mib-bench/leaderboard -> Submit, track "Circuit
     Localization", ONE submission per level with the DIRECTORY url:
       https://huggingface.co/<hf-user>/<repo>/tree/main/node   level "Node (submodule)"
       https://huggingface.co/<hf-user>/<repo>/tree/main/edge   level "Edge"
     The validator warns that interpbench is missing (we never ran it; >=2 tasks and >=2
     models is the requirement) -> "Proceed Anyway". arithmetic-addition was added for the
     submission only (scripts/mib/launch/submit_arith_add_headline.sh) and is not in the
     paper's 11-cell tables. Save the
     submission IDs; they are the only way to withdraw. Rate limit: 2 valid submissions /
     user / week, so node + edge is the whole week.

WHY THE TWO CONVERSIONS
- node: eval_mib.py's `<task>_<model>_importances.json` is already MIB's JSON schema, but it
  copies each node's score onto every one of its edges, which makes the llama3 files ~115 MB.
  The leaderboard validator warns on and skips files >50 MB. Node-level `apply_topn` reads
  only `nodes_scores` and derives edges from the selected nodes (EAP-IG graph.py), so the
  `edges` dict is dead weight: we keep `cfg` + `nodes` verbatim and write `edges: {}`.
  Verified lossless with MIB's run_evaluation.py (see verify_mib_submission.sh).
- edge: eval_mib_edge.py saves the raw score vector over `Graph.real_edge_mask` (its own
  training artifact), not a circuit file. We rebuild the graph from the model config (taken
  from the node JSON of the same cell -- no model load needed), scatter the vector onto the
  real edges exactly as eval_mib_edge.py does before ITS evaluation, and export with
  Graph.to_pt(). Non-real edges are -inf, matching the in-memory graph that produced the
  paper's edge numbers.
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import torch
from matryoshka_attribution.deps import find_mib_path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(find_mib_path() / "EAP-IG" / "src"))
from eap.graph import Graph  # noqa: E402

CELLS = [
    "ioi_gpt2", "ioi_qwen2.5", "ioi_gemma2", "ioi_llama3",
    "mcqa_qwen2.5", "mcqa_gemma2", "mcqa_llama3",
    "arithmetic_subtraction_llama3", "arithmetic_addition_llama3",
    "arc_easy_gemma2", "arc_easy_llama3", "arc_challenge_llama3",
]


def hf_name(cell: str) -> str:
    """results-dir cell name -> leaderboard folder name (task hyphenated, model as-is)."""
    task, model = cell.rsplit("_", 1)
    return task.replace("_", "-") + "_" + model


def build_node(node_dir: Path, cell: str, out: Path) -> tuple[int, int, float]:
    j = json.load(open(node_dir / f"{cell}_importances.json"))
    slim = {"cfg": j["cfg"], "nodes": j["nodes"], "edges": {}}
    d = out / "node" / hf_name(cell)
    d.mkdir(parents=True)
    json.dump(slim, open(d / "importances.json", "w"))
    n_scored = sum("score" in v for v in j["nodes"].values())
    return len(j["nodes"]), n_scored, os.path.getsize(d / "importances.json") / 1e6


def build_edge(node_dir: Path, edge_dir: Path, cell: str, out: Path) -> tuple[int, float]:
    cfg = json.load(open(node_dir / f"{cell}_importances.json"))["cfg"]
    scores = torch.load(edge_dir / f"{cell}_scores.pt", map_location="cpu",
                        weights_only=False)["scores"].float()
    g = Graph.from_model(cfg)
    real = g.real_edge_mask.bool()
    assert int(real.sum()) == scores.numel(), (cell, int(real.sum()), scores.numel())
    assert torch.isfinite(scores).all(), cell
    g.scores[:] = float("-inf")
    g.scores[real] = scores
    d = out / "edge" / hf_name(cell)
    d.mkdir(parents=True)
    g.to_pt(str(d / "importances.pt"))
    # round trip through the loader run_evaluation.py uses
    g2 = Graph.from_pt(str(d / "importances.pt"))
    assert torch.equal(g2.scores[real], scores) and not torch.isfinite(g2.scores[~real]).any()
    return scores.numel(), os.path.getsize(d / "importances.pt") / 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--node-dir", type=Path, required=True)
    ap.add_argument("--edge-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cells", nargs="+", default=CELLS)
    args = ap.parse_args()

    if args.out.exists():
        shutil.rmtree(args.out)
    for cell in args.cells:
        n_nodes, n_scored, mb_json = build_node(args.node_dir, cell, args.out)
        n_edges, mb_pt = build_edge(args.node_dir, args.edge_dir, cell, args.out)
        print(f"{hf_name(cell):30s} nodes={n_nodes:4d} scored={n_scored:4d} json={mb_json:5.2f}MB"
              f" | edges={n_edges:8d} pt={mb_pt:6.2f}MB")
        assert max(mb_json, mb_pt) < 50, "leaderboard validator skips files >50MB"
    print(f"-> {args.out}/node  {args.out}/edge")


if __name__ == "__main__":
    main()
