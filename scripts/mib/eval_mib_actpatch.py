"""Single-node activation-patching baseline for the MIB circuit track (node level).

No learning: the score of every node is its own causal effect under an interchange
intervention, averaged over training pairs. Both directions are computed in the same pass, since
the second forward per node is the only extra cost:

  denoise  score_i = mean[ LD(input clean, all CF, i clean) - LD(input clean, all CF) ]  (sufficiency)
  noise    score_i = mean[ LD(all clean) - LD(all clean, node i CF) ]    (necessity of i alone)

where LD is the logit difference correct - incorrect at the last position, exactly the training
loss of eval_mib.py. Positive = important in both conventions, which is what MIB's denoising CPR
eval (absolute=False, keep the top of the ranking) expects. The node layout, the hooker and the
MIB evaluation are eval_mib.py's, so the outputs (<task>_<model>_{validation,test}.pkl,
_scores.pt, _importances.json) drop into make_mib_table.py like any MAttr dir.

The mask is one vector over nodes shared by the whole batch (LlamaAttributionHooks broadcasts
it), so batching is over examples and nodes are looped: 2 + 2 * N_nodes forwards per batch. At
gpt2 (157 nodes) / qwen2.5-0.5B (361 nodes) that is minutes for --n-examples 200.

  uv run python scripts/mib/eval_mib_actpatch.py --model gpt2 --task ioi \
      --n-examples 200 --batch-size 20 --split validation --output results/mib_node_actpatch
  -> results/mib_node_actpatch_denoise/ and results/mib_node_actpatch_noise/
"""

import argparse
import json
import logging
import math
import pickle
import sys
import time
from functools import partial
from pathlib import Path

import torch

from eval_mib import MODEL_FULLNAMES, MODEL_TL_NAMES, TASKS_TO_HF, get_hooks_class
from learning_to_attribute import CFActivationCache
from learning_to_attribute.deps import find_mib_path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def compute_scores(args, device):
    """Phase 1: per-node single-node patching effects on the train split. Returns (den, noi)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from MIB_circuit_track.dataset import HFEAPDataset
    import transformers as _tf

    hf_model_name = MODEL_FULLNAMES[args.model]
    tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    _dtype_kw = "dtype" if int(_tf.__version__.split(".")[0]) >= 5 else "torch_dtype"
    hf_model = AutoModelForCausalLM.from_pretrained(
        hf_model_name,
        device_map="auto" if args.model in ("gemma2", "llama3") else None,
        **{_dtype_kw: torch.bfloat16
           if args.model in ("gemma2", "llama3", "qwen2.5") else torch.float32},
    )
    if args.model not in ("gemma2", "llama3"):
        hf_model = hf_model.to(device)
    hf_model.eval()
    for p in hf_model.parameters():
        p.requires_grad_(False)

    hf_task_name = f"mib-bench/{TASKS_TO_HF[args.task]}"
    dataset = HFEAPDataset(hf_task_name, tokenizer, split=args.train_split,
                           task=args.task, model_name=args.model)
    logger.info("Loaded %d TRAIN examples from %s (%s)", len(dataset), args.task, args.train_split)

    HooksCls = get_hooks_class(hf_model)
    # sufficient=False is the legacy flag sense: mask=1 stays CLEAN, mask=0 gets the CF.
    hooker = HooksCls(hf_model, "node", seq_len=1, sufficient=False, include_input=True)
    total = hooker.total
    logger.info("Node scores: %s", hooker.describe())
    hooker.register_hooks()
    pad_id = tokenizer.pad_token_id

    def collate(seqs):
        lens = torch.tensor([s.shape[0] for s in seqs])
        P = int(lens.max())
        ids = torch.full((len(seqs), P), pad_id, dtype=torch.long)
        for b, s in enumerate(seqs):
            ids[b, : s.shape[0]] = s
        attn = (torch.arange(P)[None] < lens[:, None]).long()
        return ids.to(device), attn.to(device), lens

    # Deterministic: walk the train split in order, keep same-length pairs (the training loop's
    # rejection predicate), group by length so a batch shares one padded width.
    by_len = {}
    n_kept = 0
    for i in range(len(dataset)):
        if n_kept >= args.n_examples:
            break
        clean, corrupted, labels = dataset[i]
        c_ids = tokenizer(clean, return_tensors="pt").input_ids[0]
        s_ids = tokenizer(corrupted, return_tensors="pt").input_ids[0]
        if c_ids.shape[0] != s_ids.shape[0]:
            continue
        by_len.setdefault(c_ids.shape[0], []).append((c_ids, s_ids, labels[0], labels[1], corrupted))
        n_kept += 1
    batches = []
    for L, items in sorted(by_len.items()):
        for j in range(0, len(items), args.batch_size):
            batches.append(items[j:j + args.batch_size])
    logger.info("%d same-length pairs in %d batches", n_kept, len(batches))

    cf_cache = CFActivationCache(hooker, max_gb=args.cf_cache_gb, logger=logger)
    ones = torch.ones(total, device=device)
    zeros = torch.zeros(total, device=device)
    e_in = torch.zeros(total, device=device); e_in[0] = 1.0   # input node (include_input=True)
    den_sum = torch.zeros(total, dtype=torch.float64)
    noi_sum = torch.zeros(total, dtype=torch.float64)
    n_seen = 0
    t0 = time.time()

    @torch.no_grad()
    def logit_diff(base_ids, base_attn, last_pos, correct_t, incorrect_t, mask):
        hooker.mask = mask
        logits = hf_model(base_ids, attention_mask=base_attn).logits
        last = logits[torch.arange(base_ids.shape[0], device=device), last_pos].float()
        return last.gather(1, correct_t[:, None])[:, 0] - last.gather(1, incorrect_t[:, None])[:, 0]

    for bi, picks in enumerate(batches):
        base_ids, base_attn, lens = collate([p[0] for p in picks])
        src_ids, _, src_lens = collate([p[1] for p in picks])
        last_pos = (lens - 1).to(device)
        cf_cache.prepare([p[4] for p in picks], src_ids, src_lens.tolist())
        correct_t = torch.tensor([p[2] for p in picks], device=device)
        incorrect_t = torch.tensor([p[3] for p in picks], device=device)
        ld = partial(logit_diff, base_ids, base_attn, last_pos, correct_t, incorrect_t)

        ld_clean = ld(ones)
        # Denoising baseline keeps the INPUT clean (mask e_0) with every component CF, as MIB's
        # eval does (the input node is always in the circuit). With the input CF as well, every
        # node's recomputed value already IS the source value, so restoring one changes nothing.
        ld_corr = ld(e_in)
        ld_allcf = ld(zeros)
        for i in range(total):
            e = torch.zeros(total, device=device)
            e[i] = 1.0
            if i == 0:
                den_sum[i] += (ld_corr - ld_allcf).sum().item()          # restore the input alone
            else:
                den_sum[i] += (ld(e_in + e) - ld_corr).sum().item()      # restore i alone (input clean)
            noi_sum[i] += (ld_clean - ld(ones - e)).sum().item()         # corrupt i alone
        n_seen += len(picks)
        logger.info("batch %d/%d (%d ex, len %d): clean LD %.3f corr LD %.3f  [%.0fs]",
                    bi + 1, len(batches), len(picks), int(lens[0]),
                    ld_clean.mean().item(), ld_corr.mean().item(), time.time() - t0)

    hooker.remove_hooks()
    hooker.mask = None
    dims = {"num_layers": hooker.num_layers, "num_heads": hooker.num_heads, "total": total}
    del hf_model
    torch.cuda.empty_cache()
    return (den_sum / n_seen).float(), (noi_sum / n_seen).float(), dims, n_seen


def mib_eval_and_save(args, scores, dims, out_dir, tl_model, direction, n_seen):
    """Phase 2, copied from eval_mib.py so the pkl/scores.pt/importances.json match its format."""
    from eap.graph import Graph
    from MIB_circuit_track.metrics import get_metric
    from MIB_circuit_track.dataset import HFEAPDataset
    from MIB_circuit_track.evaluation import evaluate_area_under_curve

    n_layers, n_heads = dims["num_layers"], dims["num_heads"]
    off = 1
    input_score_val = scores[0].item()
    attn_scores = scores[off:off + n_layers * n_heads].view(n_layers, n_heads).cpu()
    mlp_scores = scores[off + n_layers * n_heads:].cpu()

    graph = Graph.from_model(tl_model, node_scores=True)
    node_scores_tensor = torch.full((graph.n_forward,), float("nan"))
    for name, node in graph.nodes.items():
        if name == "logits":
            continue
        idx = graph.forward_index(node, attn_slice=False)
        if idx >= graph.n_forward:
            continue
        if name == "input":
            node_scores_tensor[idx] = input_score_val
        elif name.startswith("a"):
            parts = name.split(".")
            node_scores_tensor[idx] = attn_scores[int(parts[0][1:]), int(parts[1][1:])].item()
        elif name.startswith("m"):
            node_scores_tensor[idx] = mlp_scores[int(name[1:])].item()
    graph.nodes_scores = node_scores_tensor

    hf_task_name = f"mib-bench/{TASKS_TO_HF[args.task]}"
    eval_dataset = HFEAPDataset(hf_task_name, tl_model.tokenizer, split=args.split,
                                task=args.task, model_name=args.model)
    if args.eval_examples:
        eval_dataset.head(args.eval_examples)
        logger.info("Capped eval set to %d examples", len(eval_dataset))
    dataloader = eval_dataset.to_dataloader(batch_size=args.eval_batch_size)
    metric = get_metric("logit_diff", args.task, tl_model.tokenizer, tl_model)
    attribution_metric = partial(metric, mean=False, loss=False)

    logger.info("[%s] Running MIB evaluation (node level)...", direction)
    weighted_edge_counts, area_under, area_from_1, average, faithfulnesses, accuracies, acc_auc = \
        evaluate_area_under_curve(tl_model, graph, dataloader, attribution_metric,
                                  level="node", absolute=False)
    percentages = (0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0)
    for pct, faith in zip(percentages, faithfulnesses):
        logger.info("  %5.1f%% -> CPR=%.4f  CMD=%.4f", pct * 100, faith, abs(1 - faith))
    logger.info("[%s] CPR AUC=%.4f  CMD AUC=%.4f  Avg CPR=%.4f  acc-AUC=%.4f",
                direction, area_under, area_from_1, average, acc_auc)

    out_dir.mkdir(parents=True, exist_ok=True)
    mib_results = {
        "weighted_edge_counts": weighted_edge_counts, "area_under": area_under,
        "area_from_1": area_from_1, "average": average, "faithfulnesses": faithfulnesses,
        "accuracies": accuracies, "acc_auc": acc_auc,
    }
    with open(out_dir / f"{args.task}_{args.model}_{args.split}.pkl", "wb") as f:
        pickle.dump(mib_results, f)
    saved_args = dict(vars(args), include_input=True, direction=direction, n_seen=n_seen)
    torch.save({
        "scores": scores.cpu(), "attn_scores": attn_scores, "mlp_scores": mlp_scores,
        "args": saved_args, "loss_log": [], "train_time_s": 0.0, "mib_results": mib_results,
    }, out_dir / f"{args.task}_{args.model}_scores.pt")

    nodes_dict = {}
    for name in graph.nodes:
        if name == "logits":
            nodes_dict[name] = {"in_graph": True}
        elif name == "input":
            nodes_dict[name] = {"in_graph": True, "score": input_score_val}
        elif name.startswith("a"):
            parts = name.split(".")
            nodes_dict[name] = {"in_graph": False,
                                "score": attn_scores[int(parts[0][1:]), int(parts[1][1:])].item()}
        elif name.startswith("m"):
            nodes_dict[name] = {"in_graph": False, "score": mlp_scores[int(name[1:])].item()}
    edges_dict = {}
    node_names = list(graph.nodes.keys())
    for edge_name in graph.edges:
        src, rest = edge_name.split("->")
        dst = rest.split("<")[0]
        src_s = node_scores_tensor[node_names.index(src)].item() if src in graph.nodes and node_names.index(src) < len(node_scores_tensor) else 0
        dst_s = node_scores_tensor[node_names.index(dst)].item() if dst in graph.nodes and node_names.index(dst) < len(node_scores_tensor) else 0
        s = min(src_s, dst_s) if not (math.isnan(src_s) or math.isnan(dst_s)) else max(src_s, dst_s)
        edges_dict[edge_name] = {"score": 0.0 if math.isnan(s) else s, "in_graph": False}
    cfg_dict = {"n_layers": tl_model.cfg.n_layers, "n_heads": tl_model.cfg.n_heads,
                "parallel_attn_mlp": False, "d_model": tl_model.cfg.d_model}
    with open(out_dir / f"{args.task}_{args.model}_importances.json", "w") as f:
        json.dump({"cfg": cfg_dict, "nodes": nodes_dict, "edges": edges_dict}, f)
    logger.info("[%s] Saved to %s", direction, out_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODEL_FULLNAMES))
    ap.add_argument("--task", required=True, choices=list(TASKS_TO_HF))
    ap.add_argument("--n-examples", type=int, default=200,
                    help="train pairs to average the patching effect over")
    ap.add_argument("--batch-size", type=int, default=20, help="pairs per forward (phase 1)")
    ap.add_argument("--eval-batch-size", type=int, default=20, help="MIB eval batch (phase 2)")
    ap.add_argument("--train-split", default="train")
    ap.add_argument("--split", default="validation")
    ap.add_argument("--eval-examples", type=int, default=None)
    ap.add_argument("--cf-cache-gb", type=float, default=4.0)
    ap.add_argument("--directions", default="denoise,noise")
    ap.add_argument("--mib-path", default=None)
    ap.add_argument("--output", default="results/mib_node_actpatch",
                    help="prefix; each direction is written to <output>_<direction>")
    args = ap.parse_args()

    mib_path = find_mib_path(args.mib_path)
    sys.path.insert(0, str(mib_path))
    sys.path.insert(0, str(mib_path / "EAP-IG" / "src"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    den, noi, dims, n_seen = compute_scores(args, device)
    for name, sc in (("denoise", den), ("noise", noi)):
        top = torch.topk(sc, 5)
        logger.info("%s top-5 idx %s val %s", name, top.indices.tolist(),
                    [round(v, 3) for v in top.values.tolist()])

    from transformer_lens import HookedTransformer
    tl_name = MODEL_TL_NAMES[args.model]
    if args.model in ("gemma2", "llama3", "qwen2.5"):
        tl_model = HookedTransformer.from_pretrained(tl_name, attn_implementation="eager",
                                                    torch_dtype=torch.bfloat16)
    else:
        tl_model = HookedTransformer.from_pretrained(tl_name)
    tl_model.cfg.use_split_qkv_input = True
    tl_model.cfg.use_attn_result = True
    tl_model.cfg.use_hook_mlp_in = True
    tl_model.cfg.ungroup_grouped_query_attention = True

    for direction in args.directions.split(","):
        scores = {"denoise": den, "noise": noi}[direction]
        mib_eval_and_save(args, scores, dims, Path(f"{args.output}_{direction}"), tl_model,
                          direction, n_seen)


if __name__ == "__main__":
    main()
