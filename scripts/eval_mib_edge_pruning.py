"""Edge Pruning (Bhaskar et al., 2024) baseline on the MIB circuit track.

Learns hard-concrete gate latents (log-alphas) with the original Edge-Pruning recipe
(KL to the unmasked model + annealed-target Lagrangian sparsity; see
learning_to_attribute/edge_pruning.py), then ranks units by the final log-alphas in the
standard MIB CPR eval. ``--level edge`` masks edges (the original method, same patching
environment as eval_mib_edge.py); ``--level node`` is the node-level equivalent — one
latent per attention head / MLP, gating the node's output between its clean and
corrupted activation (z*clean + (1-z)*corrupted).

Denoising-only (mask=1 keeps clean, 0 patches corrupted), i.e. the `sufficient`
intervention MIB CPR measures — Edge Pruning is inherently this intervention.

``--gate sigmoid`` swaps the mask parameterization for pyvene's SigmoidMaskIntervention
(deterministic sigmoid gate, annealed temperature) while keeping the same patching
environment, task loss and step count — so the two rows differ only in how the mask is
learned. By default it carries no sparsity penalty (pyvene's library adds none); ``--l1-coeff``
restores one, since pyvene's own tutorial for the class does. See
learning_to_attribute/edge_pruning.py:learn_scores_sigmoid_mask.

``--gate dcm`` is DCM (Prakash et al., ICLR 2024; the ``roonbug/belief_dynamics``
re-implementation): a raw box-constrained coefficient with no sigmoid, no temperature and
no lr schedule, driven to ``--target-density`` by a PID controller on the sparsity
coefficient. Unlike the other two it produces a SET rather than a ranking, so it is only
apples-to-apples with them at the pinned density. See
learning_to_attribute/edge_pruning.py:learn_scores_dcm.
"""

import argparse
import logging
import math
import pickle
import random
import sys
import time
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

MODEL_FULLNAMES = {
    "gpt2": "gpt2-small",
    "qwen2.5": "Qwen/Qwen2.5-0.5B",
    "gemma2": "google/gemma-2-2b",
    "llama3": "meta-llama/Llama-3.1-8B",
}

TASKS_TO_HF = {
    "ioi": "ioi",
    "mcqa": "copycolors_mcqa",
    "arithmetic_addition": "arithmetic_addition",
    "arithmetic_subtraction": "arithmetic_subtraction",
    "arc_easy": "arc_easy",
    "arc_challenge": "arc_challenge",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mib-path", type=str, default="MIB-circuit-track")
    parser.add_argument("--model", type=str, required=True, choices=list(MODEL_FULLNAMES.keys()))
    parser.add_argument("--task", type=str, required=True, choices=list(TASKS_TO_HF.keys()))
    parser.add_argument("--level", type=str, default="edge", choices=["edge", "node"])
    parser.add_argument("--gate", type=str, default="hard_concrete",
                        choices=["hard_concrete", "sigmoid", "dcm"],
                        help="Mask parameterization. hard_concrete = Edge Pruning (stochastic "
                             "concrete gates + Lagrangian L0). sigmoid = pyvene's "
                             "SigmoidMaskIntervention (deterministic sigmoid(mask/temp), "
                             "temperature annealed 50->0.1, NO sparsity term); with --gate "
                             "sigmoid every --target-sparsity/--reg-lr/--*-warmup-frac flag "
                             "is inert. dcm = DCM's clamped raw coefficient with a PID "
                             "controller on the sparsity term; see --target-density.")
    parser.add_argument("--steps", type=int, default=3000,
                        help="Training steps (Edge-Pruning default: 3000; one example/step)")
    parser.add_argument("--lr", type=float, default=None,
                        help="lr for the mask parameters (default: 0.8 for hard_concrete, "
                             "pyvene's 1e-3 for sigmoid)")
    parser.add_argument("--reg-lr", type=float, default=0.8,
                        help="AdamW lr for the Lagrange multipliers (ascended)")
    parser.add_argument("--l1-coeff", type=float, default=0.0,
                        help="--gate sigmoid only: weight on an L1 sparsity penalty. pyvene's "
                             "library adds none, but its own tutorial for this class trains it "
                             "with loss + 1.0*||mask||_1, so 0 is a choice, not a given. "
                             "Inert for --gate hard_concrete (the Lagrangian owns sparsity there).")
    parser.add_argument("--l1-target", type=str, default="gate", choices=["gate", "logit"],
                        help="What the L1 penalises. gate = coeff*z.mean() (L1 relaxation of "
                             "L0; normalised so one coeff transfers across models). logit = "
                             "coeff*||mask||_1, pyvene's tutorial term verbatim -- but since "
                             "logits init at 0 that pulls gates toward z=0.5, i.e. toward the "
                             "~50%% density the unpenalised runs already show. See "
                             "edge_pruning.py:learn_scores_sigmoid_mask.")
    parser.add_argument("--target-density", type=float, default=0.05,
                        help="--gate dcm only: the density the PID controller pins the "
                             "circuit to. DCM has no ranking (clamp piles scores on 0 and "
                             "1), so its number is only meaningful at this density.")
    parser.add_argument("--dcm-penalty", type=str, default="additive",
                        choices=["additive", "multiplicative"],
                        help="--gate dcm only: which published sparsity term. additive = "
                             "Prakash et al.'s mult*mean(mask) (default). multiplicative = "
                             "belief_dynamics' mult*task_loss.detach()*mean(mask), whose SIGN "
                             "follows the task loss -- valid with --loss kl, and it will "
                             "refuse to run under --loss logit_diff, where it inverts.")
    parser.add_argument("--dcm-init-mult", type=float, default=0.025,
                        help="--gate dcm only: initial sparsity coefficient handed to the "
                             "PID controller (belief_dynamics' README example value).")
    parser.add_argument("--dcm-ramp-frac", type=float, default=0.8,
                        help="--gate dcm only: fraction of training over which the pruned-"
                             "count setpoint ramps to the target; held constant after.")
    parser.add_argument("--pid-kp", type=float, default=0.1)
    parser.add_argument("--pid-ki", type=float, default=1e-3)
    parser.add_argument("--pid-kd", type=float, default=0.0,
                        help="--gate dcm only: PID gains on log(coefficient). Defaults are "
                             "belief_dynamics' (kd=0, i.e. a PI controller).")
    parser.add_argument("--target-sparsity", type=float, default=None,
                        help="Final sparsity target (default: 0.99 edge, 0.9 node)")
    parser.add_argument("--start-sparsity", type=float, default=0.0)
    parser.add_argument("--sparsity-warmup-frac", type=float, default=0.83,
                        help="Fraction of steps to anneal the sparsity target over "
                             "(2500/3000 in the Edge-Pruning IOI recipe)")
    parser.add_argument("--lr-warmup-frac", type=float, default=0.07,
                        help="Fraction of steps for linear lr warmup (200/3000 in theirs)")
    parser.add_argument("--warmup-type", type=str, default="linear",
                        choices=["linear", "logarithmic"])
    parser.add_argument("--loss", type=str, default="kl", choices=["kl", "logit_diff"],
                        help="Task loss: KL to the unmasked model at the answer position "
                             "(Edge Pruning's objective) or -logit_diff (MAttr's)")
    parser.add_argument("--include-input", action="store_true",
                        help="node level: also learn a gate for the input embedding node "
                             "(default: input always stays clean)")
    parser.add_argument("--split", type=str, default="validation",
                        help="Split to EVALUATE the circuit on (held out)")
    parser.add_argument("--train-split", type=str, default="train",
                        help="Split to TRAIN log-alphas on (must differ from --split)")
    parser.add_argument("--batch-size", type=int, default=20, help="Eval batch size")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-examples", type=int, default=None)
    parser.add_argument("--output", type=str, default=None,
                        help="Default: results/edge_pruning_<level>")
    parser.add_argument("--save-graph", type=str, default=None,
                        help="Write the scored circuit as a MIB graph.json here, so it can be "
                             "scored by MIB's run_evaluation.py (the same harness as every "
                             "other baseline in the tables). Default: <output>/graph_<task>_<model>.json")
    parser.add_argument("--no-save-graph", action="store_true",
                        help="Skip the graph.json dump")
    parser.add_argument("--skip-eval", action="store_true",
                        help="Train and dump the circuit only; leave scoring to run_evaluation.py")
    args = parser.parse_args()
    if args.l1_coeff and args.gate != "sigmoid":
        parser.error("--l1-coeff applies to --gate sigmoid only; hard_concrete's Lagrangian "
                     "already owns sparsity via --target-sparsity")
    if args.lr is None:
        # DCM's published lr, identical in Prakash et al. and belief_dynamics.
        args.lr = {"hard_concrete": 0.8, "sigmoid": 1e-3, "dcm": 1e-1}[args.gate]
    if args.target_sparsity is None:
        args.target_sparsity = 0.99 if args.level == "edge" else 0.9
    if args.output is None:
        args.output = f"results/edge_pruning_{args.level}"

    mib_path = Path(args.mib_path).resolve()
    sys.path.insert(0, str(mib_path))
    sys.path.insert(0, str(mib_path / "EAP-IG" / "src"))

    from transformer_lens import HookedTransformer
    from eap.graph import Graph
    from eap.utils import tokenize_plus
    from MIB_circuit_track.dataset import HFEAPDataset
    from MIB_circuit_track.metrics import get_metric
    from MIB_circuit_track.evaluation import evaluate_area_under_curve
    from einops import einsum

    # src/ layout: also works when the package is not pip-installed (e.g. under the MIB venv)
    sys.path.insert(0, str(Path(__file__).parent.parent))
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from learning_to_attribute.edge_pruning import (
        learn_scores_dcm, learn_scores_edge_pruning, learn_scores_sigmoid_mask)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load model (same config as eval_mib_edge.py so the patching environment matches)
    tl_name = MODEL_FULLNAMES[args.model]
    logger.info("Loading %s...", tl_name)
    if args.model in ("gemma2", "llama3", "qwen2.5"):
        model = HookedTransformer.from_pretrained(tl_name, attn_implementation="eager",
                                                   torch_dtype=torch.bfloat16)
    else:
        model = HookedTransformer.from_pretrained(tl_name)
    model.cfg.use_split_qkv_input = True
    model.cfg.use_attn_result = True
    model.cfg.use_hook_mlp_in = True
    model.cfg.ungroup_grouped_query_attention = True

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    d_model = model.cfg.d_model

    graph = Graph.from_model(model, node_scores=(args.level == "node"))
    n_real = int(graph.real_edge_mask.sum().item())
    logger.info("Graph: %d nodes, %d edges (%d real)", len(graph.nodes), len(graph.edges), n_real)

    # Edge level: differentiable scatter from the flat real-edge vector to the full
    # [n_forward, n_backward] adjacency (cumsum trick, as in eval_mib_edge.py)
    real_flat = graph.real_edge_mask.flatten().bool()
    cumsum = real_flat.float().cumsum(0).long() - 1
    cumsum = cumsum.clamp(min=0).to(device)
    real_flat_device = real_flat.float().to(device)

    # Load dataset
    hf_task = f"mib-bench/{TASKS_TO_HF[args.task]}"
    dataset = HFEAPDataset(hf_task, model.tokenizer, split=args.train_split,
                           task=args.task, model_name=args.model)
    logger.info("Loaded %d examples", len(dataset))

    # Source node info for hooks (input, a{l}.h{h}, m{l}) — shared by both levels
    source_hooks = []  # (hook_name, node_name, forward_index, is_attn)
    input_node = graph.nodes['input']
    source_hooks.append(('hook_embed', 'input', graph.forward_index(input_node), False))
    for l in range(n_layers):
        head0 = graph.nodes[f'a{l}.h0']
        head0_idx = graph.forward_index(head0, attn_slice=False)
        source_hooks.append((f'blocks.{l}.attn.hook_result', f'attn_{l}',
                            head0_idx, True))
        mlp_node = graph.nodes[f'm{l}']
        mlp_idx = graph.forward_index(mlp_node)
        source_hooks.append((f'blocks.{l}.hook_mlp_out', f'm{l}', mlp_idx, False))

    # Node level: which forward indices carry a learned gate
    input_fwd_idx = graph.forward_index(input_node)
    score_fwd_idxs = []
    if args.include_input:
        score_fwd_idxs.append(input_fwd_idx)
    for hname, nname, fidx, is_attn in source_hooks:
        if nname == 'input':
            continue
        if is_attn:
            score_fwd_idxs.extend(range(fidx, fidx + n_heads))
        else:
            score_fwd_idxs.append(fidx)
    score_fwd_idxs_t = torch.tensor(score_fwd_idxs, dtype=torch.long, device=device)

    total = n_real if args.level == "edge" else len(score_fwd_idxs)
    if args.gate == "sigmoid":
        logger.info("%s-level pyvene sigmoid mask: %d mask logits (no sparsity target)",
                    args.level, total)
    elif args.gate == "dcm":
        logger.info("%s-level DCM: %d clamped coefficients, PID-pinned to density %.4f "
                    "(%d of %d units)", args.level, total, args.target_density,
                    round(args.target_density * total), total)
    else:
        logger.info("%s-level Edge Pruning: %d log-alpha parameters, target sparsity %.4f",
                    args.level, total, args.target_sparsity)

    n_examples = len(dataset)

    def sample_pair():
        """One same-length clean/corrupted pair (None on mismatch -> skip step)."""
        idx = random.randint(0, n_examples - 1)
        clean, corrupted, labels = dataset[idx]
        clean_tokens, attention_mask, input_lengths, n_pos = tokenize_plus(model, [clean])
        corrupted_tokens, _, _, _ = tokenize_plus(model, [corrupted])
        if clean_tokens.shape[1] != corrupted_tokens.shape[1]:
            return None
        return clean_tokens, corrupted_tokens, attention_mask, labels

    def cache_corrupted(corrupted_tokens, attention_mask):
        """No-grad corrupted forward, caching every source output (per head for attn)."""
        corrupted_acts = {}

        def make_corrupted_hook(fwd_idx, is_attn):
            def hook(act, hook):
                if is_attn:
                    for h in range(n_heads):
                        corrupted_acts[fwd_idx + h] = act[:, :, h].detach()
                else:
                    corrupted_acts[fwd_idx] = act.detach()
            return hook

        hooks = [(hname, make_corrupted_hook(fidx, is_a))
                 for hname, nname, fidx, is_a in source_hooks]
        with torch.no_grad():
            model.run_with_hooks(corrupted_tokens, fwd_hooks=hooks,
                                 attention_mask=attention_mask)
        return corrupted_acts

    def task_loss(patched_logits, clean_tokens, attention_mask, labels):
        """KL(masked || full model) at the answer position, or -logit_diff."""
        correct_idx, incorrect_idx = labels[0], labels[1]
        if args.loss == "kl":
            with torch.no_grad():
                clean_logits = model(clean_tokens, attention_mask=attention_mask)
            return F.kl_div(
                F.log_softmax(patched_logits[0, -1].float(), dim=-1),
                F.log_softmax(clean_logits[0, -1].float(), dim=-1),
                reduction="sum", log_target=True)
        logit_diff = patched_logits[0, -1, correct_idx] - patched_logits[0, -1, incorrect_idx]
        return -logit_diff.float()

    # === Environment closures: apply the sampled gate vector z, return the task loss ===

    def edge_loss_fn(z):
        pair = sample_pair()
        if pair is None:
            return None
        clean_tokens, corrupted_tokens, attention_mask, labels = pair
        corrupted_acts = cache_corrupted(corrupted_tokens, attention_mask)

        # Expand flat gates to the adjacency; gated-off edges get the corrupted diff
        expanded = z[cumsum]
        full_mask = (expanded * real_flat_device).view(graph.n_forward, graph.n_backward)
        corruption_mask = (1 - full_mask).to(model.cfg.dtype)

        clean_acts = {}

        def make_clean_hook(fwd_idx, is_attn):
            def hook(act, hook):
                if is_attn:
                    for h in range(n_heads):
                        clean_acts[fwd_idx + h] = act[:, :, h]  # LIVE, grad-tracked
                else:
                    clean_acts[fwd_idx] = act
                return act
            return hook

        def make_dest_hook(dest_node, letter=None):
            prev_idx = graph.prev_index(dest_node)
            bwd_idx = graph.backward_index(dest_node, qkv=letter, attn_slice=True)
            weights = corruption_mask[:prev_idx, bwd_idx]  # [prev] or [prev, n_heads]

            def hook(activations, hook):
                diffs = []
                for src_i in range(prev_idx):
                    if src_i in corrupted_acts:
                        clean = clean_acts.get(src_i, torch.zeros_like(corrupted_acts[src_i]))
                        diffs.append(corrupted_acts[src_i] - clean)
                    else:
                        diffs.append(torch.zeros(1, activations.shape[1], d_model,
                                                 device=device, dtype=activations.dtype))
                diff_stack = torch.stack(diffs, dim=2)  # [batch, pos, prev, d_model]
                if weights.dim() == 1:
                    update = einsum(diff_stack, weights,
                                    'batch pos src hidden, src -> batch pos hidden')
                else:
                    update = einsum(diff_stack, weights,
                                    'batch pos src hidden, src heads -> batch pos heads hidden')
                return activations + update
            return hook

        clean_fwd_hooks = [(hname, make_clean_hook(fidx, is_a))
                           for hname, nname, fidx, is_a in source_hooks]
        dest_hooks = []
        for l in range(n_layers):
            for i, letter in enumerate('qkv'):
                node = graph.nodes[f'a{l}.h0']
                dest_hooks.append((node.qkv_inputs[i], make_dest_hook(node, letter=letter)))
            node = graph.nodes[f'm{l}']
            dest_hooks.append((node.in_hook, make_dest_hook(node)))
        node = graph.nodes['logits']
        dest_hooks.append((node.in_hook, make_dest_hook(node)))

        logits = model.run_with_hooks(clean_tokens, fwd_hooks=clean_fwd_hooks + dest_hooks,
                                      attention_mask=attention_mask)
        return task_loss(logits, clean_tokens, attention_mask, labels)

    def node_loss_fn(z):
        pair = sample_pair()
        if pair is None:
            return None
        clean_tokens, corrupted_tokens, attention_mask, labels = pair
        corrupted_acts = cache_corrupted(corrupted_tokens, attention_mask)

        # Gate vector over all forward nodes; unscored nodes (input by default) stay clean
        z_full = torch.ones(graph.n_forward, device=device)
        z_full[score_fwd_idxs_t] = z

        def make_node_hook(fwd_idx, is_attn):
            def hook(act, hook):
                if is_attn:
                    zs = z_full[fwd_idx:fwd_idx + n_heads].to(act.dtype).view(1, 1, -1, 1)
                    corr = torch.stack([corrupted_acts[fwd_idx + h]
                                        for h in range(n_heads)], dim=2)
                    return act * zs + corr * (1 - zs)
                z_i = z_full[fwd_idx].to(act.dtype)
                return act * z_i + corrupted_acts[fwd_idx] * (1 - z_i)
            return hook

        hooks = [(hname, make_node_hook(fidx, is_a))
                 for hname, nname, fidx, is_a in source_hooks]
        logits = model.run_with_hooks(clean_tokens, fwd_hooks=hooks,
                                      attention_mask=attention_mask)
        return task_loss(logits, clean_tokens, attention_mask, labels)

    loss_fn = edge_loss_fn if args.level == "edge" else node_loss_fn
    if args.gate == "sigmoid":
        logger.info("Training for %d steps (pyvene sigmoid mask, loss=%s, lr=%.3g, "
                    "temp 50->0.1, l1=%.3g on %s)...", args.steps, args.loss, args.lr,
                    args.l1_coeff, args.l1_target if args.l1_coeff else "n/a")
        result = learn_scores_sigmoid_mask(
            total, loss_fn, steps=args.steps, lr=args.lr,
            l1_coeff=args.l1_coeff, l1_target=args.l1_target,
            device=device, logger=logger, log_every=50,
        )
    elif args.gate == "dcm":
        logger.info("Training for %d steps (DCM, loss=%s, lr=%.3g, density %.4f, "
                    "penalty=%s, init_mult=%.3g, ramp %.2f, PID kp/ki/kd=%.3g/%.3g/%.3g)...",
                    args.steps, args.loss, args.lr, args.target_density, args.dcm_penalty,
                    args.dcm_init_mult, args.dcm_ramp_frac,
                    args.pid_kp, args.pid_ki, args.pid_kd)
        result = learn_scores_dcm(
            total, loss_fn, steps=args.steps, lr=args.lr,
            target_density=args.target_density, penalty_mode=args.dcm_penalty,
            init_mult=args.dcm_init_mult,
            ramp_frac=args.dcm_ramp_frac,
            pid_kp=args.pid_kp, pid_ki=args.pid_ki, pid_kd=args.pid_kd,
            device=device, logger=logger, log_every=50,
        )
    else:
        logger.info("Training for %d steps (loss=%s, lr=%.3g, reg_lr=%.3g, warmup_type=%s)...",
                    args.steps, args.loss, args.lr, args.reg_lr, args.warmup_type)
        result = learn_scores_edge_pruning(
            total, loss_fn, steps=args.steps,
            target_sparsity=args.target_sparsity, start_sparsity=args.start_sparsity,
            lr=args.lr, reg_lr=args.reg_lr,
            lr_warmup_frac=args.lr_warmup_frac, sparsity_warmup_frac=args.sparsity_warmup_frac,
            warmup_type=args.warmup_type,
            device=device, logger=logger, log_every=50,
        )
    scores = result.scores
    logger.info("Training complete in %.1fs", result.train_time_s)

    # === MIB Evaluation: rank by log-alpha ===
    logger.info("Running MIB evaluation (%s level)...", args.level)
    if args.level == "edge":
        graph.scores[:] = float('-inf')
        real_edges = graph.real_edge_mask.bool()
        graph.scores[real_edges] = scores.data.cpu()
    else:
        node_scores_tensor = torch.full((graph.n_forward,), float("nan"))
        node_scores_tensor[score_fwd_idxs_t.cpu()] = scores.data.cpu()
        if not args.include_input:
            # input is never gated during training -> always keep it in the circuit
            node_scores_tensor[input_fwd_idx] = scores.data.max().item() + 1.0
        graph.nodes_scores = node_scores_tensor

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    def save_scores(mib_results):
        torch.save({
            "scores": scores.data.cpu(),
            "args": vars(args),
            "loss_log": result.loss_log,
            "k_log": result.k_log,
            "train_time_s": result.train_time_s,
            "mib_results": mib_results,
        }, output_dir / f"{args.task}_{args.model}_scores.pt")

    if not args.no_save_graph:
        graph_path = Path(args.save_graph) if args.save_graph else (
            output_dir / f"graph_{args.task}_{args.model}.json")
        graph_path.parent.mkdir(parents=True, exist_ok=True)
        graph.to_json(str(graph_path))
        logger.info("Saved circuit to %s", graph_path)

    if args.skip_eval:
        save_scores(None)
        logger.info("Saved scores to %s (skipping in-process eval)", output_dir)
        return

    eval_dataset = HFEAPDataset(hf_task, model.tokenizer, split=args.split,
                                task=args.task, model_name=args.model)
    if args.eval_examples:
        eval_dataset.head(args.eval_examples)
    dataloader = eval_dataset.to_dataloader(batch_size=args.batch_size)
    metric = get_metric("logit_diff", args.task, model.tokenizer, model)
    attribution_metric = partial(metric, mean=False, loss=False)

    # cluster's patched MIB clone returns 7 values (+accuracies, acc_auc); stock returns 5
    ret = evaluate_area_under_curve(model, graph, dataloader, attribution_metric,
                                    level=args.level, absolute=False)
    if len(ret) == 7:
        weighted_edge_counts, area_under, area_from_1, average, faithfulnesses, accuracies, acc_auc = ret
    else:
        weighted_edge_counts, area_under, area_from_1, average, faithfulnesses = ret
        accuracies, acc_auc = None, None

    logger.info("MIB Results (%s level, Edge Pruning):", args.level)
    percentages = (0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0)
    for pct, faith in zip(percentages, faithfulnesses):
        logger.info("  %5.1f%% -> CPR=%.4f", pct * 100, faith)
    logger.info("  CPR AUC=%.4f  Avg CPR=%.4f", area_under, average)

    # Save
    mib_results = {
        "weighted_edge_counts": weighted_edge_counts,
        "area_under": area_under,
        "area_from_1": area_from_1,
        "average": average,
        "faithfulnesses": faithfulnesses,
        "accuracies": accuracies,
        "acc_auc": acc_auc,
    }
    with open(output_dir / f"{args.task}_{args.model}_{args.split}.pkl", "wb") as f:
        pickle.dump(mib_results, f)

    save_scores(mib_results)
    logger.info("Saved results to %s", output_dir)


if __name__ == "__main__":
    main()
