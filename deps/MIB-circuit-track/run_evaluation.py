import os
import pickle
import torch
from functools import partial
import argparse

from transformer_lens import HookedTransformer
from huggingface_hub import hf_hub_download

from eap.graph import Graph
from MIB_circuit_track.metrics import get_metric
from MIB_circuit_track.utils import TASKS_TO_HF_NAMES, MODEL_NAME_TO_FULLNAME, COL_MAPPING
from MIB_circuit_track.dataset import HFEAPDataset
from MIB_circuit_track.evaluation import evaluate_area_under_curve, evaluate_area_under_roc
from run_attribution import load_interpbench_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=str, nargs='+', required=True)
    parser.add_argument("--tasks", type=str, nargs='+', required=True)
    parser.add_argument("--ablation", type=str, choices=['patching', 'zero', 'mean', 'mean-positional', 'optimal'], default='patching')
    parser.add_argument("--split", type=str, choices=['train', 'validation', 'test'], default='validation')
    parser.add_argument("--method", type=str, default=None, help="Method used to generate the circuit (only needed to infer circuit file name)")
    parser.add_argument("--level", type=str, choices=['edge', 'node', 'neuron'], default='edge')
    parser.add_argument("--absolute", action="store_true")
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--head", type=int, default=None)
    parser.add_argument("--circuit-dir", type=str, default='circuits')
    parser.add_argument("--circuit-files", type=str, nargs='+', default=None)
    parser.add_argument("--output-dir", type=str, default='results')
    args = parser.parse_args()
    
    apply_greedy =  args.method in ["information-flow-routes"]#add your own method to the list if necessary

    i = 0
    for model_name in args.models:
        if model_name in ("qwen2.5", "gemma2", "llama3"):
            model = HookedTransformer.from_pretrained(MODEL_NAME_TO_FULLNAME[model_name],
                                                    attn_implementation="eager", torch_dtype=torch.bfloat16)
        elif model_name == "interpbench":
            model = load_interpbench_model()
            reference_graph = Graph.from_json(hf_hub_download("mib-bench/interpbench", filename="interpbench_graph.json"))
        else:
            model = HookedTransformer.from_pretrained(MODEL_NAME_TO_FULLNAME[model_name])
        model.cfg.use_split_qkv_input = True
        model.cfg.use_attn_result = True
        model.cfg.use_hook_mlp_in = True
        model.cfg.ungroup_grouped_query_attention = True

        for task in args.tasks:
            if f"{task.replace('_', '-')}_{model_name}" not in COL_MAPPING:
                continue
            method_name_saveable = f"{args.method}_{args.ablation}_{args.level}"
            p = f"{args.circuit_dir}/{method_name_saveable}/{task.replace('_', '-')}_{model_name}/importances.json"

            if args.circuit_files is not None:
                p = args.circuit_files[i]
                i += 1

            print(f"Loading circuit from {p}")
            if p.endswith('.json'):
                graph = Graph.from_json(p)
            elif p.endswith('.pt'):
                graph = Graph.from_pt(p)
            else:
                raise ValueError(f"Invalid file extension: {p.suffix}")
            
            hf_task_name = f'mib-bench/{TASKS_TO_HF_NAMES[task]}'
            dataset = HFEAPDataset(hf_task_name, model.tokenizer, split=args.split, task=task, model_name=model_name)
            if args.head is not None:
                head = args.head
                if len(dataset) < head:
                    print(f"Warning: dataset has only {len(dataset)} examples, but head is set to {head}; using all examples.")
                    head = len(dataset)
                dataset.head(head)
            dataloader = dataset.to_dataloader(batch_size=args.batch_size)
            metric = get_metric('logit_diff', task, model.tokenizer, model)
            attribution_metric = partial(metric, mean=False, loss=False)
            
            if model_name == "interpbench":
                d = evaluate_area_under_roc(reference_graph, graph)
            else:
                eval_auc_outputs = evaluate_area_under_curve(
                    model, graph, dataloader, attribution_metric, level=args.level,
                    absolute=args.absolute,
                    apply_greedy=apply_greedy,
                )
                weighted_edge_counts, area_under, area_from_1, average, faithfulnesses, accuracies, acc_auc = eval_auc_outputs
                print(f"  CPR-AUC={area_under:.4f}  acc-AUC={acc_auc:.4f}  acc curve={[round(a,3) for a in accuracies]}")
                d = {
                    "weighted_edge_counts": weighted_edge_counts,
                    "area_under": area_under,
                    "area_from_1": area_from_1,
                    "average": average,
                    "faithfulnesses": faithfulnesses,
                    "accuracies": accuracies,
                    "acc_auc": acc_auc,
                }

            method_name_saveable = f"{args.method}_{args.ablation}_{args.level}"
            output_path = os.path.join(args.output_dir, method_name_saveable)
            os.makedirs(output_path, exist_ok=True)
            with open(f"{output_path}/{task.replace('_', '-')}_{model_name}_{args.split}_abs-{args.absolute}.pkl", 'wb') as f:
                pickle.dump(d, f)
