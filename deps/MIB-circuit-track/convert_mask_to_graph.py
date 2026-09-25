from argparse import ArgumentParser
import json
import os
import torch
from transformer_lens import HookedTransformer
from eap.graph import Graph

parser = ArgumentParser()
parser.add_argument('--path', type=str, required=True, help='Path to the UGS outputs')
parser.add_argument('--task', type=str, required=True, help='Task that UGS was run on')
parser.add_argument('--ablation', type=str, required=True, help='Ablation that UGS was run with')
parser.add_argument('--model', type=str, required=True, choices=['gpt2-small', 'qwen'], help='Model that UGS was run on')
parser.add_argument('--lambdas', type=float, nargs='+', default=[1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7],
                    help='Regularization constants to convert (a missing snapshot is skipped)')

args = parser.parse_args()

path = args.path
task = args.task
ablation = args.ablation
model_str = args.model
name = f'ugs_mib_{model_str}'
lambdas = args.lambdas

if model_str == "gpt2-small":
    model_name = "gpt2-small"
elif model_str == "qwen":
    # base model, matching MODEL_NAME_TO_FULLNAME in MIB_circuit_track/utils.py
    model_name = "Qwen/Qwen2.5-0.5B"
else:
    raise Exception('Model name not defined')

# load an empty graph to use as a reference
model = HookedTransformer.from_pretrained(model_name)
model.cfg.use_split_qkv_input = True
model.cfg.use_attn_result = True
model.cfg.use_hook_mlp_in = True
model.cfg.ungroup_grouped_query_attention = True
g = Graph.from_model(model)
print(f'Loaded graph with {len(g.nodes)} nodes and {len(g.edges)} edges')
print(f'Graph has {g.real_edge_mask.sum()} real edges')

for lamb in lambdas:
    res_folder = f'{path}/{task}/{ablation}/{name}/{lamb}'
    snapshot_path = f'{res_folder}/snapshot.pth'

    if not os.path.exists(snapshot_path):
        print(f'No snapshot at {snapshot_path}, skipping')
        continue

    # load snapshot
    snapshot = torch.load(snapshot_path, map_location=torch.device('cpu'))

    sampling_params = {}
    thetas = {}
    total_param_count = 0
    for key, params in snapshot['pruner_dict'].items():
        layer_idx = int(key.split('.')[-1])
        edge_type = key.split('.')[-2]
        if edge_type not in sampling_params:
            sampling_params[edge_type] = {}
            thetas[edge_type] = {}
        sampling_params[edge_type][layer_idx] = params
        thetas[edge_type][layer_idx] = 1 / (1 + torch.exp(-params))
        total_param_count += params.numel()
    print(f'Loaded {total_param_count} parameters')

    # convert thetas to edge scores
    edges = {}
    for layer_idx in range(g.cfg['n_layers']):
        for i, letter in enumerate('qkv'):
            # curr_thetas_attn: n_heads x num_prev_layers (= layer_idx) x n_heads
            curr_thetas_attn = thetas['attn-attn'][layer_idx][i]
            # curr_thetas_mlp: n_heads x num_prev_layers (= layer_idx) + 1 
            curr_thetas_mlp = thetas['mlp-attn'][layer_idx][i]

            # add attn-attn edges
            for src_layer_idx in range(layer_idx):
                for src_head_idx in range(curr_thetas_attn.shape[2]):
                    for dest_head_idx in range(curr_thetas_attn.shape[0]):
                        edges[f'a{src_layer_idx}.h{src_head_idx}->a{layer_idx}.h{dest_head_idx}<{letter}>'] = curr_thetas_attn[dest_head_idx, src_layer_idx, src_head_idx].item()

            # add mlp-attn edges
            for src_layer_idx in range(layer_idx+1):
                for src_head_idx in range(curr_thetas_mlp.shape[0]): 
                    if src_layer_idx == 0:
                        src_str = 'input'
                    else:
                        src_str = f'm{src_layer_idx-1}'
                    edges[f'{src_str}->a{layer_idx}.h{src_head_idx}<{letter}>'] = curr_thetas_mlp[src_head_idx, src_layer_idx].item() 

        # curr_thetas_attn: num_prev_layers (= layer_idx) + 1 x n_heads
        curr_thetas_attn = thetas['attn-mlp'][layer_idx]
        # curr_thetas_mlp: num_prev_layers (= layer_idx) 
        curr_thetas_mlp = thetas['mlp-mlp'][layer_idx]

        # add attn-mlp edges
        for src_layer_idx in range(layer_idx+1):
            for src_head_idx in range(curr_thetas_attn.shape[1]):
                edges[f'a{src_layer_idx}.h{src_head_idx}->m{layer_idx}'] = curr_thetas_attn[src_layer_idx, src_head_idx].item()

        # add mlp-mlp edges
            if src_layer_idx == 0:
                src_str = 'input'
            else:
                src_str = f'm{src_layer_idx-1}'

            edges[f'{src_str}->m{layer_idx}'] = curr_thetas_mlp[src_layer_idx].item()

    curr_thetas_attn = thetas['attn-mlp'][layer_idx+1]
    curr_thetas_mlp = thetas['mlp-mlp'][layer_idx+1]

    for src_layer_idx in range(model.cfg.n_layers):
        for src_head_idx in range(curr_thetas_attn.shape[1]):
            edges[f'a{src_layer_idx}.h{src_head_idx}->logits'] = curr_thetas_attn[src_layer_idx, src_head_idx].item()

    for src_layer_idx in range(model.cfg.n_layers+1):
        if src_layer_idx == 0:
            src_str = 'input'
        else:
            src_str = f'm{src_layer_idx-1}'

        edges[f'{src_str}->logits'] = curr_thetas_mlp[src_layer_idx].item()

    print(f'Converted theta values to {len(edges)} edges')

    # check if there are missing edges
    missing = []
    excess = []
    for edge_name in g.edges.keys():
        if edge_name not in edges.keys():
            missing.append(edge_name)

    for edge_name in edges.keys():
        if edge_name not in g.edges.keys():
            excess.append(edge_name)

    assert len(missing) == 0, f'Missing edges: {missing}'
    assert len(excess) == 0, f'Excess edges: {excess}'
    
    # format and save graph
    edges_formatted = {k : { "score": v, "in_graph": False } for k, v in edges.items()}
    nodes_dict = { name: {"in_graph": False} for name in g.nodes.keys() }
    dict_to_store = {}
    cfg_dict = { "n_layers": model.cfg.n_layers, "n_heads": model.cfg.n_heads, "parallel_attn_mlp": False, "d_model": model.cfg.d_model } 
    dict_to_store["cfg"] = cfg_dict
    dict_to_store["edges"] = edges_formatted
    dict_to_store["nodes"] = nodes_dict
    
    dest_path = f'{res_folder}/graph.json'
    with open(dest_path, 'w') as f:
        json.dump(dict_to_store, f)