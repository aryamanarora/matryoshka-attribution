"""Top-k Jaccard overlap and Spearman correlation between methods.

Usage:
    uv run python scripts/compare_topk_overlap.py
"""

import json, torch, numpy as np
from scipy.stats import spearmanr
from pathlib import Path

R = Path('results')

COLUMNS_NODE = [
    ('ioi', 'gpt2'), ('ioi', 'qwen2.5'), ('ioi', 'gemma2'),
    ('arithmetic_subtraction', 'llama3'),
    ('mcqa', 'qwen2.5'), ('mcqa', 'gemma2'), ('mcqa', 'llama3'),
    ('arc_easy', 'gemma2'), ('arc_easy', 'llama3'), ('arc_challenge', 'llama3'),
]
COLUMNS_EDGE = [('ioi', 'gpt2'), ('ioi', 'qwen2.5'), ('ioi', 'gemma2'), ('mcqa', 'qwen2.5'), ('mcqa', 'gemma2')]

def load_node(path):
    d = json.load(open(path))
    nodes = d.get('nodes', d)
    return {n: info['score'] for n, info in nodes.items() if n not in ('input','logits')}

def topk_analysis(ours_vals, bl_vals, ks):
    n = len(ours_vals)
    ours_order = np.argsort(-ours_vals)
    bl_order = np.argsort(-bl_vals)
    results = []
    for k_frac in ks:
        k = max(1, int(n * k_frac))
        ours_topk = set(ours_order[:k].tolist())
        bl_topk = set(bl_order[:k].tolist())
        jaccard = len(ours_topk & bl_topk) / len(ours_topk | bl_topk)
        union = sorted(ours_topk | bl_topk)
        rho = spearmanr(ours_vals[union], bl_vals[union])[0] if len(union) >= 4 else float('nan')
        results.append((k_frac, k, jaccard, rho))
    return results

ks = [0.01, 0.02, 0.05, 0.10, 0.20]

print('NODE LEVEL: Ours vs NAP-IG (repro)')
header = '{:25s}  {:>5s}'.format('task/model', 'n')
for k in ks:
    header += '  J@{:<4s} rho@{:<4s}'.format('{:.0%}'.format(k), '{:.0%}'.format(k))
print(header)
print('-' * 140)
for task, model in COLUMNS_NODE:
    ours_path = R / 'mib_node_hard_topk' / '{}_{}_importances.json'.format(task, model)
    stask = task.replace('_', '-')
    bl_path = R / 'napig_repro' / 'EAP-IG-inputs_patching_node' / '{}_{}'.format(stask, model) / 'importances.json'
    if not ours_path.exists() or not bl_path.exists(): continue
    ours = load_node(ours_path); bl = load_node(bl_path)
    common = sorted(set(ours) & set(bl))
    ov = np.array([ours[n] for n in common]); bv = np.array([bl[n] for n in common])
    res = topk_analysis(ov, bv, ks)
    line = '{:25s}  {:5d}'.format(task+'/'+model, len(common))
    for _, k, j, r in res:
        line += '  {:.2f}   {:6.3f}'.format(j, r)
    print(line)

print()
print('EDGE LEVEL: Ours vs EAP-IG (repro)')
header = '{:25s}  {:>5s}'.format('task/model', 'n')
for k in ks:
    header += '  J@{:<4s} rho@{:<4s}'.format('{:.0%}'.format(k), '{:.0%}'.format(k))
print(header)
print('-' * 140)
for task, model in COLUMNS_EDGE:
    stask = task.replace('_', '-')
    ours_path = R / 'mib_edge_hard_topk' / '{}_{}_scores.pt'.format(task, model)
    bl_path = R / 'eapig_repro' / 'EAP-IG-inputs_patching_edge' / '{}_{}'.format(stask, model) / 'importances.json'
    if not ours_path.exists() or not bl_path.exists(): continue
    ours_data = torch.load(ours_path, weights_only=False, map_location='cpu')
    ov = ours_data['scores'].numpy()
    bl_data = json.load(open(bl_path))
    bv = np.array([e['score'] for e in bl_data['edges'].values()])
    res = topk_analysis(ov, bv, ks)
    line = '{:25s}  {:5d}'.format(task+'/'+model, len(ov))
    for _, k, j, r in res:
        line += '  {:.2f}   {:6.3f}'.format(j, r)
    print(line)
