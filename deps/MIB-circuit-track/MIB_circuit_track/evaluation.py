import math 

from typing import Literal, Optional

import torch
from torch.utils.data import DataLoader
from transformer_lens import HookedTransformer

from eap.graph import Graph 
from eap.evaluate import evaluate_baseline, evaluate_graph

def evaluate_area_under_curve(model: HookedTransformer, graph: Graph, dataloader, metrics, quiet:bool=False, 
                              level:Literal['edge', 'node','neuron']='edge', log_scale:bool=False, absolute:bool=True, 
                              intervention: Literal['patching', 'zero', 'mean','mean-positional', 'optimal']='patching', 
                              intervention_dataloader:DataLoader=None, optimal_ablation_path:Optional[str]=None, 
                              no_normalize:Optional[bool]=False, apply_greedy:bool=False,
                              percentages=None, refs=None, invert=False, extra=None):
    # invert=True (learning-to-attribute, 2026-09-16): NOISING. At each proportion the top-n
    # nodes are taken OUT of the graph and every other node kept in, so evaluate_graph corrupts
    # exactly the selected components and leaves the rest clean -- the necessity direction, the
    # mirror of the sufficiency sweep this function otherwise runs. Node level only. The two
    # references, the grid and both integrals are unchanged, so faithfulnesses starts near 1
    # (nothing patched) and falls; read area_from_1 as the score. The input node (forward index 0)
    # is always kept clean: corrupting the embedding is not an ablation of a component, and in
    # our graphs it carries the largest score, so it would otherwise be the first node patched.
    # extra, if a dict, receives 'flip_accuracies' (fraction of examples whose metric goes
    # NEGATIVE, i.e. the counterfactual answer wins -- the reverse-IIA reading of noising) and
    # 'flip_acc_auc' (its log-trapezoid), so the 7-value return stays as every caller expects.
    # BACKWARD-COMPATIBLE REFERENCE CACHE. Both references are GRAPH-INDEPENDENT: the baseline
    # never touches the graph, and the corrupted score is apply_topn(0) -- the empty circuit,
    # which is the same circuit whatever scores the graph carries. So a caller that evaluates
    # many graphs over one dataloader can compute them once. Pass an empty dict to have them
    # filled on the first call and reused after; pass nothing (the default) and every call
    # recomputes exactly as before. Cuts a two-point call from 4 dataset passes to 2.
    if refs is not None and 'baseline' in refs:
        baseline_score, corrupted_score = refs['baseline'], refs['corrupted']
    else:
        baseline_score = evaluate_baseline(model, dataloader, metrics).mean().item()
        graph.apply_topn(0, True)
        corrupted_score = evaluate_graph(model, graph, dataloader, metrics, quiet=quiet, intervention=intervention,
                                         intervention_dataloader=intervention_dataloader, optimal_ablation_path=optimal_ablation_path).mean().item()
        if refs is not None:
            refs['baseline'], refs['corrupted'] = baseline_score, corrupted_score

    if level == 'neuron':
        assert graph.neurons_scores is not None, "Neuron scores must be present for neuron-level evaluation"
        n_scored_items = (~torch.isnan(graph.neurons_scores)).sum().item()
    elif level == 'node':
        assert graph.nodes_scores is not None, "Node scores must be present for node-level evaluation"
        n_scored_items = (~torch.isnan(graph.nodes_scores)).sum().item()
    else:
        n_scored_items = len(graph.edges)
    
    # BACKWARD-COMPATIBLE: None keeps MIB's published ten-point grid, so every existing caller
    # is unchanged. An explicit sequence lets a caller integrate over its own sparsities --
    # scripts/mib/eval_dbm_multisparsity.py in the learning-to-attribute repo passes each DBM
    # run's converged L0, so that row is produced by THIS function rather than a reimplementation
    # of it. Must be sorted ascending and within (0, 1]; the trapezoid and the log-AUC below both
    # assume it.
    if percentages is None:
        percentages = (.001, .002, .005, .01, .02, .05, .1, .2, .5, 1)
    percentages = tuple(percentages)
    # A ONE-POINT call is allowed and returns the raw faithfulness/accuracy at that proportion
    # with the integrals as NaN -- there is no area under a single point, and returning 0 for it
    # would be a number someone could average. Used by callers that build a curve out of points
    # measured on DIFFERENT graphs and integrate it themselves.
    assert len(percentages) >= 1, "need at least one point to evaluate"
    assert all(a < b for a, b in zip(percentages, percentages[1:])), \
        f"percentages must be strictly ascending, got {percentages}"

    faithfulnesses = []
    weighted_edge_counts = []
    accuracies = []
    for pct in percentages:
        this_graph = graph
        curr_num_items = int(pct * n_scored_items)
        print(f"Computing results for {pct*100}% of {level}s (N={curr_num_items})")
        if apply_greedy:
            assert level == 'edge', "Greedy application only supported for edge-level evaluation"
            this_graph.apply_greedy(curr_num_items, absolute=absolute, prune=True)
        else:
            this_graph.apply_topn(curr_num_items, absolute, level=level, prune=True)
        if invert:
            assert level == 'node', 'invert (noising) is implemented for node level only'
            scored = ~torch.isnan(this_graph.nodes_scores)
            selected = this_graph.nodes_in_graph.clone()
            this_graph.reset()
            this_graph.nodes_in_graph[:] = (~selected) | (~scored)
            this_graph.nodes_in_graph[0] = True
            this_graph.in_graph += this_graph.nodes_in_graph.view(-1, 1)

        weighted_edge_count = this_graph.weighted_edge_count()
        weighted_edge_counts.append(weighted_edge_count)

        ablated_ex = evaluate_graph(model, this_graph, dataloader, metrics,
                                    quiet=quiet, intervention=intervention,
                                    intervention_dataloader=intervention_dataloader,
                                    optimal_ablation_path=optimal_ablation_path)
        ablated_score = ablated_ex.mean().item()
        # accuracy at this sparsity: fraction of examples with metric>0 (base beats source).
        # NB: correct for logit_diff-style tasks; greater-than/arithmetic use a different metric.
        accuracies.append((ablated_ex > 0).float().mean().item())
        if extra is not None:
            extra.setdefault('flip_accuracies', []).append((ablated_ex < 0).float().mean().item())
        if no_normalize:
            faithfulness = ablated_score
        else:
            faithfulness = (ablated_score - corrupted_score) / (baseline_score - corrupted_score)
        faithfulnesses.append(faithfulness)
    
    area_under = 0.
    area_from_1 = 0.
    for i in range(len(faithfulnesses) - 1):
        i_1, i_2 = i, i+1
        x_1 = percentages[i_1]
        x_2 = percentages[i_2]
        # area from point to 100
        if log_scale:
            x_1 = math.log(x_1)
            x_2 = math.log(x_2)
        trapezoidal = (x_2 - x_1) * \
                        (((abs(1. - faithfulnesses[i_1])) + (abs(1. - faithfulnesses[i_2]))) / 2)
        area_from_1 += trapezoidal 
        
        trapezoidal = (x_2 - x_1) * ((faithfulnesses[i_1] + faithfulnesses[i_2]) / 2)
        area_under += trapezoidal
    average = sum(faithfulnesses) / len(faithfulnesses)
    # accuracy AUC: log-sparsity-weighted mean of accuracy (matches eval_sva's acc_auc), so it
    # rewards recovering the decision at few nodes rather than the dense end.
    lx = [math.log(p) for p in percentages]
    if len(percentages) < 2:
        area_under = area_from_1 = acc_auc = float('nan')
    else:
        acc_auc = (sum((lx[i + 1] - lx[i]) * (accuracies[i] + accuracies[i + 1]) / 2
                       for i in range(len(accuracies) - 1)) / (lx[-1] - lx[0]))
        if extra is not None and 'flip_accuracies' in extra:
            fa = extra['flip_accuracies']
            extra['flip_acc_auc'] = (sum((lx[i + 1] - lx[i]) * (fa[i] + fa[i + 1]) / 2
                                         for i in range(len(fa) - 1)) / (lx[-1] - lx[0]))
    return weighted_edge_counts, area_under, area_from_1, average, faithfulnesses, accuracies, acc_auc


def compare_graphs(reference: Graph, hypothesis: Graph, by_node: bool = False):
    # Track {true, false} {positives, negatives}
    TP, FP, TN, FN = 0, 0, 0, 0
    total = 0

    if by_node:
        ref_objs = reference.nodes
        hyp_objs = hypothesis.nodes
    else:
        ref_objs = reference.edges
        hyp_objs = hypothesis.edges

    for obj in ref_objs.values():
        total += 1
        if obj.name not in hyp_objs:
            if obj.in_graph:
                TP += 1
            else:
                FP += 1
            continue
            
        if obj.in_graph and hyp_objs[obj.name].in_graph:
            TP += 1
        elif obj.in_graph and not hyp_objs[obj.name].in_graph:
            FN += 1
        elif not obj.in_graph and hyp_objs[obj.name].in_graph:
            FP += 1
        elif not obj.in_graph and not hyp_objs[obj.name].in_graph:
            TN += 1
    
    if TP + FP == 0:
        precision = 0
    else:
        precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    TP_rate = recall
    FP_rate = FP / (FP + TN)

    return {"precision": precision,
            "recall": recall,
            "TP_rate": TP_rate,
            "FP_rate": FP_rate}

def evaluate_area_under_roc(reference: Graph, hypothesis: Graph, by_node: bool = False):
    tpr_list = []
    fpr_list = []
    precision_list = []
    recall_list = []

    if by_node:
        ref_objs = reference.nodes
        hyp_objs = hypothesis.nodes
    else:
        ref_objs = reference.edges
        hyp_objs = hypothesis.edges
    
    num_objs = len(ref_objs.values())
    for pct in (.001, .002, .005, .01, .02, .05, .1, .2, .5, 1):
        this_num_objs = pct * num_objs
        if by_node:
            raise NotImplementedError("")
        else:
            hypothesis.apply_greedy(this_num_objs)
        scores = compare_graphs(reference, hypothesis)
        tpr_list.append(scores["TP_rate"])
        fpr_list.append(scores["FP_rate"])
        precision_list.append(scores["precision"])
        recall_list.append(scores["recall"])
    
    return {"TPR": tpr_list, "FPR": fpr_list,
            "precision": precision_list, "recall": recall_list}