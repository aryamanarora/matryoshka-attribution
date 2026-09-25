import os
import pickle
import argparse
import sys
from tabulate import tabulate

from MIB_circuit_track.utils import COL_MAPPING

"""
If you set --metric to `cpr` or `cmd`, this script will print a table of the following form:
Method      | IOI (GPT) | IOI (QWen) | IOI (Gemma) | IOI (Llama) | MCQA (QWen) | MCQA (Gemma) | MCQA (Llama) | Arithmetic (Llama) | ARC-E (Gemma) | ARC-E (Llama) | ARC-C (Llama)
Random      |
Method 1    |
Method 2    |
...
"""


def area_under_curve(X, Y):
    area_under = 0.
    for idx in range(len(X) - 1):
        x_1, x_2 = X[idx] / X[-1], X[idx+1] / X[-1]
        trapezoidal = (x_2 - x_1) * ((Y[idx+1] + Y[idx]) / 2)
        area_under += trapezoidal
    return area_under


def print_interpbench_scores():
    scores = {}
    for method_name in os.listdir(args.output_dir):
        if method_name not in scores:
            scores[method_name] = "-"

        method_path = os.path.join(args.output_dir, method_name)
        for results_file in os.listdir(method_path):
            task, model, split, abs = os.path.splitext(results_file)[0].split("_")
            if f"{task}_{model}" not in COL_MAPPING:
                continue
            if model != "interpbench":
                continue
            if split != args.split:
                continue

            results_path = os.path.join(method_path, results_file)
            with open(results_path, 'rb') as handle:
                results = pickle.load(handle)
            auroc = area_under_curve(results["FPR"], results["TPR"])
            scores[method_name] = auroc

    table = []
    header = ["Method", "IOI (InterpBench)"]
    table.append(header)

    for method_name in scores:
        row = [method_name]
        print(type(scores[method_name]))
        row.append(f"{scores[method_name]:.2f}" if type(scores[method_name]) == float else scores[method_name])
        table.append(row)

    print(tabulate(table))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default='results')
    parser.add_argument("--split", type=str, choices=['train', 'validation', 'test'], default='validation')
    parser.add_argument("--metric", type=str, default='cpr', choices=['cpr', 'cmd', 'auroc'])
    args = parser.parse_args()

    if args.metric == "auroc":
        print_interpbench_scores()
        sys.exit()

    scores = {}
    for method_name in os.listdir(args.output_dir):
        if method_name not in scores:
            scores[method_name] = ["-"] * 12

        method_path = os.path.join(args.output_dir, method_name)
        for results_file in os.listdir(method_path):
            task, model, split, abs = os.path.splitext(results_file)[0].split("_")
            if (f"{task}_{model}" not in COL_MAPPING) or model=='interpbench':
                continue
            if split != args.split:
                continue
            print(args.metric, abs)
            
            abs = abs.split("abs-")[1]
            if args.metric == 'cpr' and abs == "True":
                # Check whether results exist for abs=False
                if os.path.exists(os.path.join(method_path, results_file.replace("abs-True", "abs-False"))):
                    print("made it")
                    continue
            elif args.metric == 'cmd' and abs == "False":
                # Check whether results exist for abs=True
                if os.path.exists(os.path.join(method_path, results_file.replace("abs-False", "abs-True"))):
                    continue

            results_path = os.path.join(method_path, results_file)
            with open(results_path, 'rb') as handle:
                results = pickle.load(handle)

            col_idx = COL_MAPPING[f"{task}_{model}"]
            scores[method_name][col_idx] = results

    table = []
    header = ["Method", "IOI (GPT)", "IOI (QWen)", "IOI (Gemma)", "IOI (Llama)", "MCQA (QWen)", "MCQA (Gemma)", "MCQA (Llama)",
                "Arithmetic (Llama)", "ARC-E (Gemma)", "ARC-E (Llama)", "ARC-C (Llama)"]
    table.append(header)

    for method_name in scores:
        row = [method_name]

        if args.metric == 'cpr':
            scorelist = [f"{s['area_under']:.2f}" if type(s) == dict else "-" for s in scores[method_name]]
        else:
            scorelist = [f"{s['area_from_1']:.2f}" if type(s) == dict else "-" for s in scores[method_name]]
        row.extend(scorelist)

        table.append(row)

    print(tabulate(table))
