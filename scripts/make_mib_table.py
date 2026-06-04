"""Generate LaTeX table of MIB CPR AUC results matching the MIB paper format."""

# Our results
node = {
    ("IOI", "GPT-2"): {"topk": 1.85, "hard_topk": 1.84, "hard_concrete": 1.57},
    ("IOI", "Qwen-2.5"): {"topk": 1.48, "hard_topk": 1.46, "hard_concrete": 1.35},
    ("IOI", "Gemma-2"): {"topk": 1.37, "hard_topk": 1.20, "hard_concrete": 1.14},
    ("IOI", "Llama-3.1"): {"topk": 2.17},
    ("MCQA", "Qwen-2.5"): {"topk": 1.68, "hard_topk": 1.74, "hard_concrete": 1.51},
    ("MCQA", "Gemma-2"): {"topk": 1.47, "hard_topk": 1.43},
    ("Arithmetic", "Llama-3.1"): {"topk": 1.23},
}

edge = {
    ("IOI", "GPT-2"): {"topk": 10.40, "hard_topk": 10.06, "hard_concrete": 4.42},
    ("IOI", "Qwen-2.5"): {"topk": 4.15, "hard_topk": 4.55, "hard_concrete": 1.60},
    ("IOI", "Gemma-2"): {"topk": 7.37},
    ("IOI", "Llama-3.1"): {"topk": 7.37},
    ("MCQA", "Qwen-2.5"): {"topk": 5.25, "hard_topk": 7.45, "hard_concrete": 2.93},
    ("MCQA", "Gemma-2"): {"topk": 1.38},
}

# MIB baselines (from the paper image)
baselines = {
    "Random": {
        ("IOI", "GPT-2"): 0.25, ("IOI", "Qwen-2.5"): 0.28, ("IOI", "Gemma-2"): 0.30,
        ("IOI", "Llama-3.1"): 0.25, ("Arithmetic", "Llama-3.1"): 0.25,
        ("MCQA", "Qwen-2.5"): 0.27, ("MCQA", "Gemma-2"): 0.32, ("MCQA", "Llama-3.1"): 0.26,
        ("ARC (E)", "Gemma-2"): 0.32, ("ARC (E)", "Llama-3.1"): 0.26, ("ARC (C)", "Llama-3.1"): 0.25,
    },
    "EAP-IG-inputs (CF)": {
        ("IOI", "GPT-2"): 1.85, ("IOI", "Qwen-2.5"): 1.63, ("IOI", "Gemma-2"): 3.20,
        ("IOI", "Llama-3.1"): 2.08, ("Arithmetic", "Llama-3.1"): 0.99,
        ("MCQA", "Qwen-2.5"): 1.16, ("MCQA", "Gemma-2"): 1.64, ("MCQA", "Llama-3.1"): 1.05,
        ("ARC (E)", "Gemma-2"): 1.53, ("ARC (E)", "Llama-3.1"): 1.04, ("ARC (C)", "Llama-3.1"): 0.98,
    },
    "NAP-IG (CF)": {
        ("IOI", "GPT-2"): 0.76, ("IOI", "Qwen-2.5"): 0.29, ("IOI", "Gemma-2"): 1.52,
        ("IOI", "Llama-3.1"): 0.42, ("Arithmetic", "Llama-3.1"): 0.39,
        ("MCQA", "Qwen-2.5"): 0.77, ("MCQA", "Gemma-2"): 1.71, ("MCQA", "Llama-3.1"): 1.87,
        ("ARC (E)", "Gemma-2"): 1.53, ("ARC (E)", "Llama-3.1"): 0.26, ("ARC (C)", "Llama-3.1"): 0.26,
    },
    "UGS": {
        ("IOI", "GPT-2"): 0.97, ("IOI", "Qwen-2.5"): 0.98,
        ("MCQA", "Qwen-2.5"): 1.17,
    },
}

columns = [
    ("IOI", "GPT-2"), ("IOI", "Qwen-2.5"), ("IOI", "Gemma-2"), ("IOI", "Llama-3.1"),
    ("Arithmetic", "Llama-3.1"),
    ("MCQA", "Qwen-2.5"), ("MCQA", "Gemma-2"), ("MCQA", "Llama-3.1"),
    ("ARC (E)", "Gemma-2"), ("ARC (E)", "Llama-3.1"), ("ARC (C)", "Llama-3.1"),
]

def fmt(v):
    if v is None:
        return "-"
    return f"{v:.2f}"

def row(name, data, bold_threshold=None):
    vals = []
    for col in columns:
        v = data.get(col)
        s = fmt(v)
        if bold_threshold and v is not None and v >= bold_threshold:
            s = f"\\textbf{{{s}}}"
        vals.append(s)
    return f"{name} & " + " & ".join(vals) + " \\\\"

# Print table
print("\\begin{tabular}{l" + "r" * 11 + "}")
print("\\toprule")
print("& \\multicolumn{4}{c}{IOI} & Arithmetic & \\multicolumn{3}{c}{MCQA} & \\multicolumn{2}{c}{ARC (E)} & ARC (C) \\\\")
print("\\cmidrule(lr){2-5} \\cmidrule(lr){6-6} \\cmidrule(lr){7-9} \\cmidrule(lr){10-11} \\cmidrule(lr){12-12}")
print("\\textbf{Method} & GPT-2 & Qwen-2.5 & Gemma-2 & Llama-3.1 & Llama-3.1 & Qwen-2.5 & Gemma-2 & Llama-3.1 & Gemma-2 & Llama-3.1 & Llama-3.1 \\\\")
print("\\midrule")

# Baselines
for name, data in baselines.items():
    print(row(name, data))
print("\\midrule")

# Our node methods
our_methods = [
    ("Ours: node (sigmoid top-k)", "topk", node),
    ("Ours: node (hard top-k + ST)", "hard_topk", node),
    ("Ours: node (hard concrete + L0)", "hard_concrete", node),
]
for name, key, source in our_methods:
    data = {col: source.get(col, {}).get(key) for col in columns}
    print(row(name, data))
print("\\midrule")

# Our edge methods
our_edge_methods = [
    ("Ours: edge (sigmoid top-k)", "topk", edge),
    ("Ours: edge (hard top-k + ST)", "hard_topk", edge),
    ("Ours: edge (hard concrete + L0)", "hard_concrete", edge),
]
for name, key, source in our_edge_methods:
    data = {col: source.get(col, {}).get(key) for col in columns}
    print(row(name, data))

print("\\bottomrule")
print("\\end{tabular}")
