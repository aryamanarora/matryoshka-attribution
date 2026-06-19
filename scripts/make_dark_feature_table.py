"""LaTeX longtable: one row per unique (layer,dim) dark-red DAS feature, listing the tasks
it tops (most-important in its layer) and its top-5 / bottom-5 logit-lens tokens.
Writes paper/tabs/dark_feature_logitlens.tex. Run on sc from repo root."""
import json
from collections import defaultdict

d = json.load(open("results/mtdas_dark_feature_logitlens.json"))

SPECIAL = [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("$", r"\$"),
           ("#", r"\#"), ("_", r"\_"), ("{", r"\{"), ("}", r"\}"),
           ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}")]


def esc(s):
    for a, b in SPECIAL:
        s = s.replace(a, b)
    return s


def tokfmt(toks):
    out = []
    for t in toks:
        s = "".join(c if 32 <= ord(c) <= 126 else "?" for c in t)  # ASCII-safe
        s = s.replace(" ", "_")          # visible space marker (GPT-style)
        if len(s) > 14:
            s = s[:13] + ".."
        out.append(r"\texttt{" + esc(s) + "}")
    return ", ".join(out)


feat = defaultdict(lambda: {"tasks": [], "top": None, "bot": None})
for task, entries in d.items():
    for e in entries:
        k = (e["layer"], e["dim"])
        feat[k]["tasks"].append(task)
        feat[k]["top"] = e["top5"]
        feat[k]["bot"] = e["bottom5"]

# most-shared first, then later layers
rows = sorted(feat.items(), key=lambda kv: (-len(kv[1]["tasks"]), -kv[0][0]))

L = [r"\begin{longtable}{@{}l l p{3.6cm} p{3.4cm} p{3.4cm}@{}}",
     r"\toprule",
     r"Layer & Dim & Tasks (\# top) & Top-5 logits & Bottom-5 logits \\",
     r"\midrule \endhead"]
for (lyr, dim), v in rows:
    tasks = ", ".join(esc(t) for t in v["tasks"])
    tcell = "(%d) %s" % (len(v["tasks"]), tasks)
    L.append("%d & %d & %s & %s & %s \\\\" % (lyr, dim, tcell, tokfmt(v["top"]), tokfmt(v["bot"])))
L += [r"\bottomrule", r"\end{longtable}"]
open("paper/tabs/dark_feature_logitlens.tex", "w").write("\n".join(L) + "\n")
print("rows (unique features):", len(rows),
      "| multi-task rows:", sum(1 for _, v in rows if len(v["tasks"]) > 1))
