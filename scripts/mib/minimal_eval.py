"""Self-contained (no eval_mib import): load our MAttr importances via Graph.from_json and
evaluate on gemma2/arc_easy. Run the SAME script in both venvs to confirm the transformer_lens
version is what shifts the Gemma corrupted floor.
"""
import argparse, sys
from functools import partial
from pathlib import Path
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--importances", default="results/topklog_lr_0.05/arc_easy_gemma2_importances.json")
ap.add_argument("--eval-examples", type=int, default=200)
ap.add_argument("--batch-size", type=int, default=4)
ap.add_argument("--mib-path", default="./MIB-circuit-track")
args = ap.parse_args()

mib = Path(args.mib_path).resolve()
sys.path.insert(0, str(mib)); sys.path.insert(0, str(mib / "EAP-IG" / "src"))
import transformer_lens, importlib.metadata as M
from transformer_lens import HookedTransformer
from eap.graph import Graph
from MIB_circuit_track.metrics import get_metric
from MIB_circuit_track.dataset import HFEAPDataset
from MIB_circuit_track.evaluation import evaluate_area_under_curve

print(f"transformer_lens=={M.version('transformer_lens')}  torch=={M.version('torch')}")
tl = HookedTransformer.from_pretrained("google/gemma-2-2b", attn_implementation="eager", torch_dtype=torch.bfloat16)
tl.cfg.use_split_qkv_input = True; tl.cfg.use_attn_result = True
tl.cfg.use_hook_mlp_in = True; tl.cfg.ungroup_grouped_query_attention = True

g = Graph.from_json(args.importances)
ds = HFEAPDataset("mib-bench/arc_easy", tl.tokenizer, split="validation", task="arc_easy", model_name="gemma2")
if args.eval_examples: ds.head(args.eval_examples)
dl = ds.to_dataloader(batch_size=args.batch_size)
am = partial(get_metric("logit_diff", "arc_easy", tl.tokenizer, tl), mean=False, loss=False)
out = evaluate_area_under_curve(tl, g, dl, am, level="node", absolute=False)
acc, acc_auc = out[5], out[6]
print(f"RESULT acc-AUC={acc_auc:.4f} floor={acc[0]:.3f} curve={[round(x,3) for x in acc]}")
