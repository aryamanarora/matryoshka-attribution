"""Loader for the goodfire-ai/arithmetic-wild datasets (addition, months, weekdays, hours;
Llama-3.1-8B base). Each record pairs a base prompt with a counterfactual prompt and carries the
single-token answer of each; ``scripts/sva/eval_sva.py`` turns them into (clean, corrupted,
[base_id, source_id]) triples like the SVA tasks.
"""
import json
from pathlib import Path


class ArithmeticWildDataset:
    def __init__(self, task_name, data_dir):
        key = task_name.split("/")[-1]
        self.key = key
        path = Path(data_dir) / key / "filtered_dataset.json"
        data = json.load(open(path))
        self.bases = data["input"]                                 # {raw_input, raw_output, ...}
        self.cfs = [c[0] for c in data["counterfactual_inputs"]]  # paired by index

    @staticmethod
    def available_tasks(data_dir):
        return sorted(p.name for p in Path(data_dir).iterdir()
                      if (p / "filtered_dataset.json").exists())
