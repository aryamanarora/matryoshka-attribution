"""Subject-Verb Agreement (SVA) counterfactual datasets.

The four feature-circuits SVA tasks (nounpp, rc, simple, within_rc), as minimal-pair
counterfactuals: a clean prefix that licenses one verb-number and a patch prefix (the same
sentence with the subject's number flipped) that licenses the other. Each example carries the
correct vs incorrect next-token verb form, so the metric is a logit-diff exactly like MIB.

Drop-in for ``MIB_circuit_track.dataset.HFEAPDataset``: ``dataset[idx]`` returns
``(clean_text, corrupted_text, [correct_token_id, incorrect_token_id])`` and ``len(dataset)``
works — so it plugs straight into ``scripts/eval_mib.py``'s ``loss_fn`` (and the shared
``learn_scores`` trainer / hook-based masking) with no nnsight.

Data: ``data/sva_data/{task}_{split}.json`` (copied from circuits' ``data/feature_circuits``),
records ``{clean_prefix, patch_prefix, clean_answer, patch_answer, case}`` where the answers
are space-prefixed verb tokens (e.g. " are" / " is").
"""

import json
from pathlib import Path

SVA_TASKS = ("nounpp", "rc", "simple", "within_rc")
_DATA_DIR = Path(__file__).parent / "sva_data"


class SVADataset:
    """SVA minimal-pair dataset, HFEAPDataset-compatible.

    Args:
        task: one of SVA_TASKS.
        tokenizer: HF tokenizer (used to map the answer strings to token ids).
        split: "train" | "test".
        data_dir: override the default packaged data directory.
        answer_token: which token of the (possibly multi-token) answer to score; -1 = last
            (matches circuits' ``tokenizer(answer).input_ids[-1]``). The answers here are
            single space-prefixed words, so -1 is the verb token.
    """

    def __init__(self, task: str, tokenizer, split: str = "train",
                 data_dir=None, answer_token: int = -1):
        if task not in SVA_TASKS:
            raise ValueError(f"Unknown SVA task {task!r}; expected one of {SVA_TASKS}")
        self.task = task
        self.split = split
        self.tokenizer = tokenizer
        self.answer_token = answer_token
        path = Path(data_dir or _DATA_DIR) / f"{task}_{split}.json"
        with open(path) as f:
            self.records = [json.loads(line) for line in f if line.strip()]
        # precompute answer token ids (match circuits: tokenizer(answer).input_ids[-1])
        self._labels = [
            [self._answer_id(r["clean_answer"]), self._answer_id(r["patch_answer"])]
            for r in self.records
        ]

    def _answer_id(self, answer: str) -> int:
        return self.tokenizer(answer).input_ids[self.answer_token]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        # (clean, corrupted, [correct_id, incorrect_id]); clean prefix licenses clean_answer.
        return r["clean_prefix"], r["patch_prefix"], self._labels[idx]
