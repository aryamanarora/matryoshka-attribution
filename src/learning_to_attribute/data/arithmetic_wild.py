"""Adapter for the goodfire-ai/arithmetic-wild dataset, exposing the CausalGymDataset API
(sample_pair / tokenize_pair) for span-aligned interchange-intervention attribution.

Tasks: addition, months, weekdays, hours (Llama-3.1-8B base). Each example has a base and a
counterfactual prompt; spans are {input (operand1), offset (operand2), last_token}. The label
is the (single) answer token at the final position.
"""
import json, random, re
from dataclasses import dataclass, field
from pathlib import Path
import torch

TEMPLATES = {
    "addition": "{input}+{offset}=",
    "months":   "Q: What month is {offset} months after {input}?\nA:",
    "weekdays": "Q: What day is {offset} days after {input}?\nA:",
    "hours":    "Q: In 24-hour time, it is now {input}:00. What time will it be in {offset} hours?\nA: In 24-hour time, it will be ",
}
SPAN_NAMES = ["input", "offset", "last_token"]

# Per-task causal model (faithful to goodfire-ai/arithmetic-wild tasks/*/causal_models.py).
# output = compute(premod), premod = idx(input) + idx(offset); interchange targets for the
# three causal variables {input, offset, output} are derived generically from these.
_MONTHS = ["January", "February", "March", "April", "May", "June",
           "July", "August", "September", "October", "November", "December"]
_DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
TASK_MODELS = {
    "months":   {"inp_idx": lambda s: _MONTHS.index(s) + 1,
                 "out": lambda p: " " + _MONTHS[(p % 12) - 1]},
    "weekdays": {"inp_idx": lambda s: _DAYS.index(s) + 1,
                 "out": lambda p: " " + _DAYS[(p % 7) - 1]},
    "hours":    {"inp_idx": lambda s: int(s),
                 "out": lambda p: f"{p % 24:02d}:00"},
    "addition": {"inp_idx": lambda s: int(s),
                 "out": lambda p: str(p)},
}


def _offset_val(task, ex):
    """idx(offset) recovered from premod (premod = idx(input) + idx(offset))."""
    return ex["premod"] - TASK_MODELS[task]["inp_idx"](ex["input"])


def concept_target_strings(task, base_ex, cf_ex):
    """Interchange target surface strings for patching each causal variable from cf into base:
      input  -> use cf's input, base's offset;  offset -> base's input, cf's offset;
      output -> cf's whole answer. Matches a per-variable interchange intervention."""
    m = TASK_MODELS[task]
    p_input = m["inp_idx"](cf_ex["input"]) + _offset_val(task, base_ex)
    p_offset = m["inp_idx"](base_ex["input"]) + _offset_val(task, cf_ex)
    return {"input": m["out"](p_input), "offset": m["out"](p_offset),
            "output": cf_ex["raw_output"]}


@dataclass
class ArithTokenizedPair:
    base_input_ids: torch.Tensor
    src_input_ids: torch.Tensor
    base_alignment: list          # [span][tok positions]
    src_alignment: list
    base_label_id: int
    src_label_id: int
    num_spans: int = 3
    span_names: tuple = ("input", "offset", "last_token")
    label_span_indices: list = field(default_factory=lambda: [2])  # last_token
    targets: dict = field(default_factory=dict)   # {concept: token_id} interchange targets


def _slot_char_spans(template, filled):
    """Recover char spans of {input}/{offset} in `filled` by aligning template literals."""
    parts = re.split(r"(\{input\}|\{offset\})", template)
    spans, pos = {}, 0
    for i, seg in enumerate(parts):
        if seg in ("{input}", "{offset}"):
            nxt = parts[i + 1] if i + 1 < len(parts) else ""
            end = len(filled) if nxt == "" else filled.index(nxt, pos)
            spans[seg[1:-1]] = (pos, end)
            pos = end
        elif seg:
            idx = filled.index(seg, pos)
            pos = idx + len(seg)
    return spans


class ArithmeticWildDataset:
    def __init__(self, task_name, data_dir, seed=42):
        # task_name like "arith/months" or "months"
        self.task_name = task_name
        key = task_name.split("/")[-1]
        self.key = key
        self.template = TEMPLATES[key]
        self.span_names = SPAN_NAMES
        path = Path(data_dir) / key / "filtered_dataset.json"
        data = json.load(open(path))
        self.bases = data["input"]
        self.cfs = [c[0] for c in data["counterfactual_inputs"]]
        self.rng = random.Random(seed)

    @property
    def num_spans(self):
        return 3

    def sample_pair(self):
        i = self.rng.randrange(len(self.bases))
        return (self.bases[i], self.cfs[i])

    def _spans_and_label(self, ex, tokenizer, device):
        raw = ex["raw_input"]
        ids = tokenizer(raw, return_tensors="pt", return_offsets_mapping=True)
        offsets = ids["offset_mapping"][0].tolist()
        input_ids = ids["input_ids"].to(device)
        n = input_ids.shape[1]
        cs = _slot_char_spans(self.template, raw)
        def toks_for(cspan):
            a, b = cspan
            return [t for t, (ts, te) in enumerate(offsets) if te > ts and ts < b and te > a]
        align = [toks_for(cs["input"]), toks_for(cs["offset"]), [n - 1]]  # last_token
        label_id = tokenizer.encode(ex["raw_output"], add_special_tokens=False)[0]
        return input_ids, align, label_id

    def tokenize_pair(self, pair, tokenizer, device="cpu"):
        base, cf = pair
        b_ids, b_align, b_lab = self._spans_and_label(base, tokenizer, device)
        s_ids, s_align, s_lab = self._spans_and_label(cf, tokenizer, device)
        tgt_str = concept_target_strings(self.key, base, cf)
        targets = {c: tokenizer.encode(s, add_special_tokens=False)[0]
                   for c, s in tgt_str.items()}   # {input, offset, output} -> token id
        return ArithTokenizedPair(
            base_input_ids=b_ids, src_input_ids=s_ids,
            base_alignment=b_align, src_alignment=s_align,
            base_label_id=b_lab, src_label_id=s_lab, targets=targets,
        )

    @staticmethod
    def available_tasks(data_dir):
        return sorted(p.name for p in Path(data_dir).iterdir()
                      if (p / "filtered_dataset.json").exists())
