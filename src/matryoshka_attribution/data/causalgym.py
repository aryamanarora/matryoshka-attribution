"""CausalGym dataset loader for span-aware attribution experiments."""

import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path

import torch


TEMPLATES_DIR = Path(__file__).parent / "causalgym_templates"


@dataclass
class SpanAlignedPair:
    """A minimal pair with span-level structure."""
    base_spans: list[str]
    src_spans: list[str]
    span_names: list[str]
    base_label: str
    src_label: str
    base_type: str
    src_type: str
    label_span_indices: list[int] = field(default_factory=list)


@dataclass
class TokenizedSpanPair:
    """A tokenized SpanAlignedPair with position-to-span mappings."""
    base_input_ids: torch.Tensor         # [1, base_seq_len]
    src_input_ids: torch.Tensor          # [1, src_seq_len]
    base_alignment: list[list[int]]      # alignment[span_i] = [tok_pos, ...]
    src_alignment: list[list[int]]
    base_label_id: int
    src_label_id: int
    num_spans: int
    span_names: list[str]
    label_span_indices: list[int]
    targets: dict = field(default_factory=dict)   # {concept: token_id} interchange targets


class CausalGymDataset:
    """Wraps causalgym template data for span-aware attribution.

    Usage::

        ds = CausalGymDataset("syntaxgym/agr_gender")
        pair = ds.sample_pair()
        tok = ds.tokenize_pair(pair, tokenizer)
    """

    def __init__(self, task_name: str, seed: int = 42):
        self.task_name = task_name
        self.rng = random.Random(seed)

        # Parse task name: "syntaxgym/agr_gender" -> file="syntaxgym", key="agr_gender"
        template_file, template_key = task_name.split("/", 1)
        if template_key.endswith("_inverted"):
            template_key = template_key[: -len("_inverted")]

        json_path = TEMPLATES_DIR / f"{template_file}.json"
        with open(json_path) as f:
            all_data = json.load(f)
        data = all_data[template_key]

        # Parse template into spans (replicating causalgym's regex)
        self.templates = data["templates"]
        raw_template = "<|endoftext|>" + self.rng.choice(self.templates)
        self.template = [
            x for x in re.split(
                r"(?<=\})|(?= \{)|(?<! )(?=\{)", raw_template
            ) if x != ""
        ]

        # Label variables and types
        label = data["label"]
        self.label_vars = label if isinstance(label, list) else [label]
        self.labels = data["labels"]
        self.types = list(self.labels.keys())
        self.variables = data["variables"]
        self.result_prepend_space = data.get("result_prepend_space", False)

        # Per-span metadata
        self.vars_per_span = []
        self.span_names = []
        self.label_span_indices = []
        for i, span in enumerate(self.template):
            var = re.findall(r"\{(.+?)\}", span)
            self.vars_per_span.append(var)
            if len(var) == 1:
                self.span_names.append("{" + var[0] + "}")
                if var[0] in self.label_vars:
                    self.label_span_indices.append(i)
            else:
                self.span_names.append(span.replace(" ", "_"))

        # Validate correlated variables (dot notation)
        length = {}
        for var in self.variables:
            if "." in var:
                head = var.split(".")[0]
                if head not in length:
                    length[head] = len(self.variables[var])
                else:
                    assert length[head] == len(self.variables[var])

    @property
    def num_spans(self) -> int:
        return len(self.template)

    def sample_pair(self) -> SpanAlignedPair:
        """Sample a random minimal pair."""
        # Pick two different types
        base_type = self.rng.choice(self.types)
        src_type = base_type
        while src_type == base_type:
            src_type = self.rng.choice(self.types)

        base = self.template[:]
        src = self.template[:]

        # Fill in variables
        stored_choices = {}
        for i in range(len(self.template)):
            var_list = self.vars_per_span[i]
            if not var_list:
                continue
            var = var_list[0]
            var_temp = "{" + var + "}"

            if var in self.label_vars:
                # Label variable: different for base and src
                base[i] = base[i].replace(
                    var_temp, self.rng.choice(self.variables[var][base_type]))
                src[i] = src[i].replace(
                    var_temp, self.rng.choice(self.variables[var][src_type]))
            elif "." in var:
                # Correlated variable
                head = var.split(".")[0]
                if head not in stored_choices:
                    stored_choices[head] = self.rng.randint(
                        0, len(self.variables[var]) - 1)
                val = self.variables[var][stored_choices[head]]
                base[i] = base[i].replace(var_temp, val)
                src[i] = src[i].replace(var_temp, val)
            else:
                # Shared variable
                val = self.rng.choice(self.variables[var])
                base[i] = base[i].replace(var_temp, val)
                src[i] = src[i].replace(var_temp, val)

        # Labels
        base_label = self.rng.choice(self.labels[base_type])
        src_label = self.rng.choice(self.labels[src_type])
        if self.result_prepend_space:
            base_label = " " + base_label
            src_label = " " + src_label

        return SpanAlignedPair(
            base_spans=base,
            src_spans=src,
            span_names=self.span_names,
            base_label=base_label,
            src_label=src_label,
            base_type=base_type,
            src_type=src_type,
            label_span_indices=self.label_span_indices,
        )

    def tokenize_pair(self, pair: SpanAlignedPair, tokenizer,
                      device: str = "cpu") -> TokenizedSpanPair:
        """Tokenize a pair and compute per-span token alignments."""
        base_text = "".join(pair.base_spans)
        src_text = "".join(pair.src_spans)
        base_ids = tokenizer(base_text, return_tensors="pt").input_ids.to(device)
        src_ids = tokenizer(src_text, return_tensors="pt").input_ids.to(device)

        # The tokenizer may add a BOS token that tokenizer.tokenize() doesn't
        # produce. Compute the offset by comparing lengths.
        base_toks_flat = tokenizer.tokenize(base_text)
        bos_offset_base = base_ids.shape[1] - len(base_toks_flat)
        src_toks_flat = tokenizer.tokenize(src_text)
        bos_offset_src = src_ids.shape[1] - len(src_toks_flat)

        base_alignment, src_alignment = [], []
        pos_base, pos_src = bos_offset_base, bos_offset_src

        for i in range(len(pair.base_spans)):
            tok_base = tokenizer.tokenize(pair.base_spans[i])
            tok_src = tokenizer.tokenize(pair.src_spans[i])
            base_alignment.append(list(range(pos_base, pos_base + len(tok_base))))
            src_alignment.append(list(range(pos_src, pos_src + len(tok_src))))
            pos_base += len(tok_base)
            pos_src += len(tok_src)

        base_label_id = tokenizer.encode(
            pair.base_label, add_special_tokens=False)[0]
        src_label_id = tokenizer.encode(
            pair.src_label, add_special_tokens=False)[0]

        return TokenizedSpanPair(
            base_input_ids=base_ids,
            src_input_ids=src_ids,
            base_alignment=base_alignment,
            src_alignment=src_alignment,
            base_label_id=base_label_id,
            src_label_id=src_label_id,
            num_spans=len(pair.base_spans),
            span_names=pair.span_names,
            label_span_indices=pair.label_span_indices,
            targets={"src": src_label_id},   # interchange target = source label
        )

    @staticmethod
    def list_tasks() -> list[str]:
        """List all available task names."""
        tasks = []
        for json_file in sorted(TEMPLATES_DIR.glob("*.json")):
            name = json_file.stem
            with open(json_file) as f:
                data = json.load(f)
            tasks.extend(f"{name}/{key}" for key in data)
        return tasks
