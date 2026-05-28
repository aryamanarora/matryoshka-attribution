"""Counterfactual dataset for interchange intervention experiments."""

from dataclasses import dataclass
from torch.utils.data import Dataset


@dataclass
class CounterfactualPair:
    """A single (clean, counterfactual) text pair."""
    clean: str
    cf: str


class CounterfactualDataset(Dataset):
    """Dataset of (clean_text, cf_text) pairs for interchange intervention.

    Each pair should tokenize to the same length. Use `validate()` with a
    tokenizer to check this after construction.

    Example usage::

        ds = CounterfactualDataset.from_template(
            template="Q: What month is {offset} months after {month}?\\nA:",
            clean_values={"offset": "four", "month": "January"},
            cf_values=[
                {"month": "October"},
                {"month": "March"},
            ],
        )
        # -> 2 pairs, both varying month while keeping offset fixed
    """

    def __init__(self, pairs: list[CounterfactualPair]):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx) -> CounterfactualPair:
        return self.pairs[idx]

    @classmethod
    def from_pairs(cls, pairs: list[tuple[str, str]]):
        """Create from a list of (clean, cf) string tuples."""
        return cls([CounterfactualPair(clean=c, cf=f) for c, f in pairs])

    @classmethod
    def from_template(cls, template: str, clean_values: dict,
                      cf_values: list[dict]):
        """Create from a template with clean and counterfactual substitutions.

        Args:
            template: Format string with placeholders, e.g.
                "Q: What month is {offset} months after {month}?\\nA:"
            clean_values: Dict of placeholder -> value for the clean input.
            cf_values: List of dicts, each overriding one or more clean values
                to produce a counterfactual. Unspecified keys inherit from
                clean_values.
        """
        clean_text = template.format(**clean_values)
        pairs = []
        for cf_overrides in cf_values:
            merged = {**clean_values, **cf_overrides}
            cf_text = template.format(**merged)
            pairs.append(CounterfactualPair(clean=clean_text, cf=cf_text))
        return cls(pairs)

    def validate(self, tokenizer, chat=False, seed_response=None):
        """Check that all pairs tokenize to the same length. Returns list of errors."""
        errors = []
        for i, pair in enumerate(self.pairs):
            clean_len = len(self._tokenize(pair.clean, tokenizer, chat, seed_response))
            cf_len = len(self._tokenize(pair.cf, tokenizer, chat, seed_response))
            if clean_len != cf_len:
                errors.append(
                    f"Pair {i}: clean={clean_len} tokens, cf={cf_len} tokens "
                    f"({pair.clean!r} vs {pair.cf!r})")
        return errors

    @staticmethod
    def _tokenize(text, tokenizer, chat=False, seed_response=None):
        if chat:
            messages = [{"role": "user", "content": text}]
            if seed_response:
                messages.append({"role": "assistant", "content": seed_response})
            rendered = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=seed_response is None,
                tokenize=False,
            )
            ids = tokenizer.encode(rendered, add_special_tokens=False)
            if seed_response:
                while ids and ids[-1] == tokenizer.eos_token_id:
                    ids.pop()
            return ids
        return tokenizer.encode(text)
