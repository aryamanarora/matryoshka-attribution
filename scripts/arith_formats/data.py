"""Items, formats and number words for the cross-format arithmetic replication.

Replicates the item set of de Varda et al. (2026), "Shared circuits predict whether LLMs
generalize across formats in arithmetic reasoning" (arXiv:2609.04463):

  * fixed set of N items of the form ``a1 op a2 (op a3) =`` over positive integers,
    op in {+, -}; operands 2 or 3 digits (balanced), 2 or 3 terms (balanced), half of the
    items require a carry / borrow;
  * every item rendered in four formats: numeric, English, Spanish, Italian;
  * the sign-flipped alternative x' replaces EVERY + by - and vice versa.

Things the paper leaves unstated, and the choices made here (all flagged in the write-up):
  * results are kept >= 0 (operands are sampled/ordered so every partial result is >= 0);
  * the answer is written in the format's own surface form (digits for numeric, number
    words for the verbal formats); correctness = exact match after normalisation;
  * one-shot exemplar: a fixed 2-term addition ``53 + 28 = 81`` in the same format, one
    per line, followed by the query.
"""

import random
import re
import unicodedata

FORMATS = ("numeric", "english", "spanish", "italian")

# --------------------------------------------------------------------------- number words
_EN_ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
            "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
            "seventeen", "eighteen", "nineteen"]
_EN_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]


def en_words(n: int) -> str:
    if n < 0:
        return "minus " + en_words(-n)
    if n < 20:
        return _EN_ONES[n]
    if n < 100:
        t, o = divmod(n, 10)
        return _EN_TENS[t] + ("-" + _EN_ONES[o] if o else "")
    if n < 1000:
        h, r = divmod(n, 100)
        return _EN_ONES[h] + " hundred" + (" " + en_words(r) if r else "")
    th, r = divmod(n, 1000)
    return en_words(th) + " thousand" + (" " + en_words(r) if r else "")


_ES_ONES = ["cero", "uno", "dos", "tres", "cuatro", "cinco", "seis", "siete", "ocho", "nueve",
            "diez", "once", "doce", "trece", "catorce", "quince", "dieciséis", "diecisiete",
            "dieciocho", "diecinueve", "veinte", "veintiuno", "veintidós", "veintitrés",
            "veinticuatro", "veinticinco", "veintiséis", "veintisiete", "veintiocho",
            "veintinueve"]
_ES_TENS = ["", "", "veinte", "treinta", "cuarenta", "cincuenta", "sesenta", "setenta",
            "ochenta", "noventa"]
_ES_HUNDREDS = ["", "ciento", "doscientos", "trescientos", "cuatrocientos", "quinientos",
                "seiscientos", "setecientos", "ochocientos", "novecientos"]


def es_words(n: int) -> str:
    if n < 0:
        return "menos " + es_words(-n)
    if n < 30:
        return _ES_ONES[n]
    if n < 100:
        t, o = divmod(n, 10)
        return _ES_TENS[t] + (" y " + _ES_ONES[o] if o else "")
    if n == 100:
        return "cien"
    if n < 1000:
        h, r = divmod(n, 100)
        return _ES_HUNDREDS[h] + (" " + es_words(r) if r else "")
    th, r = divmod(n, 1000)
    head = "mil" if th == 1 else es_words(th) + " mil"
    return head + (" " + es_words(r) if r else "")


_IT_ONES = ["zero", "uno", "due", "tre", "quattro", "cinque", "sei", "sette", "otto", "nove",
            "dieci", "undici", "dodici", "tredici", "quattordici", "quindici", "sedici",
            "diciassette", "diciotto", "diciannove"]
_IT_TENS = ["", "", "venti", "trenta", "quaranta", "cinquanta", "sessanta", "settanta",
            "ottanta", "novanta"]


def _it_below_100(n: int) -> str:
    if n < 20:
        return _IT_ONES[n]
    t, o = divmod(n, 10)
    tens = _IT_TENS[t]
    if o == 0:
        return tens
    if o in (1, 8):          # ventuno, ventotto: drop the final vowel of the tens word
        tens = tens[:-1]
    if o == 3:
        return tens + "tré"
    return tens + _IT_ONES[o]


def it_words(n: int) -> str:
    if n < 0:
        return "meno " + it_words(-n)
    if n < 100:
        return _it_below_100(n)
    if n < 1000:
        h, r = divmod(n, 100)
        head = "cento" if h == 1 else _IT_ONES[h] + "cento"
        if r == 0:
            return head
        tail = _it_below_100(r)
        if tail[0] == "o":        # centotto, centottanta: the final o of cento elides
            head = head[:-1]
        return head + tail
    th, r = divmod(n, 1000)
    head = "mille" if th == 1 else it_words(th) + "mila"
    return head + (it_words(r) if r else "")


NUM_WORDS = {"english": en_words, "spanish": es_words, "italian": it_words}
OPS = {
    "numeric": {"+": "+", "-": "-", "=": "="},
    "english": {"+": "plus", "-": "minus", "=": "equals"},
    "spanish": {"+": "más", "-": "menos", "=": "es igual a"},
    "italian": {"+": "più", "-": "meno", "=": "fa"},
}
EXEMPLAR = ([53, 28], ["+"])        # 53 + 28 = 81, has a carry


def render_number(n: int, fmt: str) -> str:
    return str(n) if fmt == "numeric" else NUM_WORDS[fmt](n)


def render_problem(operands, ops, fmt: str) -> str:
    """``a1 op a2 (op a3) =`` in the given format, WITHOUT the answer or a trailing space."""
    parts = [render_number(operands[0], fmt)]
    for op, a in zip(ops, operands[1:]):
        parts += [OPS[fmt][op], render_number(a, fmt)]
    parts.append(OPS[fmt]["="])
    return " ".join(parts)


def evaluate(operands, ops) -> int:
    v = operands[0]
    for op, a in zip(ops, operands[1:]):
        v = v + a if op == "+" else v - a
    return v


def flip(ops):
    return ["-" if o == "+" else "+" for o in ops]


def _needs_carry(operands, ops) -> bool:
    """Left-to-right evaluation; True if any step carries (addition) or borrows (subtraction)."""
    v = operands[0]
    for op, a in zip(ops, operands[1:]):
        x, y = v, a
        carry = False
        if op == "+":
            c = 0
            while x or y:
                d = x % 10 + y % 10 + c
                if d >= 10:
                    carry = True
                c = d // 10
                x //= 10; y //= 10
        else:
            b = 0
            while x or y:
                d = x % 10 - y % 10 - b
                if d < 0:
                    carry = True
                b = 1 if d < 0 else 0
                x //= 10; y //= 10
        if carry:
            return True
        v = v + a if op == "+" else v - a
    return False


def make_items(n: int = 2000, seed: int = 0):
    """Balanced over (digits in {2,3}) x (terms in {2,3}) x (carry in {False,True}).

    Results (and every left-to-right partial result) are >= 0 for the ORIGINAL item; the
    sign-flipped alternative may be negative, which is fine because the alternative's gold
    answer is never used (the metric only uses the model's own answers).
    """
    rng = random.Random(seed)
    cells = [(d, t, c) for d in (2, 3) for t in (2, 3) for c in (False, True)]
    per = n // len(cells)
    items, seen = [], set()
    for d, t, c in cells:
        lo, hi = (10, 99) if d == 2 else (100, 999)
        got, tries = 0, 0
        while got < per:
            tries += 1
            if tries > 200 * per:
                raise ValueError(f"cell digits={d} terms={t} carry={c} cannot supply {per} unique items")
            operands = [rng.randint(lo, hi) for _ in range(t)]
            ops = [rng.choice("+-") for _ in range(t - 1)]
            v = operands[0]
            ok = True
            for op, a in zip(ops, operands[1:]):
                v = v + a if op == "+" else v - a
                if v < 0:
                    ok = False
                    break
            if not ok or _needs_carry(operands, ops) != c:
                continue
            key = (tuple(operands), tuple(ops))
            if key in seen:
                continue
            seen.add(key)
            items.append({"id": len(items), "operands": operands, "ops": ops,
                          "digits": d, "terms": t, "carry": c,
                          "answer": evaluate(operands, ops),
                          "answer_flip": evaluate(operands, flip(ops))})
            got += 1
    rng.shuffle(items)
    for i, it in enumerate(items):
        it["id"] = i
    return items


def exemplar_line(fmt: str) -> str:
    ops_, ops = EXEMPLAR
    return render_problem(ops_, ops, fmt) + " " + render_number(evaluate(ops_, ops), fmt)


def prompt(item, fmt: str, flipped: bool = False) -> str:
    ops = flip(item["ops"]) if flipped else item["ops"]
    return exemplar_line(fmt) + "\n" + render_problem(item["operands"], ops, fmt)


def gold(item, fmt: str, flipped: bool = False) -> str:
    return render_number(item["answer_flip"] if flipped else item["answer"], fmt)


def normalise(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return re.sub(r"[\s\-,]", "", s.lower())


def is_correct(generated: str, item, fmt: str) -> bool:
    """Exact match of the first line of the greedy continuation with the gold answer."""
    first = generated.strip().split("\n")[0].strip()
    first = first.rstrip(".")
    return normalise(first) == normalise(gold(item, fmt))


if __name__ == "__main__":
    for n in [0, 1, 8, 13, 21, 23, 28, 31, 44, 66, 100, 101, 108, 123, 180, 181, 200, 500,
              700, 999, 1000, 1001, 1108, 2000, 2997]:
        print(n, "|", en_words(n), "|", es_words(n), "|", it_words(n))
    items = make_items(2000)
    from collections import Counter
    print(Counter((i["digits"], i["terms"], i["carry"]) for i in items))
    for it in items[:3]:
        for f in FORMATS:
            print(repr(prompt(it, f)), "->", gold(it, f), "| flip:", repr(prompt(it, f, True)))
