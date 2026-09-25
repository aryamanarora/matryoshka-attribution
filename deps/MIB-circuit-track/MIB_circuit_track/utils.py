TASKS_TO_HF_NAMES = {
    'ioi': 'ioi',
    'mcqa': 'copycolors_mcqa',
    'arithmetic_addition': 'arithmetic_addition',
    'arithmetic_subtraction': 'arithmetic_subtraction',
    'arc_easy': 'arc_easy',
    'arc_challenge': 'arc_challenge',
}

MODEL_NAME_TO_FULLNAME = {
    "gpt2": "gpt2-small",
    "qwen2.5": "Qwen/Qwen2.5-0.5B",
    "gemma2": "google/gemma-2-2b",
    "llama3": "meta-llama/Llama-3.1-8B",
}

"""
This script will print a table of the following form:
Method      | IOI (GPT) | IOI (QWen) | IOI (Gemma) | IOI (Llama) | MCQA (QWen) | MCQA (Gemma) | MCQA (Llama) | Arithmetic (Llama) | ARC-E (Gemma) | ARC-E (Llama) | ARC-C (Llama)
Random      |
Method 1    |
Method 2    |
...
"""

COL_MAPPING = {
    "ioi_gpt2": 0, "ioi_qwen2.5": 1, "ioi_gemma2": 2, "ioi_llama3": 3,
    "mcqa_qwen2.5": 4, "mcqa_gemma2": 5, "mcqa_llama3": 6,
    "arithmetic-addition_llama3": 7, "arithmetic-subtraction_llama3": 8,
    "arc-easy_gemma2": 9, "arc-easy_llama3": 10,
    "arc-challenge_llama3": 11,
    "ioi_interpbench": None,
}