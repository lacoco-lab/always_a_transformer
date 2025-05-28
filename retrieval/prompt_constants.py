from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

############################################
# Constants
############################################

SETTINGS: Tuple[str, ...] = (
    "UL", "UR", "NLFirst", "NRFirst", "NLLast", "NRLast",
)
# A single prompt variant will take one of the following forms
PROMPT_VARIANTS: Dict = {
    'space': [True, False],
    'few_shot': ['small', 'same'],
    'template': ['bare', 'simple_rule', 'math_rule'],
}

STOP_SEQUENCE = "\n"
MAX_COMPLETION_TOKENS = 6
MAX_INSTRUCT_TOKENS = 20

UNIQUE_PATH = 'datasets/retrieval/unique'
UNIQUE_FEWSHOT = 'datasets/retrieval/unique/sample_100_len_6_seed_2025.jsonl'

NON_UNIQUE_PATH = 'datasets/retrieval/nonunique'
NON_UNIQUE_FEWSHOT = 'datasets/retrieval/nonunique/sample_1500_len_7_seed_20.jsonl'

SYSTEM_PROMPT = ('You are a very careful and precise assistant. '
                 'You always follow the instructions without caring about what the task '
                 'and how it is formulated. You always solve tasks yourself. You never generate code.')

FORMAT_PROMPT = ('Put the final answer between <target> and </target> tags. '
                 'Start your response with the answer first and only then explain if needed.')
# ---------------------------------------------------------------------------
# Task catalogue
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TaskRule:
    """Container for the plain English and mathematical rule strings."""
    key: str
    rule_simple: str
    rule_math: str

TASKS: Dict[str, TaskRule] = {
    # Unique‑left / unique‑right ------------------------------------------------
    "UL": TaskRule(
        key="UL",
        rule_simple="The answer is the token immediately to the left of the single instance of the query token.",
        rule_math=r"$t = 1$ and $x_{n+1} = x_{q_1-1}$",
    ),
    "UR": TaskRule(
        key="UR",
        rule_simple="The answer is the token immediately to the right of the single instance of the query token.",
        rule_math=r"$t = 1$ and $x_{n+1} = x_{q_1+1}$",
    ),
    # Non‑unique / last‑occurrence --------------------------------------------
    "NLLast": TaskRule(
        key="NLLast",
        rule_simple="When the query appears multiple times, the answer is the token just to the left of its last appearance.",
        rule_math=r"$t > 1$ and $x_{n+1} = x_{q_t-1}$",
    ),
    "NRLast": TaskRule(
        key="NRLast",
        rule_simple="When the query appears multiple times, the answer is the token just to the right of its last appearance.",
        rule_math=r"$t > 1$ and $x_{n+1} = x_{q_t+1}$",
    ),
    # Non‑unique / first‑occurrence -------------------------------------------
    "NLFirst": TaskRule(
        key="NLFirst",
        rule_simple="When the query appears multiple times, the answer is the token just to the left of its first appearance.",
        rule_math=r"$t > 1$ and $x_{n+1} = x_{q_1-1}$",
    ),
    "NRFirst": TaskRule(
        key="NRFirst",
        rule_simple="When the query appears multiple times, the answer is the token just to the right of its first appearance.",
        rule_math=r"$t > 1$ and $x_{n+1} = x_{q_1+1}$",
    ),
}

# ---------------------------------------------------------------------------
# Prompt templates (no imperatives!)
# ---------------------------------------------------------------------------
# Place‑holders {examples}, {rule_simple}, {rule_math} are filled by build_prompt().
TEMPLATES: Dict[str, str] = {
    # 1) Bare –– only the examples ------------------------------------------
    "bare": """{examples}""",

    # 2) Simple English rule with plain wording ----------------------------
    "simple_rule": (
        """Each line is written as ‘context|query: target’. The vertical bar ‘|’ separates the context from the query token.
        \nAll strings below follow this rule: {rule_simple}\n\n{examples}"""
    ),

    # 3) Formal mathematical definition + rule -----------------------------
    "math_rule": (
        r"""Let $X = x_1 \ldots x_n$ with $n \ge 4$ and $x_i \in \Sigma$ (token vocabulary).  The final token $x_n$ is the query token $q$.\n"
        r"In the context $x_1 \ldots x_{{n-1}}$, $q$ appears $t$ times at indices $q_1, \ldots, q_t$ ($1 \le q_1 \le q_t \le n-1$).\n\n"
        r"Continuation token $x_{{n+1}}$ is defined by: {rule_math}\n\n"
        r"Examples:\n"
        r"{examples}"""
    ),

    # 4) Simple rule + worked-example explanation
    "simple_rule_explained": (
        "Each line is written as context|query: target. The vertical bar | marks the query token.\n"
        "All strings below follow this rule:\n"
        "{rule_simple}\n\n"
        "Worked example\n"
        "{explanation}\n\n"
        "Now continue the sequence (fill in the targets):\n\n"
        "{examples}"
    ),

    # 5) Math rule + worked-example explanation
    "math_rule_explained": (
        r"Let $X = x_1 \ldots x_n$ with $n \ge 4$ and $x_i \in \Sigma$ (token vocabulary). "
        r"The final token $x_n$ is the query token $q$.\n"
        r"In the context $x_1 \ldots x_{{n-1}}$, $q$ appears $t$ times at indices "
        r"$q_1, \ldots, q_t$ ($1 \le q_1 \le q_t \le n-1$).\n\n"
        r"Continuation token $x_{{n+1}}$ is defined by:\n"
        r"{rule_math}\n\n"
        r"Example - \n"
        r"{explanation}\n\n"
        r"Strings:\n\n"
        r"{examples}"
    ),
}


INSTRUCT_TEMPLATES: Dict[str, str] = {
    # 1) Bare –– only the examples ------------------------------------------
    "bare": """{examples}""",

    # 2) Simple English rule with plain wording ----------------------------
    "simple_rule": (
        """Each line is written as ‘context|query: target’. The vertical bar ‘|’ separates the context from the query token.
        \nAll strings below follow this rule: {rule_simple}\n\n{examples}\n
        Based on the examples, what is the answer to the following context|query?\n {q}"""
    ),

    # 3) Formal mathematical definition + rule -----------------------------
    "math_rule": (
        r"""Let $X = x_1 \ldots x_n$ with $n \ge 4$ and $x_i \in \Sigma$ (token vocabulary).  The final token $x_n$ is the query token $q$.\n"
        r"In the context $x_1 \ldots x_{{n-1}}$, $q$ appears $t$ times at indices $q_1, \ldots, q_t$ ($1 \le q_1 \le q_t \le n-1$).\n\n"
        r"Continuation token $x_{{n+1}}$ is defined by: {rule_math}\n\n"
        r"Examples:\n"
        r"{examples}"""
    ),

    # 4) Simple rule + worked-example explanation
    "simple_rule_explained": (
        "Each line is written as ‘context|query: target’. The vertical bar ‘|’ separates the context from the query token.\n"
        "All strings below follow this rule:\n"
        "{rule_simple}\n\n"
        "Worked example:\n"
        "{explanation}\n\n"
        "Based on the examples, what is the answer to the following context|query?\n {q}"
    ),

    # 5) Math rule + worked-example explanation
    "math_rule_explained": (
        r"Let $X = x_1 \ldots x_n$ with $n \ge 4$ and $x_i \in \Sigma$ (token vocabulary). "
        r"The final token $x_n$ is the query token $q$.\n"
        r"In the context $x_1 \ldots x_{{n-1}}$, $q$ appears $t$ times at indices "
        r"$q_1, \ldots, q_t$ ($1 \le q_1 \le q_t \le n-1$).\n\n"
        r"Continuation token $x_{{n+1}}$ is defined by:\n"
        r"{rule_math}\n\n"
        r"Example - \n"
        r"{explanation}\n\n"
        r"Strings:\n\n"
        r"{examples}"
    ),
}



MODELS: Dict[str, str] = {
    'llama3_8B': "/local/common_models/Llama-3.1-8B", 
    'llama3_70B': "/local/common_models/Llama-3.1-70B",
    'gemma3_1B': "/local/common_models/gemma-3-1b-pt",
    'gemma3_27B': "/local/common_models/gemma-3-27b-pt",
    'qwen2.5_7B': "/local/common_models/Qwen2.5-7B",
    'qwen2.5_32B': "/local/common_models/Qwen2.5-32B",
    'llama3_8B_instruct': "/local/common_models/Llama-3.1-8B-Instruct",
    'llama3_70B_instruct': "/local/common_models/Llama-3.3-70B-Instruct",
    'qwen2.5_7B_instruct': "/local/common_models/Qwen2.5-7B-Instruct",
    'qwen2.5_32B_instruct': "/local/common_models/Qwen2.5-32B-Instruct",
    'llama3_8B_instruct_hf': "meta-llama/Llama-3.1-8B-Instruct",
}
