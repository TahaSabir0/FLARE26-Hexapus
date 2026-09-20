"""Per-task format conditioning constants shared by inference and data preparation."""

from __future__ import annotations

import re

LETTER_OPTION_PATTERN = re.compile(r"\b[A-E][.):]\s")

# Keyed by normalized task type (lowercase, underscores → spaces, hyphens preserved).
FORMAT_INSTRUCTIONS: dict[str, str] = {
    "classification": (
        "Answer with the option letter only (e.g., A, B, or C). "
        "Do not explain. Do not write full sentences."
    ),
    "multi-label classification": (
        "Select only the findings you can clearly see. "
        "List them separated by commas. "
        "Use exact label names from the provided options. "
        "Do not explain."
    ),
    "counting": (
        "Answer with a single integer only. "
        "Do not write units or explanation."
    ),
    "cell counting": (
        "Answer with a single integer only. "
        "Do not write units or explanation."
    ),
    "regression": (
        "Answer with a single numeric value only. "
        "Do not write units or explanation."
    ),
    "detection": (
        "Answer with a list of bounding boxes in format "
        "[[x_min, y_min, x_max, y_max], ...]. "
        "Include one entry per detected object. No explanation."
    ),
    "instance detection": (
        "Answer with a list of bounding boxes in format "
        "[[x_min, y_min, x_max, y_max], ...]. "
        "Include one entry per detected instance. No explanation."
    ),
    "report generation": (
        "Write a structured clinical report describing all relevant findings."
    ),
}

CLASSIFICATION_LETTER_INSTRUCTION: str = FORMAT_INSTRUCTIONS["classification"]
CLASSIFICATION_LABEL_INSTRUCTION: str = (
    "Answer with the exact label name only. "
    "Do not explain. Do not write full sentences."
)

MULTI_LABEL_BASE_INSTRUCTION: str = (
    "List only findings clearly visible. "
    "Separate with commas. "
    "Do not explain."
)

SYSTEM_PROMPT: str = (
    "You are a concise medical image analysis assistant. "
    "Always use the minimum required output format. "
    "Do not add explanations unless explicitly asked."
)

# Prefix seeds injected into the assistant turn at inference time (prefix forcing).
# Empty string disables seeding for that task type.
ANSWER_SEEDS: dict[str, str] = {
    "classification": "Answer: ",
    "multi-label classification": "",
    "counting": "",
    "cell counting": "",
    "regression": "",
    "detection": "[[",
    "instance detection": "[[",
    "report generation": "",
}


def normalize_task_type(task_type: str) -> str:
    """Return lowercase task type with underscores replaced by spaces."""
    return task_type.strip().lower().replace("_", " ")


def question_has_letter_options(question: str) -> bool:
    """Return True when the question contains lettered answer choices."""
    return bool(LETTER_OPTION_PATTERN.search(question or ""))


def get_format_instruction(
    task_type: str,
    question: str = "",
    valid_labels: list[str] | None = None,
) -> str:
    """Return a task-aware format instruction for a specific question."""
    norm_task_type = normalize_task_type(task_type)

    if norm_task_type == "classification":
        if question_has_letter_options(question):
            return CLASSIFICATION_LETTER_INSTRUCTION
        return CLASSIFICATION_LABEL_INSTRUCTION

    if norm_task_type == "multi-label classification":
        if valid_labels:
            labels = ", ".join(valid_labels)
            return (
                f"Valid labels for this dataset: [{labels}]\n"
                f"{MULTI_LABEL_BASE_INSTRUCTION}"
            )
        return FORMAT_INSTRUCTIONS[norm_task_type]

    return FORMAT_INSTRUCTIONS.get(norm_task_type, "")
