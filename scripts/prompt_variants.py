"""Prompt variants shared by data preparation and inference.

The registry intentionally contains only the three runs used in the 10% prompt
ablation: bare_baseline, baseline, and cot.
"""

from __future__ import annotations

from dataclasses import dataclass

from format_templates import ANSWER_SEEDS, FORMAT_INSTRUCTIONS, SYSTEM_PROMPT


DEFAULT_MAX_TOKENS: dict[str, int] = {
    "classification": 512,
    "multi-label classification": 512,
    "counting": 512,
    "cell counting": 512,
    "regression": 512,
    "detection": 512,
    "instance detection": 512,
    "report generation": 512,
}


@dataclass(frozen=True)
class PromptVariant:
    """Complete prompt configuration for one experimental variant."""

    name: str
    description: str
    system_prompt: str
    format_instructions: dict[str, str]
    answer_seeds: dict[str, str]
    max_new_tokens: dict[str, int]
    # Veronika's H4 pixel-grounding: a dedicated classification instruction for the no-letter-options
    # case. When the cls format_instruction itself contains "option letter" (the pixel prompt does),
    # the generic rewrite below would mangle it -> she supplies a clean label-name variant instead.
    classification_label_instruction: str | None = None

    def get_format_instruction(
        self,
        task_type: str,
        question: str = "",
        valid_labels: list[str] | None = None,
    ) -> str:
        """Return a task-aware format instruction."""
        from format_templates import (
            CLASSIFICATION_LABEL_INSTRUCTION,
            CLASSIFICATION_LETTER_INSTRUCTION,
            MULTI_LABEL_BASE_INSTRUCTION,
            normalize_task_type,
            question_has_letter_options,
        )

        norm = normalize_task_type(task_type)

        if norm == "classification":
            base = self.format_instructions.get(
                "classification", CLASSIFICATION_LETTER_INSTRUCTION
            )
            if not base:
                return ""
            if not question_has_letter_options(question):
                if self.classification_label_instruction is not None:
                    return self.classification_label_instruction
                if base == FORMAT_INSTRUCTIONS["classification"]:
                    return CLASSIFICATION_LABEL_INSTRUCTION
                return base.replace(
                    "where X is the correct option letter",
                    "where X is the exact label name",
                ).replace("option letter", "exact label name")
            return base

        if norm == "multi-label classification":
            base = self.format_instructions.get(
                "multi-label classification", MULTI_LABEL_BASE_INSTRUCTION
            )
            if not base:
                return ""
            if valid_labels:
                labels = ", ".join(valid_labels)
                return f"Valid labels for this dataset: [{labels}]\n{base}"
            return base

        return self.format_instructions.get(norm, "")

    def get_max_new_tokens(self, task_type: str, fallback: int = 512) -> int:
        """Return the token budget for a given task type."""
        from format_templates import normalize_task_type

        return self.max_new_tokens.get(normalize_task_type(task_type), fallback)


COT_FORMAT_INSTRUCTIONS: dict[str, str] = {
    "classification": (
        "Examine the image carefully for relevant clinical features, "
        "then write 'Answer: X' where X is the correct option letter."
    ),
    "multi-label classification": (
        "For each listed finding, assess whether it is clearly visible in the image. "
        "List only confirmed findings separated by commas."
    ),
    "counting": (
        "Methodically count each object in the image (scan left-to-right, row by row). "
        "Write the final count as a single integer."
    ),
    "cell counting": (
        "Methodically count each cell in the image (scan left-to-right, row by row). "
        "Write the final count as a single integer."
    ),
    "regression": (
        "Estimate or measure the requested value from the image. "
        "Write a single numeric value."
    ),
    "detection": (
        "Locate every instance of the target object in the image. "
        "Write bounding boxes as [[x_min,y_min,x_max,y_max],...], one entry per object."
    ),
    "instance detection": (
        "Locate every instance of the target object in the image. "
        "Write bounding boxes as [[x_min,y_min,x_max,y_max],...], one entry per object."
    ),
    "report generation": (
        "Write a structured clinical report. "
        "Begin with 'Findings:' describing all visible findings, "
        "then 'Impression:' with the clinical interpretation."
    ),
}


# --- Veronika H4 pixel-grounding (verbatim from veronika/experiment/h4-pixel-ab-prime-cv) ----------
# Asks the model to INTERNALLY localize visual evidence (lesion pixels, contours, high-contrast
# regions) before answering -- no boxes are emitted. ADOPTED for the CLASSIFICATION adapter only
# (TRUTH.md §5.3: +0.0071 / +0.0458 BA both directions; multi-label neutral, counting mixed). The
# variant carries instructions for several tasks, but production routes ONLY classification through
# it (--prompt_variant pixel_grounded_multi_adapter on the cls adapter); every other adapter stays bare.
_PIXEL_DETECTION = (
    "Return lesion bounding boxes in original image pixel coordinates as "
    "[[x_min,y_min,x_max,y_max],...]. Use a tight rectangle around the visible "
    "ultrasound lesion or contour. Do not use normalized coordinates. Do not describe the image."
)
PIXEL_FORMAT_INSTRUCTIONS: dict[str, str] = {
    "classification": (
        "Before answering, localize the visible evidence in original image pixel space: "
        "look for lesion pixels, object boundaries, contours, high-contrast regions, and "
        "tight visual regions that support the class. Mentally place tight boxes around "
        "the relevant evidence, but do not output boxes. Return exactly one answer: the "
        "option letter if choices are shown, otherwise the exact class name. Do not describe the image."
    ),
    "multi-label classification": (
        "For each candidate label, search the original image pixels for visible evidence: "
        "lesion pixels, object boundaries, contours, high-contrast regions, and tight "
        "visual regions. Mentally place tight boxes around each finding, but do not output boxes. "
        "Return only the labels supported by visible pixel evidence using exact label names. "
        "Use Normal only when no abnormal finding is visible; Normal must be the only label. "
        "Do not describe the image."
    ),
    "counting": (
        "Count by localizing each visible object in original image pixel space. Treat each "
        "object as a tight pixel region or tight box; separate touching regions when their "
        "contours indicate different objects. Scan the full image and avoid double-counting "
        "the same pixel region. Return one integer only. Do not describe the image."
    ),
    "detection": _PIXEL_DETECTION,
    "instance detection": _PIXEL_DETECTION,
}
PIXEL_FORMAT_INSTRUCTIONS["cell counting"] = PIXEL_FORMAT_INSTRUCTIONS["counting"]
PIXEL_CLASSIFICATION_LABEL = (
    "Before answering, localize the visible evidence in original image pixel space: "
    "look for lesion pixels, object boundaries, contours, high-contrast regions, and "
    "tight visual regions that support the class. Mentally place tight boxes around "
    "the relevant evidence, but do not output boxes. Return exactly one class name "
    "from the classes listed in the question. Do not describe the image."
)


_REGISTRY: dict[str, PromptVariant] = {
    "bare_baseline": PromptVariant(
        name="bare_baseline",
        description=(
            "Original-style prompt: image token plus dataset question only; "
            "no system prompt, no format instruction, no prefix seed."
        ),
        system_prompt="",
        format_instructions={},
        answer_seeds={},
        max_new_tokens=dict(DEFAULT_MAX_TOKENS),
    ),
    "baseline": PromptVariant(
        name="baseline",
        description="Format-conditioned baseline with concise task-specific instructions.",
        system_prompt=SYSTEM_PROMPT,
        format_instructions=dict(FORMAT_INSTRUCTIONS),
        answer_seeds=dict(ANSWER_SEEDS),
        max_new_tokens=dict(DEFAULT_MAX_TOKENS),
    ),
    "cot": PromptVariant(
        name="cot",
        description="Brief chain-of-thought-style task framing with concise final answers.",
        system_prompt="You are a medical image analysis expert. Think carefully before answering.",
        format_instructions=dict(COT_FORMAT_INSTRUCTIONS),
        answer_seeds=dict(ANSWER_SEEDS),
        max_new_tokens=dict(DEFAULT_MAX_TOKENS),
    ),
    "medical_cot": PromptVariant(
        name="medical_cot",
        description="Veronika's expert-radiologist system prompt + CoT format instructions. Tested only "
                    "stacked on the ML averaging specialist (its one both-directions effect is a small ML gain).",
        system_prompt=(
            "You are an expert radiologist and medical image analyst specialising in "
            "X-ray, CT, MRI, ultrasound, pathology, endoscopy, and microscopy. "
            "Think carefully and systematically before answering."
        ),
        format_instructions=dict(COT_FORMAT_INSTRUCTIONS),
        answer_seeds=dict(ANSWER_SEEDS),
        max_new_tokens=dict(DEFAULT_MAX_TOKENS),
    ),
    "pixel_grounded_multi_adapter": PromptVariant(
        name="pixel_grounded_multi_adapter",
        description="Veronika's pixel-grounded prompts (localize evidence before answering). "
                    "Production routes ONLY the classification adapter through this; keep bare elsewhere.",
        system_prompt="",
        format_instructions=dict(PIXEL_FORMAT_INSTRUCTIONS),
        answer_seeds={},
        max_new_tokens=dict(DEFAULT_MAX_TOKENS),
        classification_label_instruction=PIXEL_CLASSIFICATION_LABEL,
    ),
}


def get_variant(name: str) -> PromptVariant:
    """Return a PromptVariant by name."""
    if name not in _REGISTRY:
        known = ", ".join(sorted(_REGISTRY))
        raise ValueError(f"Unknown prompt variant {name!r}. Known variants: {known}")
    return _REGISTRY[name]


def list_variants() -> list[str]:
    """Return sorted list of registered variant names."""
    return sorted(_REGISTRY)
