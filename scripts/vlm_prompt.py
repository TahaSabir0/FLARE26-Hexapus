"""Backbone-aware ChatML prompt construction shared by eval_cls_arm.py and infer_single_adapter.py.

The eval scripts build the prompt MANUALLY (not via processor.apply_chat_template) so it is
byte-identical to what LLaMA-Factory's training template emits — train<->infer byte-identity is
essential or the eval is silently wrong (this bit us on InternVL3.5; see INTERNVL3_5_NOTES.md).
Two pieces vary by backbone, and both are derived from `processor.image_token`:

  * IMAGE PLACEHOLDER written into the user turn. LF's mm_plugin rewrites each literal "<image>" in
    the deck before tokenizing, so the manual prompt must write the SAME post-rewrite string:
      - InternVL3   (transformers 4.52.0.dev0): processor.image_token == "<image>"
      - InternVL3.5 (transformers 4.52.4):      processor.image_token == "<IMG_CONTEXT>"
        Both are a single bare placeholder the processor expands to per-tile context tokens
        (InternVLPlugin wraps internally as <img>..</img> at tokenize time — the bare token is right).
      - Qwen2 / Qwen2.5-VL: processor.image_token == "<|image_pad|>", BUT the qwen2_vl LF plugin
        rewrites each image as "<|vision_start|><|image_pad|><|vision_end|>"
        (src/llamafactory/data/mm_plugin.py Qwen2VLPlugin L1488). The HF processor then expands the
        single pad to grid_thw.prod()//merge_size**2 pads in place. Writing a BARE "<|image_pad|>"
        omits the vision markers -> wrong token-id sequence -> silently broken eval.

  * SYSTEM turn. LF injects the template's default_system when the deck row carries no system message
    (our decks never do). So the manual system string must match the template default:
      - intern_vl template: default_system == ""                         -> empty system
      - qwen2_vl  template: default_system == "You are a helpful assistant." -> non-empty system

The Phase-0 smoke's byte-identity check (compare ids of the LF-rendered training row vs the prompt
this builds) confirms these choices before any Phase-1 training is trusted.
"""

import re

QWEN_IMAGE_TOKEN = "<|image_pad|>"
QWEN_DEFAULT_SYSTEM = "You are a helpful assistant."

_THINK_BLOCK_RE = re.compile(r"^\s*<think>.*?</think>\s*", re.DOTALL)
_ORPHAN_CLOSE_RE = re.compile(r"^\s*</think>\s*")


def strip_think_block(text):
    """Strip a leading Qwen3-VL reasoning-channel artifact from a generated answer.

    ROOT CAUSE (traced 2026-07-30 -- the older comment here was WRONG, (see the paper):
    the tag does NOT come from Qwen3-VL. The HF chat_template.json contains ZERO occurrences of
    "think", and Qwen3-VL-8B-Instruct is the NON-thinking edition. It comes from OUR training
    config: LLaMA-Factory registers `template: qwen3_vl` with `template_class=ReasoningTemplate`
    (LF data/template.py:1946) and `enable_thinking` defaults to True (hparams/data_args.py:122),
    which we never override. ReasoningTemplate.encode_oneturn then does, for every row whose
    target lacks thought tags:
        response_ids = get_thought_word_ids(tokenizer) + response_ids   # "# do compute loss"
    i.e. it prepends an EMPTY <think>\n\n</think>\n\n to the LOSS-COMPUTED target. So we
    supervised ~58k rows to emit an empty reasoning block before the answer. (The
    `enable_thinking=False` branch instead appends it to the PROMPT with no loss -- the standard
    force-skip-thinking trick, and the alternative root-cause fix. `qwen3_vl_nothink` also exists.)

    Observed effect: the OPEN-ENDED tasks (multi-label, report generation) emit the block at
    inference; the short tasks (cls/det/count/reg) empirically do not, so this is a no-op there
    (why the short tasks skip it is still unexplained). The block is always empty with clean
    training targets (verified), so stripping never removes real content.
    Without it, the multi-label scorer (splits on [;,]) counts </think> as a spurious predicted
    label on every case (~ -0.09 micro-F1).

    KEEP THIS STRIP even if the root cause is ever fixed: `predictions.json` must contain no
    </think> anywhere (submission schema), so it is also a required defensive layer.

    NOTE for any future CoT work: ReasoningTemplate only injects the empty block when the target
    has NO thought tags. Put a real trace in the ShareGPT target and LF passes it through and
    computes loss on it -- target-side CoT needs no infra change.
    """
    if not isinstance(text, str):
        return text
    text = _THINK_BLOCK_RE.sub("", text)
    text = _ORPHAN_CLOSE_RE.sub("", text)
    return text.strip()


def backbone_format(processor):
    """Return {'system', 'image_placeholder'} matching the LF training template for this backbone.

    Keyed off processor.image_token (image placeholder) + processor class (system), so it
    auto-selects InternVL vs Qwen2.x vs Qwen3-VL with no model flag.

    NOTE on Qwen2 vs Qwen3: both expose image_token == "<|image_pad|>" and BOTH LF plugins
    (Qwen2VLPlugin / Qwen3VLPlugin) rewrite the image identically as
    "<|vision_start|><|image_pad|><|vision_end|>" (Qwen3VLPlugin subclasses Qwen2VLPlugin and only
    overrides the VIDEO timestamp path; the image wrap is inherited verbatim). They differ ONLY in
    the template default_system: the LF `qwen2_vl` template sets
    default_system="You are a helpful assistant." while the `qwen3_vl` template sets NO default_system
    (-> empty ""). Our decks carry no system turn, so LF injects the template default at train time;
    the eval system string must match per backbone or train<->infer diverge (the byte-identity gate
    catches this). We distinguish them by the processor class name (Qwen3VLProcessor vs
    Qwen2_5_VLProcessor / Qwen2VLProcessor)."""
    itok = getattr(processor, "image_token", None) or "<image>"
    if itok == QWEN_IMAGE_TOKEN:
        # Qwen3-VL template has NO default_system (empty); Qwen2/2.5-VL default is "You are a helpful assistant."
        is_qwen3 = "Qwen3" in type(processor).__name__
        if is_qwen3:
            # qwen3_vl template has empty default_system -> LF's _encode omits the whole system
            # block when system is empty (`if system or tools`). omit_system reproduces that.
            return {"system": "", "omit_system": True,
                    "image_placeholder": f"<|vision_start|>{itok}<|vision_end|>"}
        return {"system": QWEN_DEFAULT_SYSTEM, "omit_system": False,
                "image_placeholder": f"<|vision_start|>{itok}<|vision_end|>"}
    # InternVL3 ("<image>") / InternVL3.5 ("<IMG_CONTEXT>"): bare placeholder, empty system block
    # (InternVL training render carries an (empty) system block -> keep it; do not omit).
    return {"system": "", "omit_system": False, "image_placeholder": itok}


def build_chatml(system, user_content, omit_system=False):
    """ChatML wrapper. InternVL / Qwen2.5-VL emit a system block (even if empty for InternVL); Qwen3-VL
    omits it (omit_system=True) because its LF template has empty default_system and LF only emits the
    system turn when non-empty."""
    sys_block = "" if omit_system else f"<|im_start|>system\n{system}<|im_end|>\n"
    return (f"{sys_block}"
            f"<|im_start|>user\n{user_content}<|im_end|>\n"
            f"<|im_start|>assistant\n")
