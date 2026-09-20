"""Qwen3.5-9B routed inference for FLARE-2026 Task-3 (v5 stack) - ported from
docker_qwen3/inference.py (the verified 8B container) with the gate-verified 9B path from
experiments/qwen35_sprint/eval_qwen35_arm.py. Architecture unchanged: a GLiClass question-router
picks one per-task LoRA adapter (general fallback when low-confidence), ONE 4-bit base VLM answers.

What changed vs the 8B container (see the paper):

  - Backbone: Qwen3-VL-8B-Instruct -> Qwen3.5-9B (4-bit nf4, double_quant=False to match the
    training YAML `double_quantization: false`).
  - Prompt: processor.apply_chat_template(..., add_generation_prompt=True, enable_thinking=False)
    -- byte-identity to the LF qwen3_5 training render is gate-verified (template_gate.py).
    NEVER leave enable_thinking unset (unset-defaults doctrine).
  - EOS PIN: Qwen3.5-9B ships NO generation_config.json -> generate() does not stop at
    <|im_end|> (measured 2026-08-31: correct first line, then a babbled fake user turn to the
    token budget). eos_token_id is pinned explicitly at EVERY generate call; training targets end
    with <|im_end|> so this matches training.
  - do_sample=False at EVERY generate call (the sampled-not-greedy lesson, 1e8bdc2).
  - PER-ROUTE RESOLUTION: adapters were trained at different pixel caps (2048^2 vs 768^2);
    train and inference resolution MUST match PER ROUTE. TASK_CONFIG carries
    (adapter, variant, image_max_pixels) and images are capped AFTER routing.
  - No DeepStack averaging anywhere (no 9B route uses it) -- _deepstack_average is not imported.
  - Router: gliclass==0.1.20 (0.1.11 does NOT load under transformers 5.16.1 -- two v4-to-v5 API
    breaks: implicit config.pad_token_id default removed + tie_weights(recompute_mapping=...)).
    0.1.20 loads clean and routes 15/15 identically (conf 1.0) to the 4.57.1 reference on real
    hidden-val questions (probe 2026-09-05).

Helper modules (vlm_prompt, prompt_variants, format_templates) come from the LLaMA-Factory scripts
dir; set FLARE_SCRIPTS_DIR (default: the repo scripts/ dir) or have them
copied beside this file in the Docker image.
"""
import os
import sys
import json
import argparse
import math
import re
from PIL import Image
from tqdm import tqdm
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig, AutoTokenizer
from peft import PeftModel

from gliclass import GLiClassModel, ZeroShotClassificationPipeline

# --- locate the shared FLARE helper modules (vlm_prompt etc.) ---
_DEFAULT_SCRIPTS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "scripts")
_SCRIPTS_DIR = os.environ.get("FLARE_SCRIPTS_DIR", _DEFAULT_SCRIPTS)
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)
from vlm_prompt import strip_think_block
from prompt_variants import get_variant
from format_templates import normalize_task_type

MAX_IMAGES = 4
PIX_2048 = 4194304   # 2048^2 -- the flagship training cap
PIX_768 = 589824     # 768^2  -- the ws-arm training cap

# Diagnostic: when LOG_ROUTING is set, record the router pick per case into the output.
_LOG_ROUTING = bool(os.environ.get("LOG_ROUTING"))

# ================================================================================
# PER-TASK ROUTING CONFIG  (router label -> adapter dir, prompt variant, image_max_pixels)
# ================================================================================
# Adapter dirs resolve under ADAPTERS_ROOT (default MODELS_DIR/qwen35_final).
# EVERY ADAPTER IS PINNED BY EXPLICIT NAME (the _full_s1337 name-collision lesson,
#   -- no tag templating. Each route is env-overridable so a single route can be
#   swapped without rebuilding helpers (e.g. CLS_ADAPTER=qwen35_cls_rebal_AB_2048_s1337).
# THE RESOLUTION IS PART OF THE ROUTE: (adapter, pixels) swap TOGETHER. Override pixels with
#   <ROUTE>_PIXELS only when swapping to an adapter trained at a different cap.
#
# THE AGREED ROUTE SET:
#   classification -> qwen35_cls_rebal_full_2048_s1337               @2048^2  bare (all-data, swapped 09-07)
#   multi-label    -> qwen35_multilabel_noavg_medcot_full_2048_s1337 @2048^2  medical_cot
#   counting       -> qwen35_counting_full_2048_s1337                  @2048^2  cot (WITH system --
#                     the 9B counting deck was trained WITH the cot system prompt, unlike the
#                     latent 8B bug where the deck was trained without it but served with it)
#   detection      -> qwen35_detection_full_2048_s1337               @2048^2  bare
#   regression     -> qwen35_regmix_A5_dermacls_AB_2048_s1337        @2048^2  bare (res CHANGED 768->2048, 09-07)
#   report_gen     -> qwen35_reportgen_AB_768_s1337                  @768^2   bare
#   fallback       -> qwen35_general_AB_768_s1337                    @768^2   bare
def _adapter(env_name, default):
    return (os.environ.get(env_name) or default).strip()


def _pixels(env_name, default):
    v = os.environ.get(env_name)
    return int(v) if v else default


TASK_CONFIG = {
    "classification":             (_adapter("CLS_ADAPTER",        "qwen35_cls_rebal_full_2048_s1337"),
                                   "none",        _pixels("CLS_PIXELS",        PIX_2048)),
    "detection":                  (_adapter("DETECTION_ADAPTER",  "qwen35_detection_full_2048_s1337"),
                                   "none",        _pixels("DETECTION_PIXELS",  PIX_2048)),
    "counting":                   (_adapter("COUNTING_ADAPTER",   "qwen35_counting_full_2048_s1337"),
                                   "cot",         _pixels("COUNTING_PIXELS",   PIX_2048)),
    "multi-label classification": (_adapter("MULTILABEL_ADAPTER", "qwen35_multilabel_noavg_medcot_full_2048_s1337"),
                                   "medical_cot", _pixels("MULTILABEL_PIXELS", PIX_2048)),
    "report_generation":          (_adapter("REPORTGEN_ADAPTER",  "qwen35_reportgen_AB_768_s1337"),
                                   "none",        _pixels("REPORTGEN_PIXELS",  PIX_768)),
    "regression":                 (_adapter("REGRESSION_ADAPTER", "qwen35_regmix_A5_dermacls_AB_2048_s1337"),
                                   "none",        _pixels("REGRESSION_PIXELS", PIX_2048)),
}
# drop any route explicitly disabled by setting its env var to "" -> falls through to GENERAL.
TASK_CONFIG = {k: v for k, v in TASK_CONFIG.items() if v[0]}
GENERAL = (_adapter("GENERAL_ADAPTER", "qwen35_general_AB_768_s1337"), "none",
           _pixels("GENERAL_PIXELS", PIX_768))
ROUTER_LABELS = list(TASK_CONFIG.keys()) + [_l for _l in ("regression", "instance_detection")
                                            if _l not in TASK_CONFIG]


# ================================================================================
# ANSWER PARSING (verbatim from docker_qwen3/inference.py -- backbone-agnostic)
# ================================================================================
def parse_answer(output, task_type=None):
    output = strip_think_block(output.strip())
    if "Please provide a clear and concise answer." in output:
        try:
            output = output.split("Please provide a clear and concise answer.")[-1].strip()
        except Exception:
            pass
    if "\n" in output:
        output = output.split("\n", 1)[-1].strip()
    task_type = (task_type or "").strip().lower()
    if task_type == "classification":
        return _parse_classification(output)
    elif task_type == "multi-label classification":
        return _parse_multi_label_classification(output)
    elif task_type in ["detection", "instance_detection"]:
        return _parse_detection(output)
    elif task_type in ["cell counting", "regression", "counting"]:
        return _parse_numeric(output)
    elif task_type in ["report generation", "report_generation"]:
        return output
    return output


def _parse_classification(output):
    lines = output.splitlines()
    return lines[-1].strip() if lines else output


def _parse_multi_label_classification(output):
    labels = []
    for line in output.splitlines():
        for part in re.split(r"[;]", line):
            if part.strip():
                labels.append(part.strip())
    return "; ".join(labels)


def _parse_detection(output):
    match = re.search(r"\{.*\}|\[.*\]", output, re.DOTALL)
    if match:
        try:
            return json.dumps(json.loads(match.group()))
        except Exception:
            return match.group()
    return output


def _parse_numeric(output):
    match = re.search(r"[-+]?[0-9]*\.?[0-9]+", output)
    return match.group() if match else "0"


def cap_pixels(im, mx):
    """Gate-verified resize (eval_qwen35_arm.py / template_gate.py convention: int() rounding,
    matches the LF training pad counts). NOT the 8B round() version."""
    w, h = im.size
    if mx and w * h > mx:
        f = math.sqrt(mx / (w * h))
        im = im.resize((max(1, int(w * f)), max(1, int(h * f))))
    return im


# ================================================================================
# MODEL LOADING
# ================================================================================
def load_model_and_processor(device="cuda:0"):
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

    models_dir = os.environ.get("MODELS_DIR", "/app/models")
    base_model_path = os.environ.get("BASE_MODEL", os.path.join(models_dir, "Qwen3.5-9B"))
    adapters_root = os.environ.get("ADAPTERS_ROOT", os.path.join(models_dir, "qwen35_final"))
    router_path = os.path.join(models_dir, "FLARE-gliclass-small-v1.0")
    for p in (base_model_path, adapters_root, router_path):
        if not os.path.exists(p):
            raise FileNotFoundError(f"required model path missing: {p}")

    print(f"Base: {base_model_path}\nAdapters root: {adapters_root}\nRouter: {router_path}")
    processor = AutoProcessor.from_pretrained(base_model_path, local_files_only=True)
    # double_quant=False matches the training YAML (double_quantization: false) -- the
    # gate-verified eval path (eval_qwen35_arm.py). Do not "optimize" this to True.
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=False,
                               bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    base = AutoModelForImageTextToText.from_pretrained(
        base_model_path, quantization_config=quant, dtype=torch.bfloat16,
        device_map=device, local_files_only=True)

    # load every per-task adapter (PeftModel name == TASK_CONFIG key) + general
    name_to_dir = {k: v[0] for k, v in TASK_CONFIG.items()}
    name_to_dir["general"] = GENERAL[0]
    model = None
    for name, d in name_to_dir.items():
        adir = os.path.join(adapters_root, d)
        if not os.path.isdir(adir):
            raise FileNotFoundError(f"adapter dir missing for '{name}': {adir}")
        if model is None:
            model = PeftModel.from_pretrained(base, adir, adapter_name=name,
                                              is_trainable=False, local_files_only=True)
        else:
            model.load_adapter(adir, adapter_name=name)
        print(f"  loaded adapter '{name}' <- {d}")
    model.eval()

    # EOS PIN (see module docstring): Qwen3.5-9B ships no generation_config.json.
    tok = processor.tokenizer
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    eos_ids = sorted({im_end, tok.eos_token_id} - {None})
    pad_id = tok.pad_token_id or im_end
    print(f"eos_token_id pinned: {eos_ids}  pad_token_id: {pad_id}")

    # Router: gliclass 0.1.20 under transformers 5.16.1 (eager attention kept from the 4.57 fix;
    # verified loading + routing 15/15 identical to the 4.57 reference, probe 2026-09-05).
    router_model = GLiClassModel.from_pretrained(
        router_path, local_files_only=True, attn_implementation="eager")
    router_tok = AutoTokenizer.from_pretrained(router_path, local_files_only=True)
    return model, processor, router_model, router_tok, eos_ids, pad_id


# ================================================================================
# ROUTING + PROMPTING
# ================================================================================
def route(router_model, router_tok, question, device):
    pipe = ZeroShotClassificationPipeline(router_model, router_tok,
                                          classification_type="single-label", device=device)
    res = pipe(question, ROUTER_LABELS, threshold=0.1)[0]
    return res[0]["label"], res[0]["score"]


def build_messages(variant_name, task_type, question, n_imgs):
    """Render the prompt EXACTLY as the 9B training deck export does (Image-i markers ->
    variant format instruction -> question) and return chat messages + the answer seed.
    Mirrors eval_qwen35_arm.py, whose byte-identity to the LF qwen3_5 render is gate-verified."""
    variant = None if variant_name == "none" else get_variant(variant_name)
    parts = ["\n".join(f"Image-{i + 1}: <image>" for i in range(n_imgs))]
    seed = ""
    if variant is not None:
        fi = variant.get_format_instruction(task_type or "", question)
        if fi:
            parts.append(fi)
        seed = variant.answer_seeds.get(normalize_task_type(task_type or ""), "")
    parts.append(question)
    human = "\n".join(parts)
    content = []
    segs = human.split("<image>")
    for i, seg in enumerate(segs):
        if seg:
            content.append({"type": "text", "text": seg})
        if i < len(segs) - 1:
            content.append({"type": "image"})
    messages = []
    if variant is not None and variant.system_prompt:
        messages.append({"role": "system", "content": variant.system_prompt})
    messages.append({"role": "user", "content": content})
    return messages, seed


def subsample(img_field):
    if isinstance(img_field, list):
        if len(img_field) > MAX_IMAGES:
            idx = sorted({round(i * (len(img_field) - 1) / (MAX_IMAGES - 1)) for i in range(MAX_IMAGES)})
            return [img_field[j] for j in idx]
        return img_field
    return [img_field]


# ================================================================================
# PREDICTION
# ================================================================================
def predict_one(s, fdir, model, processor, router_model, router_tok, eos_ids, pad_id,
                max_new_tokens=512):
    """Answer ONE sample in place (routing -> per-route resolution cap -> per-task knobs ->
    pinned-greedy generation -> strip-then-parse)."""
    try:
        # route FIRST -- the routed task decides the image resolution cap (per-route train res).
        label, score = route(router_model, router_tok, s["Question"], model.device)
        if score >= 0.3 and label in TASK_CONFIG:
            adapter_name = label
            _, variant_name, max_pix = TASK_CONFIG[label]
        else:
            adapter_name, variant_name, max_pix = "general", GENERAL[1], GENERAL[2]
        if _LOG_ROUTING:
            s["RoutedTask"] = adapter_name
            s["RouteLabel"] = label
            s["RouteScore"] = round(float(score), 4)

        imgs = []
        for p in subsample(s["ImageName"]):
            try:
                imgs.append(cap_pixels(Image.open(os.path.join(fdir, p)).convert("RGB"), max_pix))
            except Exception as e:
                print(f"img fail {p}: {e}")
        if not imgs:
            s["Answer"] = "Error: No valid images"
            return s

        model.set_adapter(adapter_name)

        # route/prompt/parse by the ROUTED label ONLY -- input TaskType is NOT available at
        # inference (challenge rule); it passes through to the output as untouched metadata.
        task_type = label
        messages, seed = build_messages(variant_name, task_type, s["Question"], len(imgs))
        text = processor.apply_chat_template(messages, tokenize=False,
                                             add_generation_prompt=True, enable_thinking=False)
        if seed:
            text += seed
        inputs = processor(text=[text], images=imgs, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            # do_sample=False (greedy) + eos pin: BOTH are required, neither is a default here.
            gen = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                 eos_token_id=eos_ids, pad_token_id=pad_id)
        raw = processor.batch_decode(gen[:, inputs["input_ids"].shape[1]:],
                                     skip_special_tokens=True)[0]
        s["Answer"] = parse_answer(seed + raw, task_type)
    except Exception as e:
        print(f"sample fail: {e}")
        s["Answer"] = f"Error: {e}"
    return s


def predict_on_file(input_file, model, processor, router_model, router_tok, eos_ids, pad_id,
                    max_new_tokens=512):
    with open(input_file) as f:
        data = json.load(f)
    fdir = os.path.dirname(input_file)
    for s in tqdm(data, desc=os.path.basename(input_file)):
        predict_one(s, fdir, model, processor, router_model, router_tok, eos_ids, pad_id,
                    max_new_tokens)
    return data


def find_json_files(base_path):
    out = []
    for root, _dirs, files in os.walk(base_path):
        for fn in files:
            if fn.endswith(".json"):
                out.append(os.path.join(root, fn))
    return out


def main():
    ap = argparse.ArgumentParser(description="Qwen3.5-9B routed prediction (FLARE-2026 Task-3)")
    ap.add_argument("--base_dataset_path", default="/workspace/inputs")
    ap.add_argument("--output_dir", default="/workspace/outputs")
    ap.add_argument("--output_filename", default="predictions.json")
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    files = find_json_files(args.base_dataset_path)
    if not files:
        raise FileNotFoundError(f"No JSON files in {args.base_dataset_path}")
    print(f"Found {len(files)} input JSON files")

    model, processor, router_model, router_tok, eos_ids, pad_id = \
        load_model_and_processor(args.device)
    all_preds = []
    for f in files:
        all_preds.extend(predict_on_file(f, model, processor, router_model, router_tok,
                                         eos_ids, pad_id, args.max_new_tokens))
    os.makedirs(args.output_dir, exist_ok=True)
    out = os.path.join(args.output_dir, args.output_filename)
    json.dump(all_preds, open(out, "w"), indent=2)
    print(f"Saved {len(all_preds)} predictions -> {out}")


if __name__ == "__main__":
    main()
