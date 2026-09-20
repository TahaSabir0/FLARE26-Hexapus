"""Single general-adapter inference for FLARE-2026 general-adapter eval (backbone-configurable).

Loads a base VLM (4-bit) + ONE LoRA adapter; NO GLiClass router / no adapter swap — this is the
"general fallback" adapter the deployed system uses when the router is low-confidence. Mirrors
training: committed Image-i: per-image markers + blind uniform MAX_IMAGES=4 frame cap. Reuses
parse_answer verbatim from docker_internvl/inference.py.

Backbone-agnostic: pass --base_model (InternVL3-8B-hf / InternVL3_5-8B-HF / Qwen2.5-VL-7B-Instruct).
The system turn + image placeholder come from vlm_prompt.backbone_format keyed off processor.image_token
(InternVL bare placeholder + empty system; Qwen2.5-VL vision-wrapped pad + "You are a helpful assistant."
default system), so the same markered prompt stays byte-identical to each backbone's LF training template.
This is the fold-into-repo + multi-backbone adaptation of the cluster-only ~/genadapter_eval harness.

Eval-set schemas:
  --eval_set valpub : walk validation-public/**/*.json (TaskType/Modality/ImageName/Question/Answer)
  --eval_set jsonl  : read a self-contained split JSONL (true_test.jsonl / A.jsonl / B.jsonl;
                      keys task/answer/question/image/dataset, absolute image paths)

Emits a flat predictions.json list with ImageName, Question, TaskType, dataset, gt_answer, Answer.
"""
import os, json, argparse, glob, re, sys
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig

from vlm_prompt import backbone_format, build_chatml
from prompt_variants import list_variants  # registry-derived --prompt_variant choices (no stale hardcoded list)

# repo root on sys.path so the env-gated border-crop import (experiments.*) resolves (mirrors eval_cls_arm.py)
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

MAX_IMAGES = 4

# Alberto's matched border-crop (QWEN3_BORDER_CROP, Qwen3 classification route only). Env-gated and
# OFF by default; must mirror training (mm_plugin._maybe_apply_qwen3_border_crop). When the cls
# adapter is trained crop-on, inference MUST run crop-on or train<->infer mismatch. The crop is the
# same utility / same position (before the processor) as eval_cls_arm.py, so behavior is matched by
# construction (Alberto validated via eval_cls_arm.py; this extends the rule to the general path).
_BORDER_CROP = os.environ.get("QWEN3_BORDER_CROP", "").lower() in {"1", "true", "yes", "on"}

def _maybe_border_crop(img):
    if not _BORDER_CROP:
        return img
    from experiments.qwen3_matched_border_crop.crop_preprocessing import apply_conservative_border_crop
    cropped, _ = apply_conservative_border_crop(img)
    return cropped

def _cap_pixels(img, max_pix, min_pix=1024):
    import math
    if not max_pix or max_pix <= 0:
        return img
    w, h = img.size
    if w * h > max_pix:
        rf = math.sqrt(max_pix / (w * h)); img = img.resize((max(1, round(w*rf)), max(1, round(h*rf))))
    w, h = img.size
    if w * h < min_pix:
        rf = math.sqrt(min_pix / (w * h)); img = img.resize((max(1, round(w*rf)), max(1, round(h*rf))))
    return img

# ---- parse_answer copied verbatim from committed inference.py ----
def parse_answer(output, task_type=None):
    output = output.strip()
    if "Please provide a clear and concise answer." in output:
        try:
            output = output.split("Please provide a clear and concise answer.")[-1].strip()
        except: pass
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
    elif task_type == "report generation":
        return output
    else:
        return output

def _parse_classification(output):
    lines = output.splitlines()
    if len(lines) >= 1:
        return lines[-1].strip()
    return output

def _parse_multi_label_classification(output):
    lines = output.splitlines(); labels = []
    for line in lines:
        for part in re.split(r"[;]", line):
            label = part.strip()
            if label: labels.append(label)
    return "; ".join(labels)

def _parse_detection(output):
    match = re.search(r"\{.*\}|\[.*\]", output, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group()); return json.dumps(parsed)
        except: return match.group()
    return output

def _parse_numeric(output):
    match = re.search(r"[-+]?[0-9]*\.?[0-9]+", output)
    if match: return match.group()
    return "0"
# ---- end verbatim ----

def load_model(base_model, adapter_path, device):
    os.environ["HF_HUB_OFFLINE"]="1"; os.environ["TRANSFORMERS_OFFLINE"]="1"; os.environ["HF_HUB_DISABLE_TELEMETRY"]="1"
    from peft import PeftModel
    print(f"Base: {base_model}\nAdapter: {adapter_path}")
    processor = AutoProcessor.from_pretrained(base_model, local_files_only=True)
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_storage=torch.bfloat16)
    base = AutoModelForImageTextToText.from_pretrained(base_model,
        quantization_config=quant, torch_dtype=torch.bfloat16, device_map="auto", local_files_only=True)
    # adapter_path in {none,base,""} => ZERO-SHOT base model, no PEFT (the PGF baseline anchor for the
    # task-transfer matrix: Acc(M, T_j) with no finetuning). Any real path loads the adapter as before.
    if adapter_path and str(adapter_path).lower() not in ("none", "base", ""):
        model = PeftModel.from_pretrained(base, adapter_path, adapter_name="general", is_trainable=False)
    else:
        model = base
        print("No adapter -- ZERO-SHOT base model (PGF baseline anchor).")
    model.eval()
    fmt = backbone_format(processor)   # backbone-aware system + image placeholder (vlm_prompt)
    print(f"Single general adapter loaded; no router. "
          f"image placeholder: {fmt['image_placeholder']!r} | system: {fmt['system']!r}")
    return model, processor, fmt

def subsample(img_field):
    if isinstance(img_field, list):
        if len(img_field) > MAX_IMAGES:
            idx = sorted({round(i*(len(img_field)-1)/(MAX_IMAGES-1)) for i in range(MAX_IMAGES)})
            return [img_field[j] for j in idx]
        return img_field
    return [img_field]

def build_samples(eval_set, base):
    out = []
    if eval_set == "valpub":
        root = base
        for f in glob.glob(os.path.join(root, "**", "*.json"), recursive=True):
            dsname = os.path.basename(os.path.dirname(f))
            d = json.load(open(f)); fdir = os.path.dirname(f)
            for s in d:
                imgs = subsample(s["ImageName"])
                out.append(dict(ImageName=s["ImageName"], Question=s["Question"],
                    TaskType=s["TaskType"], dataset=dsname, gt_answer=s.get("Answer",""),
                    _img_paths=[os.path.join(fdir, p) for p in imgs]))
    else:  # jsonl split (true_test / A / B); absolute image paths
        for line in open(base):
            line = line.strip()
            if not line: continue
            s = json.loads(line)
            if s.get("flagged_bug"): continue
            imgs = subsample(s["image"])
            out.append(dict(ImageName=s["image"], Question=s["question"],
                TaskType=s["task"], dataset=s["dataset"], gt_answer=s.get("answer",""),
                _img_paths=imgs))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default=os.environ.get("BASE_MODEL", "models/Qwen3.5-9B"),
                    help="local base model dir (InternVL3-8B-hf or InternVL3_5-8B-HF)")
    ap.add_argument("--adapter_path", required=True)
    ap.add_argument("--eval_set", required=True, choices=["valpub", "jsonl"])
    ap.add_argument("--data_path", required=True, help="dir (valpub) or split jsonl (jsonl)")
    ap.add_argument("--output_file", required=True)
    ap.add_argument("--exclude_task", action="append", default=["instance_detection"],
                    help="tasks to skip scoring/inference (instance_detection always dropped)")
    ap.add_argument("--only_task", default=None,
                    help="keep ONLY this task (e.g. counting) -- for evaluating a per-task specialist "
                         "adapter on its own task slice. Applied after --exclude_task.")
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--device", default="cuda:0")
    # CoT prompt variant -- MUST match the training export's --prompt_variant (byte-identity).
    # 'cot' is adopted for the counting adapter; keep 'none' (bare) elsewhere.
    ap.add_argument("--prompt_variant", default="none",
                    choices=["none", *list_variants()])
    ap.add_argument("--image_max_pixels", type=int, default=0)
    args = ap.parse_args()

    model, processor, fmt = load_model(args.base_model, args.adapter_path, args.device)

    variant = None
    if args.prompt_variant != "none":
        from prompt_variants import get_variant
        from format_templates import normalize_task_type
        variant = get_variant(args.prompt_variant)
        if variant.system_prompt:
            # the variant supplies a (non-empty) system prompt -> it must be emitted in the system
            # block even on Qwen3-VL (whose default omit_system=True would drop it), or train<->infer
            # diverge. Mirrors LF using the row's `system` field over the empty template default.
            fmt = dict(fmt); fmt["system"] = variant.system_prompt; fmt["omit_system"] = False
    samples = build_samples(args.eval_set, args.data_path)
    excl = set(t.strip().lower() for t in (args.exclude_task or []))
    samples = [s for s in samples if s["TaskType"].strip().lower() not in excl]
    if args.only_task:
        ot = args.only_task.strip().lower()
        samples = [s for s in samples if s["TaskType"].strip().lower() == ot]
        print(f"--only_task={ot!r}: kept {len(samples)} samples")
    print(f"Total samples (after excluding {sorted(excl)}): {len(samples)}")

    for s in tqdm(samples, desc=args.eval_set):
        try:
            imgs = []
            for p in s["_img_paths"]:
                try: imgs.append(_maybe_border_crop(_cap_pixels(Image.open(p).convert("RGB"), args.image_max_pixels)))
                except Exception as e: print(f"img fail {p}: {e}")
            if not imgs:
                s["Answer"] = "Error: No valid images"; continue
            image_tokens = "\n".join(f"Image-{i+1}: {fmt['image_placeholder']}" for i in range(len(imgs)))
            # human turn parts (must mirror export_sharegpt.render order): image block -> CoT
            # format instruction (if any) -> question. Bare path == image block + question (unchanged).
            parts = [image_tokens]
            answer_seed = ""
            if variant is not None:
                fi = variant.get_format_instruction(s.get("TaskType", ""), s["Question"])
                if fi:
                    parts.append(fi)
                answer_seed = variant.answer_seeds.get(normalize_task_type(s.get("TaskType", "")), "")
            parts.append(s["Question"])
            user_content = "\n".join(parts)
            prompt = build_chatml(fmt["system"], user_content, omit_system=fmt.get("omit_system", False)) + answer_seed
            inputs = processor(text=[prompt], images=imgs, padding=True, return_tensors="pt").to(model.device, dtype=torch.bfloat16)
            with torch.inference_mode():
                # do_sample=False: Qwen3-VL ships do_sample=true/temperature=0.7 in
                # generation_config.json (a chat default). Without this override every eval number is
                # measured under RANDOM decoding -- inflating variance run-to-run (same adapter, same
                # 100 rows -> 82 answers changed) and understating accuracy. Keep evals deterministic.
                gen = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
                trimmed = [o[len(i):] for i,o in zip(inputs.input_ids, gen)]
            output = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            if answer_seed:  # forced prefix is not in the generated tokens -> prepend before parsing
                output = answer_seed + output
            s["Answer"] = parse_answer(output, s.get("TaskType",""))
        except Exception as e:
            print(f"sample fail: {e}"); s["Answer"] = f"Error: {e}"

    for s in samples: s.pop("_img_paths", None)
    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    json.dump(samples, open(args.output_file,"w"), indent=2)
    print(f"Saved {len(samples)} -> {args.output_file}")

if __name__ == "__main__":
    main()
