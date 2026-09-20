#!/usr/bin/env python3
"""Train<->infer byte-identity check (Phase-0 gate for any new backbone).

LLaMA-Factory prints the FIRST tokenized training example at train start
(src/llamafactory/data/processor/supervised.py: `inputs:\\n<decoded text>` with
skip_special_tokens=False). That decoded text IS the exact prompt+response the model trains on.

This script re-builds the SAME deck row's prompt the way our eval scripts do (vlm_prompt.backbone_format
+ build_chatml, with the literal "<image>" replaced by the backbone placeholder and expanded by the HF
processor), then compares the PROMPT PORTION to the training render parsed out of the LF log. If they
match, the eval is byte-identical to training and the eval numbers are trustworthy; if not, the eval is
silently wrong (the InternVL3.5 image-token rename / the Qwen vision-wrap + default-system are exactly
the things this catches).

  python check_prompt_identity.py --base_model ~/models/Qwen2.5-VL-7B-Instruct \
      --deck data/flare_cls_qwen_smoke.jsonl --train_log logs/iv35_smoke_<jobid>.out
"""
import argparse, json, os, re, sys

import torch
from PIL import Image
from transformers import AutoProcessor

from vlm_prompt import backbone_format, build_chatml
from infer_single_adapter import _cap_pixels  # reuse the EXACT infer-path pixel cap


def parse_train_inputs(log_path):
    """Extract the decoded training example (the block between 'inputs:' and 'label_ids:') from an LF log."""
    text = open(log_path, encoding="utf-8", errors="replace").read()
    # the supervised processor prints: input_ids:\n[...]\ninputs:\n<decoded>\nlabel_ids:\n[...]
    m = re.search(r"\ninputs:\n(.*?)\nlabel_ids:\n", text, re.DOTALL)
    if not m:
        return None
    return m.group(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--deck", required=True, help="the exported ShareGPT deck (row 0 is the LF example 0)")
    ap.add_argument("--train_log", required=True, help="the LF train .out that printed example 0")
    ap.add_argument("--markers", action="store_true", default=True)
    ap.add_argument("--image_max_pixels", type=int, default=0,
                    help="cap eval-side image pixels to match LF train render (0 = no cap; must equal RES the adapter trained at)")
    args = ap.parse_args()

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    processor = AutoProcessor.from_pretrained(args.base_model, local_files_only=True)
    tokenizer = getattr(processor, "tokenizer", processor)
    fmt = backbone_format(processor)
    print(f"backbone format: system={fmt['system']!r} image_placeholder={fmt['image_placeholder']!r}")

    # --- training render (ground truth), parsed from the LF log ---
    train_full = parse_train_inputs(args.train_log)
    if train_full is None:
        print("FAIL: could not find an 'inputs:' / 'label_ids:' block in the LF train log "
              "(did stage B actually train and print example 0?)")
        sys.exit(2)
    # cut at the assistant turn -> the prompt the model conditions on (response follows)
    marker = "<|im_start|>assistant\n"
    train_prompt = train_full.split(marker)[0] + marker if marker in train_full else train_full

    # --- eval render (our prompt), rebuilt from deck row 0 + expanded by the processor ---
    row0 = json.loads(open(args.deck, encoding="utf-8").readline())
    human = next(c["value"] for c in row0["conversations"] if c["from"] == "human")
    img_paths = list(row0.get("images", []))[:1]
    # mirror LF's mm_plugin: replace the literal "<image>" the deck carries with the backbone placeholder
    human_eval = human.replace("<image>", fmt["image_placeholder"])
    prompt = build_chatml(fmt["system"], human_eval, omit_system=fmt.get("omit_system", False))
    imgs = [_cap_pixels(Image.open(p).convert("RGB"), args.image_max_pixels) for p in img_paths]
    inputs = processor(text=[prompt], images=imgs or None, return_tensors="pt")
    eval_prompt = tokenizer.decode(inputs.input_ids[0], skip_special_tokens=False)

    # --- compare (collapse the long image-pad runs so a human-readable diff is short) ---
    def collapse(s):
        for pad in ("<|image_pad|>", "<IMG_CONTEXT>"):
            s = re.sub(re.escape(pad) + r"(\s*" + re.escape(pad) + r")+", pad + "...(xN)..." + pad, s)
        return s

    match = train_prompt == eval_prompt
    print("=" * 70)
    print(f"BYTE-IDENTITY: {'MATCH' if match else 'MISMATCH'}")
    print(f"  n_eval_input_tokens = {inputs.input_ids.shape[1]}")
    if match:
        print("  -> eval prompt is byte-identical to the LF training render. Eval is trustworthy.")
    else:
        print("--- TRAIN prompt (collapsed) ---")
        print(collapse(train_prompt))
        print("--- EVAL prompt (collapsed) ---")
        print(collapse(eval_prompt))
        # first divergence
        for i, (a, b) in enumerate(zip(train_prompt, eval_prompt)):
            if a != b:
                lo = max(0, i - 40)
                print(f"--- first divergence at char {i} ---")
                print(f"  train: ...{train_prompt[lo:i+40]!r}")
                print(f"  eval : ...{eval_prompt[lo:i+40]!r}")
                break
        else:
            print(f"  (one is a prefix of the other; lengths train={len(train_prompt)} eval={len(eval_prompt)})")
    sys.exit(0 if match else 1)


if __name__ == "__main__":
    main()
