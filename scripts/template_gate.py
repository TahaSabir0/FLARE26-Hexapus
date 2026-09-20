"""Template identity gate (Qwen3.5 sprint, gate 3).

Renders ONE exported deck sample through (a) LLaMA-Factory v0.9.5 qwen3_5 template
exactly as training will see it (mm_plugin.process_messages + ReasoningTemplate.
encode_oneturn, enable_thinking=False -> empty think block appended to PROMPT ids,
no loss), and (b) the inference path (processor.apply_chat_template,
add_generation_prompt=True, enable_thinking=False -- as in the Day-1 smoke/probe).
Byte-compares token ids and decoded strings (also with image_pad runs collapsed,
so a pad-count difference is reported separately from a template-text difference).
"""
import json
import os
import math
import re
import sys

from PIL import Image
from transformers import AutoProcessor

from llamafactory.data.template import TEMPLATES

MODEL = os.environ.get("QWEN35_MODEL", "models/Qwen3.5-9B")
DECK = sys.argv[1] if len(sys.argv) > 1 else (
    "training/"
    "data/flare_final_cls_rebal.jsonl")
IMAGE_MAX_PIXELS = 4194304  # 2048^2, the flagship knob

row = json.loads(open(DECK).readline())
human = row["conversations"][0]["value"]
gpt = row["conversations"][1]["value"]
img_path = row["images"][0]
print(f"sample: image={img_path}\nhuman[:120]={human[:120]!r}\ngpt={gpt!r}\n")

processor = AutoProcessor.from_pretrained(MODEL)
tokenizer = processor.tokenizer

# ---- (a) LF training render (SupervisedDatasetProcessor path) ----
template = TEMPLATES["qwen3_5"]
template.enable_thinking = False           # YAML enable_thinking: false
processor.image_max_pixels = IMAGE_MAX_PIXELS  # patcher.py:302 (model_args.image_max_pixels)
messages = [{"role": "user", "content": human},
            {"role": "assistant", "content": gpt}]
mm_messages = template.mm_plugin.process_messages(
    messages, [img_path], [], [], processor)
prompt_ids, response_ids = template.encode_oneturn(tokenizer, mm_messages, None, None)
lf_prompt = tokenizer.decode(prompt_ids)
lf_response = tokenizer.decode(response_ids)

# ---- (b) inference render (Day-1 probe convention) ----
content = []
parts = human.split("<image>")
for i, part in enumerate(parts):
    if part:
        content.append({"type": "text", "text": part})
    if i < len(parts) - 1:
        content.append({"type": "image"})
text = processor.apply_chat_template(
    [{"role": "user", "content": content}], tokenize=False,
    add_generation_prompt=True, enable_thinking=False)

def cap_pixels(im, mx):
    w, h = im.size
    if w * h > mx:
        f = math.sqrt(mx / (w * h))
        im = im.resize((max(1, int(w * f)), max(1, int(h * f))))
    return im

img = cap_pixels(Image.open(img_path).convert("RGB"), IMAGE_MAX_PIXELS)
enc = processor(text=[text], images=[img], return_tensors="pt")
inf_ids = enc.input_ids[0].tolist()
inf_prompt = tokenizer.decode(inf_ids)

# ---- compare ----
def collapse(s):
    return re.sub(r"(<\|image_pad\|>)+", "<|image_pad|>xN", s)

exact_ids = prompt_ids == inf_ids
exact_str = lf_prompt == inf_prompt
norm_str = collapse(lf_prompt) == collapse(inf_prompt)
lf_pads = lf_prompt.count("<|image_pad|>")
inf_pads = inf_prompt.count("<|image_pad|>")

print("=== VERDICT ===")
print(f"prompt token ids identical: {exact_ids}")
print(f"prompt strings identical:   {exact_str}")
print(f"identical after collapsing image_pad runs: {norm_str}")
print(f"image_pad counts: LF={lf_pads}  inference={inf_pads}")
print(f"think block in LF PROMPT (no loss): {'<think>' in lf_prompt}")
print(f"think block in LF RESPONSE (loss): {'<think>' in lf_response}")
print("\n=== LF prompt (pads collapsed) ===")
print(collapse(lf_prompt))
print("\n=== LF response (training target) ===")
print(repr(lf_response))
print("\n=== inference prompt (pads collapsed) ===")
print(collapse(inf_prompt))
print("\nGATE_" + ("PASS" if (exact_ids or (norm_str and lf_pads == inf_pads)) else "FAIL"))
