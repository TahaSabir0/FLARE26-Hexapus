"""Qwen3.5-9B single-adapter true_test eval (v5 stack) — matched to the 768^2 ws training arms.

Renders eval prompts EXACTLY as the training deck export does (Image-i markers -> variant format
instruction -> question), then routes through processor.apply_chat_template(enable_thinking=False,
add_generation_prompt=True) — byte-identity to the LF qwen3_5 training render is gate-verified
(template_gate.py at 2048^2 cls; /tmp one-offs 2026-08-31 at 768^2 for the ML deck WITH system
column and the regmix bare deck: GATE_PASS, ids identical, pads equal).

Matched-to-training knobs: 768^2 cap (image_max_pixels=589824, gate cap_pixels int-rounding),
4-bit nf4 double_quant=False (training YAML double_quantization: false), greedy do_sample=False,
enable_thinking=False, MAX_IMAGES=4 + Image-i: markers.
Parsing: strip_think_block FIRST, then parse_answer (docker-path convention — the 08-20 rule).

Usage :
  CUDA_VISIBLE_DEVICES=<n> python eval_qwen35_arm.py --adapter_path ... --data_path true_test.jsonl       --only_task 'multi-label classification' --prompt_variant medical_cot --output_file preds.json
"""
import os, sys, json, time, math, argparse
from pathlib import Path
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig

_SCRIPTS = Path(__file__).resolve().parents[2] / 'llama_factory_internvl' / 'scripts'
sys.path.insert(0, str(_SCRIPTS))
from vlm_prompt import strip_think_block
from infer_single_adapter import parse_answer, subsample
from prompt_variants import get_variant, list_variants
from format_templates import normalize_task_type

def cap_pixels(im, mx):
    # gate-verified resize (template_gate.py convention: int() rounding, matches LF pad counts)
    w, h = im.size
    if mx and w * h > mx:
        f = math.sqrt(mx / (w * h))
        im = im.resize((max(1, int(w * f)), max(1, int(h * f))))
    return im

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base_model', default=os.environ.get('BASE_MODEL', 'models/Qwen3.5-9B'))
    ap.add_argument('--adapter_path', required=True)
    ap.add_argument('--data_path', required=True, help='split jsonl (true_test.jsonl)')
    ap.add_argument('--only_task', required=True)
    ap.add_argument('--prompt_variant', default='none', choices=['none', *list_variants()])
    ap.add_argument('--image_max_pixels', type=int, default=589824)
    ap.add_argument('--max_new_tokens', type=int, default=512)
    ap.add_argument('--limit', type=int, default=0, help='smoke: first N rows only')
    ap.add_argument('--output_file', required=True)
    args = ap.parse_args()

    for v in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_HUB_DISABLE_TELEMETRY'):
        os.environ.setdefault(v, '1')
    from peft import PeftModel
    proc = AutoProcessor.from_pretrained(args.base_model, local_files_only=True)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                             bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_use_double_quant=False)  # training: double_quantization false
    model = AutoModelForImageTextToText.from_pretrained(
        args.base_model, quantization_config=bnb, dtype=torch.bfloat16,
        device_map='cuda:0', local_files_only=True)
    model = PeftModel.from_pretrained(model, args.adapter_path, is_trainable=False)
    model.eval()
    # Qwen3.5-9B ships NO generation_config.json -> generate() does not stop at <|im_end|>
    # (smoke 2026-08-31: correct first-line answer, then a babbled fake 'user' turn to the token
    # budget). Pin the stop token explicitly (pin-decode-params doctrine); training targets end
    # with <|im_end|> (LF qwen3_5), so this matches training.
    tok = proc.tokenizer
    im_end = tok.convert_tokens_to_ids('<|im_end|>')
    eos_ids = sorted({im_end, tok.eos_token_id} - {None})
    print(f'loaded base + adapter {args.adapter_path}; eos_token_id={eos_ids}')

    variant = None if args.prompt_variant == 'none' else get_variant(args.prompt_variant)

    ot = args.only_task.strip().lower()
    rows = []
    for line in open(args.data_path):
        line = line.strip()
        if not line: continue
        s = json.loads(line)
        if s.get('flagged_bug'): continue
        if s['task'].strip().lower() != ot: continue
        rows.append(s)
    if args.limit: rows = rows[:args.limit]
    print(f'rows for task {ot!r}: {len(rows)}')

    out, gen_secs = [], []
    for s in tqdm(rows):
        try:
            paths = subsample(s['image'])
            imgs = [cap_pixels(Image.open(p).convert('RGB'), args.image_max_pixels) for p in paths]
            parts = ['\n'.join(f'Image-{i+1}: <image>' for i in range(len(imgs)))]
            seed = ''
            if variant is not None:
                fi = variant.get_format_instruction(s['task'], s['question'])
                if fi: parts.append(fi)
                seed = variant.answer_seeds.get(normalize_task_type(s['task']), '')
            parts.append(s['question'])
            human = '\n'.join(parts)
            content = []
            segs = human.split('<image>')
            for i, seg in enumerate(segs):
                if seg: content.append({'type': 'text', 'text': seg})
                if i < len(segs) - 1: content.append({'type': 'image'})
            messages = []
            if variant is not None and variant.system_prompt:
                messages.append({'role': 'system', 'content': variant.system_prompt})
            messages.append({'role': 'user', 'content': content})
            text = proc.apply_chat_template(messages, tokenize=False,
                                            add_generation_prompt=True, enable_thinking=False)
            if seed: text += seed
            inputs = proc(text=[text], images=imgs, return_tensors='pt').to('cuda:0')
            t0 = time.time()
            with torch.inference_mode():
                gen = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False, eos_token_id=eos_ids, pad_token_id=tok.pad_token_id or im_end)
            dt = time.time() - t0
            raw = proc.batch_decode(gen[:, inputs['input_ids'].shape[1]:],
                                    skip_special_tokens=True)[0]
            full = seed + raw
            stripped = strip_think_block(full)   # strip FIRST, then parse (08-20 rule)
            ans = parse_answer(stripped, s['task'])
            gen_secs.append(dt)
            out.append(dict(ImageName=s['image'], Question=s['question'], TaskType=s['task'],
                            dataset=s['dataset'], gt_answer=s.get('answer', ''),
                            raw_output=full, Answer=ans, gen_seconds=round(dt, 3)))
        except Exception as e:
            print(f'sample fail: {e}')
            out.append(dict(ImageName=s.get('image'), Question=s.get('question'),
                            TaskType=s.get('task'), dataset=s.get('dataset'),
                            gt_answer=s.get('answer', ''), raw_output='',
                            Answer=f'Error: {e}', gen_seconds=None))
    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    json.dump(out, open(args.output_file, 'w'), indent=2)
    if gen_secs:
        print(f'gen s/case mean={sum(gen_secs)/len(gen_secs):.2f} n={len(gen_secs)}')
    print(f'saved {len(out)} -> {args.output_file}')

if __name__ == '__main__':
    main()
