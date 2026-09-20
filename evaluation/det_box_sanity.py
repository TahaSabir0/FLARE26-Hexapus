"""Box-format sanity for detection predictions (qwen35 sprint).
Reports: parse rate as nested [[x1,y1,x2,y2],...], malformed / flat / inverted /
out-of-bounds counts, and boxes-per-case pred vs GT, per dataset.
  python det_box_sanity.py --pred_file predictions_true_test.json
"""
import json, argparse, ast, collections
from PIL import Image

def parse_boxes(s):
    try:
        v = ast.literal_eval(str(s).strip())
    except Exception:
        return None, 'unparseable'
    if not isinstance(v, list): return None, 'not-a-list'
    if len(v) == 4 and all(isinstance(x, (int, float)) for x in v):
        return [v], 'flat'   # flat single box [x,y,x,y]
    if all(isinstance(b, list) and len(b) == 4 and all(isinstance(x, (int, float)) for x in b) for b in v) and v:
        return v, 'ok'
    return None, 'malformed-nesting'

ap = argparse.ArgumentParser(); ap.add_argument('--pred_file', required=True)
a = ap.parse_args()
data = [s for s in json.load(open(a.pred_file)) if s['TaskType'].strip().lower() == 'detection']
st = collections.defaultdict(collections.Counter)
bp = collections.defaultdict(lambda: [0, 0, 0])  # ds -> [pred boxes, gt boxes, n]
size_cache = {}
for s in data:
    ds = s['dataset']
    pb, tag = parse_boxes(s['Answer'])
    st[ds][tag] += 1
    gb, _ = parse_boxes(s['gt_answer'])
    if pb:
        img = s['ImageName'][0] if isinstance(s['ImageName'], list) else s['ImageName']
        if img not in size_cache:
            try: size_cache[img] = Image.open(img).size
            except Exception: size_cache[img] = None
        wh = size_cache[img]
        for b in pb:
            if b[2] <= b[0] or b[3] <= b[1]: st[ds]['inverted'] += 1
            if wh and (b[0] < 0 or b[1] < 0 or b[2] > wh[0] or b[3] > wh[1]): st[ds]['out-of-bounds'] += 1
        bp[ds][0] += len(pb)
    if gb: bp[ds][1] += len(gb)
    bp[ds][2] += 1
for ds in sorted(st):
    n = bp[ds][2]
    print(f'{ds}: n={n} tags={dict(st[ds])} boxes/case pred={bp[ds][0]/n:.2f} gt={bp[ds][1]/n:.2f}')
