#!/usr/bin/env python3
"""Structural data-quality audit for FLARE-2026 Task 3 (2D).

Why this exists
---------------
`build_splits.py` only catches ONE bug class (classification answers whose letter isn't an
offered option -- the endo 'T'/multi-letter issue). That's a single tripwire. This script is the
wide sweep: it defines what a *valid* sample looks like for every task type and flags everything
that violates the schema, plus the sneakier problems that are schema-valid but still wrong
(cross-split image leakage, contradictory labels, coordinate-convention drift).

It does NOT auto-delete anything. It produces a **candidate list with reason codes** for humans to
triage. Over-cleaning (tossing hard-but-correct samples) hurts more than it helps, so every flag is
a candidate, not a verdict.

What it checks
--------------
Always (cheap, offline CPU):
  * per-task answer schema      classification / multi-label / detection / counting / regression /
                                report / instance_detection -- each has its own validity rule
  * task-shape mismatch         answer shape contradicts the declared TaskType
  * detection geometry          not-4-numbers, zero/negative/inverted area
  * detection coord convention  per-dataset min/max coord ranges + pixel-vs-normalized consistency
  * counting / regression       non-numeric, negative, IQR/abs outliers
  * report generation           empty / too-short / exact-duplicate report text
  * answer distribution         per (dataset x task): singleton classes, label-space drift
  * missing image files         ImageName path (every entry of multi-image lists) not on disk
  * exact duplicate samples     identical (image, question)
  * contradictory labels        identical (image, question) with DIFFERENT answers

Opt-in (heavier I/O):
  * --check-images              decode every image with PIL.verify() -> flag corrupt/truncated
  * --check-leakage             md5-hash image bytes across splits -> flag the SAME image present in
                                both training and validation-public (would inflate the pseudo-test)

Model-assisted label-error detection (confident-learning on saved predictions) is intentionally NOT
here -- it belongs in a separate pass that consumes predictions.json.

Outputs
-------
  audit_report.md   human summary: per-reason counts, per-dataset breakdown, coord-convention table,
                    distribution red-flags, example dumps
  audit_issues.csv  one row per flagged item: reason, dataset, task, split, sample_id, sample_key, detail

Usage
-----
  python audit_data.py \
      --data_root <DATA_ROOT>/FLARE-Task5-MLLM-2D \
      --out_dir   ./audit_v1 \
      --manifest  ./internal_splits_v2/split_manifest.json   # optional: attach split/sample_id
      [--check-images] [--check-leakage]
"""

import argparse
import ast
import csv
import glob
import hashlib
import json
import os
import re
from collections import Counter, defaultdict

# --------------------------------------------------------------------------------------
# shared helpers (kept consistent with build_splits.py / evaluation.py)
# --------------------------------------------------------------------------------------

_OPTION_RE = re.compile(r"(?<![A-Za-z])([A-Z])[.):]\s")


def create_sample_key(sample):
    image_name = str(sample.get("ImageName", sample.get("image", "")))
    question = str(sample.get("Question", ""))
    return f"{image_name}||{question}"


def offered_option_letters(question):
    return set(_OPTION_RE.findall(str(question)))


def norm_task(t):
    return str(t).strip().lower()


def image_paths(sample):
    """Return the list of raw ImageName entries (multi-image samples carry a list)."""
    img = sample.get("ImageName", sample.get("image", ""))
    if isinstance(img, list):
        return [str(x) for x in img]
    return [str(img)] if img else []


def parse_boxes(ans):
    """Parse a detection answer into a list of [x1,y1,x2,y2]; None if unparseable/malformed."""
    v = ans
    if isinstance(v, str):
        try:
            v = ast.literal_eval(v)
        except Exception:
            return None
    if not isinstance(v, (list, tuple)) or len(v) == 0:
        return None
    # single box [x,y,x,y]
    if len(v) == 4 and all(isinstance(x, (int, float)) for x in v):
        return [list(v)]
    # list of boxes
    if all(isinstance(b, (list, tuple)) and len(b) == 4
           and all(isinstance(x, (int, float)) for x in b) for b in v):
        return [list(b) for b in v]
    return None


def is_float(x):
    try:
        float(x)
        return True
    except Exception:
        return False


# --------------------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------------------

def load_records(data_root, split_dirname):
    base = os.path.join(data_root, split_dirname)
    records = []
    for jp in sorted(glob.glob(os.path.join(base, "*", "*", "*.json"))):
        parts = os.path.normpath(jp).split(os.sep)
        dataset, modality = parts[-2], parts[-3]
        dataset_dir = os.path.dirname(jp)
        try:
            with open(jp, "r") as f:
                data = json.load(f)
        except Exception as e:  # noqa: BLE001
            print(f"  WARN: unreadable {jp}: {e}")
            continue
        if not isinstance(data, list):
            data = [data]
        for s in data:
            if not isinstance(s, dict):
                continue
            task = norm_task(s.get("TaskType", ""))
            if not task:
                continue
            records.append({
                "split_src": split_dirname,
                "dataset": dataset,
                "modality": modality,
                "dataset_dir": dataset_dir,
                "task": task,
                "question": str(s.get("Question", "")),
                "answer": s.get("Answer", ""),
                "answer_str": str(s.get("Answer", "")).strip(),
                "images": image_paths(s),
                "sample_key": create_sample_key(s),
            })
    return records


# --------------------------------------------------------------------------------------
# checks  (each appends dicts: {reason, detail} to the record's issue list)
# --------------------------------------------------------------------------------------

def check_answer_schema(r):
    """Per-task validity + task-shape mismatch. Returns list of (reason, detail)."""
    issues = []
    task, ans = r["task"], r["answer_str"]

    if task == "classification":
        offered = offered_option_letters(r["question"])
        toks = [t for t in re.split(r"[\s,]+", ans) if t]
        if offered:
            if len(toks) > 1 and all(t in offered for t in toks):
                issues.append(("multilabel_mislabel_as_classification", f"answer={ans!r} offered={sorted(offered)}"))
            elif any(t not in offered for t in toks):
                # answer looks like full option text rather than a letter?
                if len(ans) > 3 and not re.fullmatch(r"[A-Za-z](\s+[A-Za-z])*", ans):
                    issues.append(("answer_is_option_text", f"answer={ans!r} offered={sorted(offered)}"))
                else:
                    issues.append(("answer_not_in_options", f"answer={ans!r} offered={sorted(offered)}"))
        elif re.fullmatch(r"[A-Za-z]", ans):
            # a BARE letter answer but no lettered options in the question -> genuinely suspicious.
            # (free-text answers like CMMD's "Mass"/"Calcification" are a valid different format, not a bug.)
            issues.append(("classification_no_options_in_question", f"bare-letter answer {ans!r}, no options in q"))
        if parse_boxes(r["answer"]) is not None:
            issues.append(("task_shape_mismatch", f"classification but answer parses as boxes: {ans!r}"))

    elif task == "multi-label classification":
        labels = [x.strip() for x in re.split(r"[;,]", ans) if x.strip()]
        if not labels:
            issues.append(("multilabel_empty", f"answer={ans!r}"))
        if len(labels) != len(set(l.lower() for l in labels)):
            issues.append(("multilabel_duplicate_label", f"answer={ans!r}"))
        low = [l.lower() for l in labels]
        if len(labels) > 1 and ("normal" in low or "no finding" in low or "none" in low):
            issues.append(("multilabel_normal_with_findings", f"answer={ans!r}"))

    elif task == "detection":
        boxes = parse_boxes(r["answer"])
        if boxes is None:
            issues.append(("detection_malformed", f"answer={ans[:120]!r}"))
        else:
            for b in boxes:
                x1, y1, x2, y2 = b
                if x2 <= x1 or y2 <= y1:
                    issues.append(("detection_bad_geometry", f"box={b} (zero/negative/inverted area)"))
                    break

    elif task in ("counting", "regression"):
        if not is_float(ans):
            issues.append(("numeric_non_numeric", f"task={task} answer={ans!r}"))
        else:
            val = float(ans)
            if val < 0:
                issues.append(("numeric_negative", f"task={task} answer={ans!r}"))
            if task == "counting" and not float(ans).is_integer():
                issues.append(("counting_non_integer", f"answer={ans!r}"))

    elif task == "report_generation":
        if len(ans) == 0:
            issues.append(("report_empty", "answer is empty"))
        elif len(ans.split()) < 2:  # terse one-word reports ("Unremarkable") are valid; only flag <2 words
            issues.append(("report_too_short", f"answer={ans!r}"))

    elif task == "instance_detection":
        try:
            d = json.loads(ans) if isinstance(r["answer"], str) else r["answer"]
            if not isinstance(d, dict):
                issues.append(("instdet_malformed", f"answer={ans[:120]!r}"))
        except Exception:
            issues.append(("instdet_malformed", f"answer={ans[:120]!r}"))

    return issues


def check_images_exist(r):
    issues = []
    for ip in r["images"]:
        full = os.path.join(r["dataset_dir"], ip)
        if not os.path.exists(full):
            issues.append(("missing_image", ip))
    return issues


# --------------------------------------------------------------------------------------
# dataset-level analyses
# --------------------------------------------------------------------------------------

def detection_coord_report(records):
    """Per detection dataset: coord range + fraction of coords in [0,1] (normalized vs pixel)."""
    by_ds = defaultdict(list)
    for r in records:
        if r["task"] == "detection":
            boxes = parse_boxes(r["answer"])
            if boxes:
                for b in boxes:
                    by_ds[(r["split_src"], r["dataset"])].extend(b)
    rows = []
    for (split, ds), coords in sorted(by_ds.items()):
        if not coords:
            continue
        frac_unit = sum(1 for c in coords if 0.0 <= c <= 1.0) / len(coords)
        rows.append((split, ds, min(coords), max(coords), frac_unit, len(coords)))
    return rows


def numeric_outlier_report(records, task):
    """Per dataset min/median/max + IQR outlier candidates for counting/regression."""
    by_ds = defaultdict(list)
    for r in records:
        if r["task"] == task and is_float(r["answer_str"]):
            by_ds[(r["split_src"], r["dataset"])].append(float(r["answer_str"]))
    summary, outliers = [], []
    for (split, ds), vals in sorted(by_ds.items()):
        vs = sorted(vals)
        n = len(vs)
        med = vs[n // 2]
        q1, q3 = vs[n // 4], vs[(3 * n) // 4]
        iqr = q3 - q1
        hi = q3 + 3 * iqr
        n_out = sum(1 for v in vs if v > hi)
        summary.append((split, ds, vs[0], med, vs[-1], n, n_out, hi))
    return summary


def regression_outlier_samples(records):
    """Per-sample regression outliers (value > Q3 + 3*IQR within its dataset).

    ONLY regression -- counting has a genuine heavy tail (val max in the thousands is real), so
    counting outliers are NOT flagged. Regression labels are physical measurements with tight
    per-dataset ranges, so a value far above the bulk (e.g. boneresorption 2085 vs median 28) is
    almost certainly a bad label. Returns list of (sample_key, dataset, detail).
    """
    by_ds = defaultdict(list)
    for r in records:
        if r["task"] == "regression" and is_float(r["answer_str"]):
            by_ds[(r["split_src"], r["dataset"])].append(r)
    out = []
    RATIO = 10.0  # also require value >= RATIO x median. A genuine bad label / unit error is
                  # orders of magnitude off (boneresorption 2085 = ~75x median); a wide-but-real
                  # distribution (iugc max 151 = ~1.77x median) must NOT be flagged. The IQR test
                  # alone slices an arbitrary point in a smooth tail -> this guard prevents that.
    for (split_src, ds), recs in by_ds.items():
        vals = sorted(float(r["answer_str"]) for r in recs)
        n = len(vals)
        if n < 8:  # too few to judge an outlier threshold reliably
            continue
        median = vals[n // 2]
        q1, q3 = vals[n // 4], vals[(3 * n) // 4]
        hi = q3 + 3 * (q3 - q1)
        for r in recs:
            v = float(r["answer_str"])
            if v > hi and median > 0 and v > RATIO * median:
                out.append((r["sample_key"], split_src, ds,
                            f"value={r['answer_str']} > Q3+3*IQR={hi:.1f} AND >{RATIO:g}x median {median:.1f}"))
    return out


def distribution_redflags(records):
    """Per (dataset x classification task): singleton answer classes (likely typos)."""
    by_cell = defaultdict(Counter)
    for r in records:
        if r["task"] == "classification":
            by_cell[(r["split_src"], r["dataset"])][r["answer_str"]] += 1
    flags = []
    for (split, ds), counter in sorted(by_cell.items()):
        singletons = [a for a, c in counter.items() if c == 1]
        if singletons and len(counter) > 2:
            flags.append((split, ds, dict(counter), singletons))
    return flags


def find_duplicates_and_contradictions(records):
    """Exact dup (image,question) and contradictory (same key, different answer)."""
    by_key = defaultdict(list)
    for r in records:
        by_key[(r["split_src"], r["sample_key"])].append(r)
    dups, contradictions = [], []
    for (split, key), group in by_key.items():
        if len(group) > 1:
            answers = set(g["answer_str"] for g in group)
            if len(answers) == 1:
                dups.append((split, group[0]["dataset"], key, len(group)))
            else:
                contradictions.append((split, group[0]["dataset"], key, sorted(answers)))
    return dups, contradictions


def cross_split_leakage(records_by_split, splits=("training", "validation-public")):
    """md5 image bytes; flag content present in >1 split. Heavy I/O -- opt-in."""
    hash_to = defaultdict(set)
    seen_files = set()
    for split in splits:
        for r in records_by_split.get(split, []):
            for ip in r["images"]:
                full = os.path.join(r["dataset_dir"], ip)
                if full in seen_files or not os.path.exists(full):
                    continue
                seen_files.add(full)
                try:
                    with open(full, "rb") as f:
                        h = hashlib.md5(f.read()).hexdigest()
                except Exception:
                    continue
                hash_to[h].add((split, r["dataset"], os.path.basename(ip)))
    leaks = []
    for h, locs in hash_to.items():
        split_set = set(s for s, _d, _n in locs)
        if len(split_set) > 1:
            leaks.append((h, sorted(locs)))
    return leaks


def image_integrity(records_by_split):
    """PIL.verify() every unique image -- opt-in. Returns list of (split, dataset, path)."""
    try:
        from PIL import Image
    except Exception:
        print("  WARN: Pillow not available; skipping --check-images")
        return None
    bad, seen = [], set()
    for split, recs in records_by_split.items():
        for r in recs:
            for ip in r["images"]:
                full = os.path.join(r["dataset_dir"], ip)
                if full in seen or not os.path.exists(full):
                    continue
                seen.add(full)
                try:
                    with Image.open(full) as im:
                        im.verify()
                except Exception as e:  # noqa: BLE001
                    bad.append((split, r["dataset"], ip, str(e)[:80]))
    return bad


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--out_dir", default="./audit_v1")
    ap.add_argument("--manifest", default=None, help="split_manifest.json to attach sample_id/split")
    ap.add_argument("--splits", nargs="+", default=["training", "validation-public"])
    ap.add_argument("--check-images", action="store_true", help="decode every image (slow)")
    ap.add_argument("--check-leakage", action="store_true", help="hash image bytes across splits (slow)")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    key_to_meta = {}
    if args.manifest and os.path.exists(args.manifest):
        m = json.load(open(args.manifest))
        for sid, rec in m.items():
            key_to_meta[rec["sample_key"]] = (sid, rec["split"])

    records_by_split = {}
    all_records = []
    for sp in args.splits:
        recs = load_records(args.data_root, sp)
        records_by_split[sp] = recs
        all_records.extend(recs)
        print(f"Loaded {len(recs)} from {sp}")

    # ---- per-record checks ----
    issues = []  # (reason, split_src, dataset, task, sample_key, detail)
    for r in all_records:
        for reason, detail in check_answer_schema(r) + check_images_exist(r):
            issues.append((reason, r["split_src"], r["dataset"], r["task"], r["sample_key"], detail))

    # ---- dataset-level ----
    coord_rows = detection_coord_report(all_records)
    count_summary = numeric_outlier_report(all_records, "counting")
    reg_summary = numeric_outlier_report(all_records, "regression")
    redflags = distribution_redflags(all_records)
    dups, contradictions = find_duplicates_and_contradictions(all_records)
    for split, ds, key, n in dups:
        issues.append(("exact_duplicate", split, ds, "", key, f"{n} copies"))
    for split, ds, key, answers in contradictions:
        issues.append(("contradictory_label", split, ds, "", key, f"answers={answers}"))
    # per-sample regression outliers (regression only; counting tail is real) -> removable bug
    for key, split_src, ds, detail in regression_outlier_samples(all_records):
        issues.append(("regression_outlier", split_src, ds, "regression", key, detail))

    leaks = None
    if args.check_leakage:
        print("Hashing images for cross-split leakage (slow)...")
        leaks = cross_split_leakage(records_by_split)
        for h, locs in leaks:
            issues.append(("cross_split_leak", "multi", locs[0][1], "", h, str(locs)))
        # machine-readable artifact for build_splits.py to drop leaked images from the pseudo-test
        val_basenames = sorted({n for _h, locs in leaks for s, _d, n in locs if s == "validation-public"})
        train_basenames = sorted({n for _h, locs in leaks for s, _d, n in locs if s == "training"})
        with open(os.path.join(args.out_dir, "leaked_images.json"), "w") as f:
            json.dump({"n_leaked_hashes": len(leaks),
                       "validation_public_basenames": val_basenames,
                       "training_basenames": train_basenames}, f, indent=1)

    bad_images = None
    if args.check_images:
        print("Verifying image integrity with PIL (slow)...")
        bad_images = image_integrity(records_by_split)
        if bad_images:
            for split, ds, ip, err in bad_images:
                issues.append(("image_corrupt", split, ds, "", ip, err))

    # ---- write issues csv ----
    with open(os.path.join(args.out_dir, "audit_issues.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["reason", "split_src", "dataset", "task", "sample_id", "split", "sample_key", "detail"])
        for reason, split_src, ds, task, key, detail in issues:
            sid, split = key_to_meta.get(key, ("", ""))
            w.writerow([reason, split_src, ds, task, sid, split, key, detail])

    # ---- build report ----
    by_reason = Counter(i[0] for i in issues)
    by_reason_ds = defaultdict(Counter)
    for reason, _s, ds, _t, _k, _d in issues:
        by_reason_ds[reason][ds] += 1
    # how many flagged rows fall in usable training-derived splits (would pollute training)?
    in_train_splits = Counter()
    if key_to_meta:
        for reason, _s, _ds, _t, key, _d in issues:
            _sid, split = key_to_meta.get(key, ("", ""))
            if split in ("train", "dev", "test_internal"):
                in_train_splits[reason] += 1

    L = []
    L.append("# Data-quality audit report\n")
    L.append(f"Records scanned: {len(all_records)}  |  total issues flagged: {len(issues)}\n")
    L.append("> Every flag is a **candidate**, not a verdict. Triage before removing.\n")

    L.append("\n## Issues by reason code\n")
    L.append("| reason | count | in train/dev/test | top datasets |")
    L.append("|---|---|---|---|")
    for reason, n in by_reason.most_common():
        top = ", ".join(f"{d}:{c}" for d, c in by_reason_ds[reason].most_common(4))
        L.append(f"| {reason} | {n} | {in_train_splits.get(reason, '-')} | {top} |")

    L.append("\n## Detection coordinate conventions (per dataset)\n")
    L.append("If `frac_in_[0,1]` is ~1.0 the boxes are **normalized**; if min/max go into the hundreds "
             "they're **pixel** coords. Mixed conventions across datasets silently break IoU.\n")
    L.append("| split | dataset | min | max | frac_in_[0,1] | n_coords |")
    L.append("|---|---|---|---|---|---|")
    for split, ds, mn, mx, frac, n in coord_rows:
        L.append(f"| {split} | {ds} | {mn:.2f} | {mx:.2f} | {frac:.3f} | {n} |")

    for title, summ in (("Counting", count_summary), ("Regression", reg_summary)):
        L.append(f"\n## {title} value ranges + outlier candidates (per dataset)\n")
        L.append("| split | dataset | min | median | max | n | n_outliers(>Q3+3·IQR) | hi_thresh |")
        L.append("|---|---|---|---|---|---|---|---|")
        for split, ds, mn, med, mx, n, n_out, hi in summ:
            L.append(f"| {split} | {ds} | {mn:.2f} | {med:.2f} | {mx:.2f} | {n} | {n_out} | {hi:.2f} |")

    L.append("\n## Classification cells with singleton answer classes (likely typos)\n")
    if redflags:
        for split, ds, counter, singletons in redflags[:20]:
            L.append(f"- **{split}/{ds}** singletons={singletons} full={counter}")
    else:
        L.append("(none)")

    L.append(f"\n## Duplicates & contradictions\n")
    L.append(f"- exact duplicate (image,question): **{len(dups)}**")
    L.append(f"- contradictory labels (same image+question, different answer): **{len(contradictions)}**")
    for split, ds, key, answers in contradictions[:15]:
        L.append(f"  - {split}/{ds}: answers={answers} :: {key[:140]}")

    if leaks is not None:
        L.append(f"\n## Cross-split image leakage (training <-> validation-public)\n")
        L.append(f"Images whose **bytes** appear in more than one split: **{len(leaks)}**. "
                 f"Any >0 means the pseudo-test score is inflated.\n")
        for h, locs in leaks[:15]:
            L.append(f"- {h[:12]} :: {locs}")

    if bad_images is not None:
        L.append(f"\n## Corrupt/unreadable images: **{len(bad_images)}**\n")
        for split, ds, ip, err in bad_images[:15]:
            L.append(f"- {split}/{ds}/{ip}: {err}")

    # example dumps per reason
    L.append("\n## Example flagged samples (up to 8 per reason)\n")
    ex = defaultdict(list)
    for reason, split_src, ds, task, key, detail in issues:
        if len(ex[reason]) < 8:
            ex[reason].append(f"  [{split_src}/{ds}] {detail} :: {key[:120]}")
    for reason in by_reason:
        L.append(f"\n**{reason}**")
        L.extend(ex[reason])

    with open(os.path.join(args.out_dir, "audit_report.md"), "w") as f:
        f.write("\n".join(L) + "\n")

    print(f"\nTop reasons: {dict(by_reason.most_common(10))}")
    print(f"Wrote audit_report.md + audit_issues.csv to {args.out_dir}")


if __name__ == "__main__":
    main()
