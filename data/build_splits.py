#!/usr/bin/env python3
"""Build the internal cross-validation splits + data-cleaning manifest for FLARE-2026 Task 3 (2D).

v2 -- see FLARE2026_data_pipeline_v2.md
-------------------------------------
Why this exists
---------------
Four interns explore different paths. For "explore then compare" to be valid, every experiment must
be measured on the *same frozen splits* with the *same cleaning*. This script is the single source of
truth for those splits, so nobody silently trains/evaluates on a different cut.

What changed from v1
--------------------
* **50/50 cross-validation geometry** (Prof. Yaqub, 2026-06-10): instead of one train/dev/test cut,
  carve a 10% untouched `true_test`, then two symmetric halves `A` (45%) and `B` (45%). Train on A,
  test on B, then flip; consistent results across folds => the method generalizes. Each half holds a
  small internal `dev` slice for checkpoint selection (so the opposite half stays a clean test), and a
  30% `prime` subset (`A'`, `B'`) for fast iteration -- interchangeable as quick train/test.
* **Lossless / separation-of-concerns:** the split files store EVERY image path, capped at NOTHING.
  Image packing (the multi-image chunk size K) is applied downstream in export_sharegpt.py, NOT here;
  tile caps are a runtime concern. The dataset layer is pure content. (v1 truncated images to 6 in the
  data files, destroying paths on disk -- that bug is gone.)

What it produces (all deterministic, fixed seed)
------------------------------------------------
From the team-shared 2D dataset (`<split>/<Modality>/<dataset>/*.json`), self-contained per-split
JSONL -- each line carries its content (question + ALL ordered absolute image paths + answer) inline,
plus metadata (dataset/modality/task), cleaning flags, and `sample_key` as the only identity.

  TRAIN-distribution (carved from `training/`, IMAGE-LEVEL stratified by dataset x task-combo):
    - true_test.jsonl        10%  -- untouched objective in-distribution test; never trained on
    - A.jsonl                45%  -- half A (full; eval target when training on B)
    - A_train.jsonl          90% of A -- fit weights for the A-fold
    - A_dev.jsonl            10% of A -- checkpoint selection for the A-fold
    - A_prime.jsonl          30% of A -- fast-iteration quick set (interchangeable)
    - B.jsonl / B_train / B_dev / B_prime      symmetric to A
    - flagged_removed.jsonl  cleaning quarantine (never trained)

  validation-public-derived (the untouched competition-distribution pseudo-test):
    - validation-public.jsonl        leaked excluded; raw -- mirrors the real eval, bugs included
    - validation-public-clean.jsonl  leaked + bug-flagged removed (the achievable ceiling)
    - validation-public-leaked.jsonl the leaked images, for the record (NOT scored)

  Bookkeeping:
    - split_manifest.csv     flat one-row-per-sample audit mirror (scalar fields only)
    - coverage_report.md     per (split x dataset x task) counts + the val-public coverage gap

Design notes
------------
* sample_key EXACTLY matches evaluation.py's create_sample_key (`str(ImageName)||str(Question)`) so
  these files join 1:1 against any predictions.json. It is the ONLY identity (content is inline).
* Splitting unit = the IMAGE (group), not the row. iugc reuses images across cls/det/reg and some
  samples are multi-image -> row-level splitting would leak. Grouping uses the SORTED image_key; the
  materialized `image` list preserves ORIGINAL order. A and B never share an image.
* Modality and dataset come from the FOLDER PATH (the `Modality` field is unreliable). TaskType is
  normalized with .lower().strip().

Usage
-----
  python build_splits.py \
      --data_root <DATA_ROOT>/FLARE-Task5-MLLM-2D \
      --out_dir   ./internal_splits_v2 \
      --seed 1337 --frac_true_test 0.10 --frac_A 0.45 --frac_dev 0.10 --frac_prime 0.30 \
      --leaked_images .../leaked_images.json --audit_issues .../audit_issues.csv
"""

import argparse
import csv
import glob
import json
import os
import random
import re
from collections import Counter, defaultdict

# --------------------------------------------------------------------------------------
# sample identity -- keep byte-for-byte compatible with evaluation.py::create_sample_key
# --------------------------------------------------------------------------------------

def create_sample_key(sample):
    image_name = str(sample.get("ImageName", sample.get("image", "")))
    question = str(sample.get("Question", ""))
    return f"{image_name}||{question}"


def image_group_key(sample):
    """Canonical key for the IMAGE(s) a sample uses, so all questions on the same image(s) land in the
    same split. List-valued ImageName (multi-image) is order-normalized."""
    img = sample.get("ImageName", sample.get("image", ""))
    if isinstance(img, list):
        return "|".join(sorted(str(x) for x in img))
    return str(img)


def ordered_image_list(sample):
    """ImageName entries in ORIGINAL order (order matters for multi-view samples, e.g. frontal then
    lateral X-ray). Unlike image_group_key this is not sorted."""
    img = sample.get("ImageName", sample.get("image", ""))
    if isinstance(img, list):
        return [str(x) for x in img]
    return [str(img)] if img else []


# --------------------------------------------------------------------------------------
# annotation-bug detection (endo 'T'=Tumor + multi-letter "select-all" mislabels)
# --------------------------------------------------------------------------------------

_OPTION_RE = re.compile(r"(?<![A-Za-z])([A-Z])[.):]\s")


def offered_option_letters(question):
    """Recover the lettered choices embedded inline in the question text (no structured field)."""
    return set(_OPTION_RE.findall(str(question)))


def classify_bug(task_norm, question, answer):
    """Return (is_flagged, bug_type) for a single-label classification row.

    bug_type in {"answer_not_in_options", "multilabel_mislabel", ""}.
    Only single-label `classification` rows are scanned; everything else is clean by definition.
    """
    if task_norm != "classification":
        return False, ""
    offered = offered_option_letters(question)
    if not offered:
        return False, ""  # no lettered options -> can't judge; treat as clean
    ans = str(answer).strip()
    if not ans:
        return False, ""
    tokens = [t for t in re.split(r"[\s,]+", ans) if t]
    if len(tokens) > 1 and all(t in offered for t in tokens):
        return True, "multilabel_mislabel"      # e.g. 'A B' on a "select all" question
    if any(t not in offered for t in tokens):
        return True, "answer_not_in_options"     # e.g. 'T' (Tumor) instead of the letter
    return False, ""


# --------------------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------------------

def load_split_records(data_root, split_dirname):
    """Walk `<data_root>/<split_dirname>/<Modality>/<dataset>/*.json` and flatten to records.

    Returns list of dicts with derived: modality, dataset, task_norm, sample_key, image_key, the FULL
    ordered absolute image list (uncapped), plus flag fields. Modality/dataset come from the folder
    path, NOT the (unreliable) field.
    """
    base = os.path.join(data_root, split_dirname)
    if not os.path.isdir(base):
        raise FileNotFoundError(f"Split dir not found: {base}")

    records = []
    seen_keys = set()
    json_paths = sorted(glob.glob(os.path.join(base, "*", "*", "*.json")))
    if not json_paths:
        raise FileNotFoundError(f"No per-dataset JSON files under {base}/*/*/*.json")

    for jp in json_paths:
        # .../<split>/<Modality>/<dataset>/<file>.json
        parts = os.path.normpath(jp).split(os.sep)
        dataset = parts[-2]
        modality = parts[-3]
        dataset_dir = os.path.dirname(jp)  # ImageName is relative to this (matches inference.py)
        try:
            with open(jp, "r") as f:
                data = json.load(f)
        except Exception as e:  # noqa: BLE001
            print(f"  WARN: failed to read {jp}: {e}")
            continue
        if not isinstance(data, list):
            data = [data]

        for s in data:
            if not isinstance(s, dict):
                continue
            task_norm = str(s.get("TaskType", "")).strip().lower()
            if not task_norm:
                continue
            key = create_sample_key(s)
            if key in seen_keys:
                continue  # exact-duplicate (image,question) -- keep first, eval would dedup anyway
            seen_keys.add(key)
            flagged, bug_type = classify_bug(task_norm, s.get("Question", ""), s.get("Answer", ""))
            raw_imgs = ordered_image_list(s)
            records.append({
                "sample_key": key,
                "image_key": image_group_key(s),       # sorted -> grouping/stratification only
                "modality": modality,
                "dataset": dataset,
                "task_norm": task_norm,
                "question": str(s.get("Question", "")),
                "answer": str(s.get("Answer", "")),
                # ALL ordered, resolved-to-absolute image paths -- the materialized content, UNCAPPED
                "images": [os.path.abspath(os.path.join(dataset_dir, ip)) for ip in raw_imgs],
                "n_images": len(raw_imgs),
                "flagged_bug": flagged,
                "bug_type": bug_type,
            })
    return records


# --------------------------------------------------------------------------------------
# stratified image-group assignment (the one primitive used at every stage)
# --------------------------------------------------------------------------------------

def _largest_remainder(fracs, n):
    """Integer counts proportional to `fracs` (summing to 1) that sum exactly to n."""
    raw = [f * n for f in fracs]
    floors = [int(x) for x in raw]
    rem = n - sum(floors)
    order = sorted(range(len(fracs)), key=lambda i: raw[i] - floors[i], reverse=True)
    for i in range(rem):
        floors[order[i % len(order)]] += 1
    return floors


def stratified_assign(records, proportions, seed, restrict_groups=None):
    """Assign each (non-flagged) IMAGE-GROUP to a bucket named in `proportions` (name->frac, sums to 1),
    stratified by (dataset, sorted task-combo). Returns {image_key: bucket_name}.

    restrict_groups: if given (a set of image_keys), only those groups are assigned (the rest skipped).
    Groups stay whole -- every question on one image lands in the same bucket.
    """
    rng = random.Random(seed)
    group_recs = defaultdict(list)
    for r in records:
        if r["flagged_bug"]:
            continue
        if restrict_groups is not None and r["image_key"] not in restrict_groups:
            continue
        group_recs[r["image_key"]].append(r)

    strata = defaultdict(list)
    for gkey, recs in group_recs.items():
        dataset = recs[0]["dataset"]
        tasks = tuple(sorted({rr["task_norm"] for rr in recs}))
        strata[(dataset, tasks)].append(gkey)

    names = list(proportions.keys())
    fracs = [proportions[n] for n in names]
    out = {}
    for stratum, gkeys in sorted(strata.items()):
        gkeys = sorted(gkeys)
        rng.shuffle(gkeys)
        counts = _largest_remainder(fracs, len(gkeys))
        idx = 0
        for name, c in zip(names, counts):
            for gk in gkeys[idx:idx + c]:
                out[gk] = name
            idx += c
    return out


# --------------------------------------------------------------------------------------
# reporting / output
# --------------------------------------------------------------------------------------

def cell_counts(records, predicate):
    c = Counter()
    for r in records:
        if predicate(r):
            c[(r["modality"], r["dataset"], r["task_norm"])] += 1
    return c


def build_coverage_md(train_records, val_records):
    lines = ["# Internal split coverage report (v2)\n"]

    def section(title, recs, pred):
        cc = cell_counts(recs, pred)
        total = sum(cc.values())
        lines.append(f"\n## {title}  (n={total})\n")
        lines.append("| modality | dataset | task | n |")
        lines.append("|---|---|---|---|")
        for (mod, ds, task), n in sorted(cc.items()):
            lines.append(f"| {mod} | {ds} | {task} | {n} |")
        return cc

    section("TRUE_TEST (untouched, in-distribution)", train_records, lambda r: r["split"] == "true_test")
    a_cc = section("A (half, full)", train_records, lambda r: r["split"] == "A")
    section("  A_train", train_records, lambda r: r["split"] == "A" and r.get("role") == "train")
    section("  A_dev", train_records, lambda r: r["split"] == "A" and r.get("role") == "dev")
    section("  A_prime (30% quick set)", train_records, lambda r: r["split"] == "A" and r.get("in_prime"))
    b_cc = section("B (half, full)", train_records, lambda r: r["split"] == "B")
    section("  B_train", train_records, lambda r: r["split"] == "B" and r.get("role") == "train")
    section("  B_dev", train_records, lambda r: r["split"] == "B" and r.get("role") == "dev")
    section("  B_prime (30% quick set)", train_records, lambda r: r["split"] == "B" and r.get("in_prime"))
    section("REMOVED (annotation bug, quarantined)", train_records, lambda r: r["split"] == "removed")
    pt_cc = section("VALIDATION-PUBLIC pseudo-test (leaked excluded)", val_records, lambda r: not r.get("leaked"))
    section("VALIDATION-PUBLIC clean (leaked + bugs removed)", val_records,
            lambda r: not r.get("leaked") and not r["flagged_bug"])
    section("VALIDATION-PUBLIC leaked (excluded from pseudo-test)", val_records, lambda r: r.get("leaked"))

    # coverage gap: cells in true_test / A / B that validation-public cannot see (FLARE ranks per dataset)
    train_cells = {(ds, task) for (_m, ds, task) in (a_cc + b_cc)}
    pt_cells = {(ds, task) for (_m, ds, task) in pt_cc}
    gap = sorted(train_cells - pt_cells)
    lines.append("\n## Coverage gap: in A/B but NOT in validation-public\n")
    lines.append("These (dataset, task) cells are ranked in the real challenge but have **zero** "
                 "validation-public coverage -- only the training-derived splits can score them.\n")
    lines.append("| dataset | task |")
    lines.append("|---|---|")
    for ds, task in gap:
        lines.append(f"| {ds} | {task} |")

    pt_tasks = {task for (_m, _d, task) in pt_cc}
    train_tasks = {task for (_m, _d, task) in (a_cc + b_cc)}
    lines.append("\n## Task-types missing from validation-public entirely\n")
    missing_tasks = sorted(train_tasks - pt_tasks)
    lines.append(", ".join(missing_tasks) if missing_tasks else "(none)")

    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_root", required=True,
                    help="Root of FLARE-Task5-MLLM-2D (contains training/ and validation-public/)")
    ap.add_argument("--out_dir", default="./internal_splits_v2")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--frac_true_test", type=float, default=0.10, help="untouched in-distribution test")
    ap.add_argument("--frac_A", type=float, default=0.45,
                    help="size of half A; B gets the rest (1 - true_test - A)")
    ap.add_argument("--frac_dev", type=float, default=0.10, help="dev slice WITHIN each half")
    ap.add_argument("--frac_prime", type=float, default=0.30, help="quick-set subset WITHIN each half")
    ap.add_argument("--leaked_images", default=None,
                    help="leaked_images.json from audit_data.py --check-leakage; its validation-public "
                         "images are dropped from the pseudo-test so we only score UNSEEN images.")
    ap.add_argument("--audit_issues", default=None,
                    help="audit_issues.csv from audit_data.py; training samples whose sample_key is "
                         "flagged with a --remove_reasons code are additionally quarantined.")
    ap.add_argument("--remove_reasons",
                    default="detection_bad_geometry,contradictory_label,exact_duplicate,"
                            "regression_outlier,image_corrupt,missing_image",
                    help="comma-separated audit reason codes to quarantine from training (endo is already "
                         "caught built-in).")
    args = ap.parse_args()

    frac_B = round(1.0 - args.frac_true_test - args.frac_A, 6)
    if frac_B <= 0:
        raise ValueError(f"frac_true_test + frac_A must be < 1 (got B={frac_B})")
    os.makedirs(args.out_dir, exist_ok=True)

    # optional cleaning inputs from the audit -------------------------------------------
    leaked_val_basenames = set()
    if args.leaked_images and os.path.exists(args.leaked_images):
        leaked_val_basenames = set(json.load(open(args.leaked_images)).get("validation_public_basenames", []))
        print(f"  leaked-image exclusions loaded: {len(leaked_val_basenames)} validation-public images")

    audit_remove = {}  # sample_key -> reason
    if args.audit_issues and os.path.exists(args.audit_issues):
        wanted = {x.strip() for x in args.remove_reasons.split(",") if x.strip()}
        with open(args.audit_issues) as f:
            for row in csv.DictReader(f):
                if row["reason"] in wanted and row["split_src"] == "training" and row["sample_key"]:
                    audit_remove[row["sample_key"]] = row["reason"]
        print(f"  audit-driven training removals loaded: {len(audit_remove)} samples ({sorted(wanted)})")

    print("Loading training/ ...")
    train_records = load_split_records(args.data_root, "training")
    print(f"  {len(train_records)} training samples")
    print("Loading validation-public/ ...")
    val_records = load_split_records(args.data_root, "validation-public")
    print(f"  {len(val_records)} validation-public samples")

    # fold audit removals into the built-in flag machinery (flagged -> quarantined)
    n_audit_applied = 0
    for r in train_records:
        if not r["flagged_bug"] and r["sample_key"] in audit_remove:
            r["flagged_bug"] = True
            r["bug_type"] = audit_remove[r["sample_key"]]
            n_audit_applied += 1
    if audit_remove:
        print(f"  audit removals applied to training: {n_audit_applied}")

    # mark leaked validation-public samples (any image byte-identical to a training image)
    n_leaked = 0
    for r in val_records:
        basenames = {os.path.basename(p) for p in r["image_key"].split("|")}
        r["leaked"] = bool(basenames & leaked_val_basenames)
        n_leaked += r["leaked"]
    if leaked_val_basenames:
        print(f"  validation-public samples dropped from pseudo-test (leaked): {n_leaked}")

    n_flag_train = sum(r["flagged_bug"] for r in train_records)
    bug_breakdown = Counter((r["dataset"], r["bug_type"]) for r in train_records if r["flagged_bug"])
    print(f"  flagged (annotation bug): train={n_flag_train}")
    for (ds, bt), n in sorted(bug_breakdown.items()):
        print(f"    train {ds:14s} {bt:22s} {n}")

    # --- STAGE 1: top-level true_test / A / B (image-group, stratified) ----------------
    print("Stage 1: true_test / A / B ...")
    top = stratified_assign(
        train_records, {"true_test": args.frac_true_test, "A": args.frac_A, "B": frac_B}, args.seed)
    a_groups = {g for g, b in top.items() if b == "A"}
    b_groups = {g for g, b in top.items() if b == "B"}

    # --- STAGE 2: train/dev WITHIN each half ------------------------------------------
    print("Stage 2: train/dev within A and within B ...")
    role_A = stratified_assign(
        train_records, {"train": 1 - args.frac_dev, "dev": args.frac_dev}, args.seed + 10,
        restrict_groups=a_groups)
    role_B = stratified_assign(
        train_records, {"train": 1 - args.frac_dev, "dev": args.frac_dev}, args.seed + 20,
        restrict_groups=b_groups)
    role = {**role_A, **role_B}

    # --- STAGE 3: 30% prime quick-set WITHIN each half --------------------------------
    print("Stage 3: prime (A', B') quick subsets ...")
    prime_A = stratified_assign(
        train_records, {"prime": args.frac_prime, "rest": 1 - args.frac_prime}, args.seed + 30,
        restrict_groups=a_groups)
    prime_B = stratified_assign(
        train_records, {"prime": args.frac_prime, "rest": 1 - args.frac_prime}, args.seed + 40,
        restrict_groups=b_groups)
    prime = {g for g, b in {**prime_A, **prime_B}.items() if b == "prime"}

    # apply assignments to every record
    for r in train_records:
        if r["flagged_bug"]:
            r["split"] = "removed"
            r["role"] = None
            r["in_prime"] = False
        else:
            r["split"] = top.get(r["image_key"], "A")  # tiny-stratum fallback -> A
            r["role"] = role.get(r["image_key"])         # train/dev for A,B
            r["in_prime"] = r["image_key"] in prime
    for r in val_records:
        r["split"] = "validation-public"
        r["role"] = None
        r["in_prime"] = False

    # --- materialize per-split JSONL (self-contained, ALL images, no cap) --------------
    def to_record(r):
        return {
            "sample_key": r["sample_key"],
            "dataset": r["dataset"],
            "modality": r["modality"],
            "task": r["task_norm"],
            "question": r["question"],
            "answer": r["answer"],
            "image": r["images"],            # ALL images, uncapped (packing happens in export)
            "n_images": r["n_images"],
            "role": r.get("role"),           # train/dev for A,B; null otherwise
            "in_prime": bool(r.get("in_prime", False)),
            "flagged_bug": bool(r["flagged_bug"]),
            "bug_type": r["bug_type"],
            "leaked": bool(r.get("leaked", False)),
        }

    def dump_jsonl(name, recs, pred):
        rows = [to_record(r) for r in recs if pred(r)]
        with open(os.path.join(args.out_dir, name), "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        return {row["sample_key"] for row in rows}

    written = {}
    written["true_test.jsonl"] = dump_jsonl("true_test.jsonl", train_records, lambda r: r["split"] == "true_test")
    written["A.jsonl"] = dump_jsonl("A.jsonl", train_records, lambda r: r["split"] == "A")
    dump_jsonl("A_train.jsonl", train_records, lambda r: r["split"] == "A" and r.get("role") == "train")
    dump_jsonl("A_dev.jsonl", train_records, lambda r: r["split"] == "A" and r.get("role") == "dev")
    a_prime = dump_jsonl("A_prime.jsonl", train_records, lambda r: r["split"] == "A" and r.get("in_prime"))
    written["B.jsonl"] = dump_jsonl("B.jsonl", train_records, lambda r: r["split"] == "B")
    dump_jsonl("B_train.jsonl", train_records, lambda r: r["split"] == "B" and r.get("role") == "train")
    dump_jsonl("B_dev.jsonl", train_records, lambda r: r["split"] == "B" and r.get("role") == "dev")
    b_prime = dump_jsonl("B_prime.jsonl", train_records, lambda r: r["split"] == "B" and r.get("in_prime"))
    written["flagged_removed.jsonl"] = dump_jsonl(
        "flagged_removed.jsonl", train_records, lambda r: r["split"] == "removed")
    written["validation-public.jsonl"] = dump_jsonl(
        "validation-public.jsonl", val_records, lambda r: not r.get("leaked"))
    dump_jsonl("validation-public-clean.jsonl", val_records,
               lambda r: not r.get("leaked") and not r["flagged_bug"])
    dump_jsonl("validation-public-leaked.jsonl", val_records, lambda r: r.get("leaked"))

    # --- leak-free guarantee: the mutually-exclusive PARTITIONS must be disjoint -------
    partitions = ["true_test.jsonl", "A.jsonl", "B.jsonl", "flagged_removed.jsonl", "validation-public.jsonl"]
    for i in range(len(partitions)):
        for j in range(i + 1, len(partitions)):
            a, b = partitions[i], partitions[j]
            overlap = written[a] & written[b]
            if overlap:
                raise AssertionError(
                    f"LEAK: {len(overlap)} sample_key(s) in both {a} and {b} (e.g. {next(iter(overlap))[:80]})")
    # subset sanity: A' subset of A, B' subset of B
    if not a_prime <= written["A.jsonl"]:
        raise AssertionError("A_prime is not a subset of A")
    if not b_prime <= written["B.jsonl"]:
        raise AssertionError("B_prime is not a subset of B")
    print("  asserts passed: true_test/A/B/removed/validation-public disjoint; A',B' subsets of A,B")

    # --- CSV audit mirror (flat, scalar) ----------------------------------------------
    manifest_path = os.path.join(args.out_dir, "split_manifest.csv")
    with open(manifest_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["source", "split", "role", "in_prime", "leaked", "modality", "dataset",
                    "task_norm", "flagged_bug", "bug_type", "n_images", "sample_key"])
        for src, recs in (("training", train_records), ("validation-public", val_records)):
            for r in recs:
                w.writerow([src, r["split"], r.get("role") or "", int(bool(r.get("in_prime"))),
                            int(bool(r.get("leaked"))), r["modality"], r["dataset"], r["task_norm"],
                            int(bool(r["flagged_bug"])), r["bug_type"], r["n_images"], r["sample_key"]])

    # --- coverage report --------------------------------------------------------------
    md = build_coverage_md(train_records, val_records)
    with open(os.path.join(args.out_dir, "coverage_report.md"), "w") as f:
        f.write(md)

    # --- console summary --------------------------------------------------------------
    split_sizes = Counter(r["split"] for r in train_records)
    n_a_train = sum(1 for r in train_records if r["split"] == "A" and r.get("role") == "train")
    n_a_dev = sum(1 for r in train_records if r["split"] == "A" and r.get("role") == "dev")
    n_b_train = sum(1 for r in train_records if r["split"] == "B" and r.get("role") == "train")
    n_b_dev = sum(1 for r in train_records if r["split"] == "B" and r.get("role") == "dev")
    print("\nTRAIN-distribution split sizes (samples):")
    print(f"  true_test      {split_sizes.get('true_test', 0)}")
    print(f"  A              {split_sizes.get('A', 0)}  (train {n_a_train} / dev {n_a_dev} / prime {len(a_prime)})")
    print(f"  B              {split_sizes.get('B', 0)}  (train {n_b_train} / dev {n_b_dev} / prime {len(b_prime)})")
    print(f"  removed        {split_sizes.get('removed', 0)}")
    n_honest = sum(1 for r in val_records if not r.get("leaked"))
    n_honest_clean = sum(1 for r in val_records if not r.get("leaked") and not r["flagged_bug"])
    print(f"  validation-public {n_honest} (pseudo-test, leaked {n_leaked} excluded; clean={n_honest_clean})")
    n_multi = sum(1 for r in train_records + val_records if r["n_images"] > 1)
    max_imgs = max((r["n_images"] for r in train_records + val_records), default=0)
    print(f"\nMulti-image: {n_multi} samples have >1 image; max {max_imgs} images on a sample "
          f"(ALL stored, uncapped -- packing is export_sharegpt's job).")
    print(f"Wrote per-split .jsonl + split_manifest.csv + coverage_report.md to {args.out_dir}")


if __name__ == "__main__":
    main()
