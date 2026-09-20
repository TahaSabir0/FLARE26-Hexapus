#!/usr/bin/env python3
"""Render a self-contained split JSONL (from internal_eval/build_splits.py, v2) into the
LLaMA-Factory ShareGPT format InternVL training consumes.

v2 changes (multi-image experiment program — see FLARE2026_data_pipeline_v2.md §6)
-------------------------------------------------------------------------------------------------------
The v2 split files are **lossless**: every QA pair carries its FULL, uncapped `image` list. All
multi-image packing therefore happens HERE, at the export layer (training-only), not in the splits.

This script now owns three knobs:

  * `--max_images K`  -- the packing chunk size. A pair with X images becomes ceil(X / K) ShareGPT
                         rows, each fed up to K images, **no image ever dropped**. K=1 == last year's
                         flatten. Omit it to keep the whole pair in one row (no chunking).
  * `--markers`       -- prefix each image with an `Image-{i}:` label (i restarts per row). This is the
                         per-image-marker formatting the multi-image papers (MMICL / Med-MIM /
                         LLaVA-NeXT-Interleave) say is the difference between gain and degradation.
                         OFF == bare `<image>\\n<image>` concatenation (the v1 / last-year behavior, which
                         the LLaMA-Factory InternVL plugin renders with NO index between images).
  * `--task` / `--exclude_task` -- keep one task (per-task adapter subsets) / drop tasks. By default we
                         drop `instance_detection` (not a scored FLARE-2026 task; see
                         project-instance-detection-dead-task).

NOTE: packing is by *image count* (K). The tile-budget packing paradigm (bin-pack by total tiles)
is deferred -- for classification the multi-image mass is X=2 pairs that rarely overflow, so per-image
K is sufficient for the Phase-1 sweep.

    neutral split record (v2)                  ShareGPT training line(s)
    --------------------                       ----------------------
    {question, answer, image:[p1,p2,p3],       chunk by K -> one row per chunk, e.g. K=2:
     task, dataset, flagged_bug, ...}            row1 images [p1,p2], row2 images [p3]
                                               human (with --markers):
                                                 "Image-1: <image>\\nImage-2: <image>\\n<question>"

Usage
-----
  # K=1 flatten, classification only, markers ON:
  python export_sharegpt.py --in_jsonl  .../internal_splits_v2/A_prime.jsonl \\
                            --out_jsonl ../data/flare_cls_A_k1.jsonl \\
                            --task classification --max_images 1 --markers

  # K=2 group pairs:
  python export_sharegpt.py --in_jsonl .../A_prime.jsonl --out_jsonl ../data/flare_cls_A_k2.jsonl \\
                            --task classification --max_images 2 --markers
"""

import argparse
import json
import math
import os

from prompt_variants import list_variants  # registry-derived --prompt_variant choices (no stale hardcoded list)


# --- per-dataset-K policy (DATASET/STATS.md §11d; removal:) -------------------
# Production training FLATTENS everything at K=1 (last-year behavior). This dict is the hook for
# grouping specific datasets at a proven K (enabled with --apply_policy); it is currently EMPTY.
# neojaundice K=3 -- the only grouping candidate (+0.113 in the controlled cls-only OFAT §11c) -- was
# tested END-TO-END in the general and cls-expert adapters and did NOT help on the held-out sets
# (per-dataset macro: flatten >= policy; the OFAT gain did not transfer). So it was removed; with the
# dict empty, --apply_policy simply flattens (K=1). Re-add a dataset here only if a future end-to-end
# result justifies it. (The --dataset_k CLI override still works for one-off packing experiments.)
POLICY_DATASET_K = {}


def chunk_images(images, k):
    """Split an ordered image list into ceil(len/k) chunks of <= k, no image dropped.
    k=None -> a single chunk with all images (no chunking)."""
    if not images:
        return [[]]
    if k is None or k >= len(images):
        return [images]
    return [images[i:i + k] for i in range(0, len(images), k)]


def parse_dataset_k(pairs):
    """['neojaundice=3','CMMD=2'] -> {'neojaundice':3,'CMMD':2}. The per-dataset K override
    used by the OFAT packing sweep: a sample's chunk size is dataset_k[dataset] if present,
    else the global --max_images. Lets one export build 'everything K=1 except dataset X at K'."""
    out = {}
    for p in pairs or []:
        if "=" not in p:
            raise ValueError(f"--dataset_k expects NAME=K, got {p!r}")
        name, k = p.rsplit("=", 1)
        k = int(k)
        if k < 1:
            raise ValueError(f"--dataset_k K must be >= 1, got {p!r}")
        out[name] = k
    return out


def render(question, answer, images, image_token, markers, format_instr="", system=""):
    """One chunk -> ShareGPT dict. The human turn is composed, top to bottom, of the parts present:
    image block -> format_instr (CoT instruction, Veronika's variant) -> question. Each layer is
    optional so the bare path and the CoT path share one code path. `system` (the CoT system prompt)
    becomes the ShareGPT `system` field. ALL of this must match infer_single_adapter.py /
    docker_internvl/inference.py exactly (train<->infer byte-identity)."""
    if images:
        if markers:
            image_block = "\n".join(f"Image-{i + 1}: {image_token}" for i in range(len(images)))
        else:
            image_block = "\n".join([image_token] * len(images))
        parts = [image_block]
    else:
        parts = []
    if format_instr:
        parts.append(format_instr)
    parts.append(question)
    out = {
        "conversations": [
            {"from": "human", "value": "\n".join(parts)},
            {"from": "gpt", "value": str(answer)},
        ],
    }
    if images:
        out["images"] = list(images)
    if system:
        out["system"] = system
    return out


def rebalance_k2a(rows, seed=1337):
    """K2a moderate class rebalancing (Classification_Stats §3; the deck-builder was never committed,
    so this reconstructs the measured recipe). Oversample CLASSIFICATION rows by duplicating, keyed on
    (dataset, answer): target = count if count>=250 else min(count*6, 250) -- a 6x cap with a ~250 floor.
    e.g. triple-neg 37->222 (6x cap binds), a 50-count class -> 250 (floor binds), luminal-b 296 -> 296
    (untouched). Only the +0.0086 macro K2a arm is reproduced (the aggressive K2b 12x arm overfit).

    `rows` is a list of (key_dict, sharegpt_dict) where key_dict has 'task','dataset','answer'.
    Returns a new oversampled row list (original order preserved, duplicates appended deterministically).
    """
    FLOOR, CAP = 250, 6
    import collections, random as _random
    groups = collections.defaultdict(list)
    for i, (k, _) in enumerate(rows):
        if (k.get("task") or "").strip().lower() == "classification":
            groups[(k.get("dataset"), k.get("answer"))].append(i)
    extra = []
    n_grp_oversampled = 0
    for (ds, ans), idxs in groups.items():
        count = len(idxs)
        target = count if count >= FLOOR else min(count * CAP, FLOOR)
        if target <= count:
            continue
        n_grp_oversampled += 1
        rng = _random.Random(f"{seed}|{ds}|{ans}")
        need = target - count
        # full repeats then a seeded partial top-up, so duplication is deterministic
        full, rem = divmod(need, count)
        dup_idx = idxs * full + rng.sample(idxs, rem)
        extra.extend(dup_idx)
    out = [r for r in rows] + [rows[i] for i in extra]
    print(f"  rebalance K2a: oversampled {n_grp_oversampled} (dataset,answer) cls groups, "
          f"+{len(extra)} rows ({len(rows)} -> {len(out)})")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in_jsonl", required=True, nargs="+",
                    help="one or more v2 split files from build_splits.py (e.g. A_prime.jsonl, or "
                         "'A.jsonl B.jsonl' to train the production model on the full A∪B pool). Concatenated in order.")
    ap.add_argument("--out_jsonl", required=True, help="ShareGPT output for LLaMA-Factory")
    ap.add_argument("--max_images", type=int, default=None,
                    help="GLOBAL packing chunk size K (ceil(X/K) rows per pair, no image dropped). "
                         "K=1 == flatten. Omit = whole pair in one row. Overridden per-dataset by --dataset_k.")
    ap.add_argument("--dataset_k", action="append", default=None,
                    help="per-dataset chunk-size override NAME=K (repeatable), e.g. --dataset_k neojaundice=3. "
                         "Used by the OFAT sweep to pack ONE dataset at K while everything else stays at "
                         "the global --max_images (typically 1). Unlisted datasets fall back to --max_images.")
    ap.add_argument("--apply_policy", action="store_true", default=False,
                    help="apply the per-dataset-K policy (POLICY_DATASET_K, currently EMPTY) and default "
                         "the global K to 1 (flatten). This is the PRODUCTION training setting: with an "
                         "empty policy it flattens everything (last-year behavior). Explicit --dataset_k / "
                         "--max_images still override. Use with --markers.")
    ap.add_argument("--markers", action="store_true", default=False,
                    help="prefix each image with an 'Image-{i}:' label (per-image markers)")
    ap.add_argument("--mix", action="store_true", default=False,
                    help="MANTIS-style single+multi mix: for multi-image pairs, emit the K-grouped row(s) "
                         "AND the per-image flattened rows (no duplication for single-image cases)")
    ap.add_argument("--task", default=None, help="keep only this task, or a comma-separated list of tasks (per-task adapter subsets)")
    ap.add_argument("--exclude_task", action="append", default=None,
                    help="drop this task (repeatable). instance_detection is ALWAYS dropped on top of these.")
    ap.add_argument("--keep_instance_detection", action="store_true", default=False,
                    help="opt back IN to the dead instance_detection task (for the train-with-vs-without ablation)")
    ap.add_argument("--exclude_dataset", action="append", default=None,
                    help="drop this dataset (repeatable), e.g. --exclude_dataset endo to keep the tiny "
                         "label-noisy multi-frame endo tail out of a controlled run")
    ap.add_argument("--only_dataset", default=None,
                    help="INCLUDE filter: keep ONLY rows whose dataset is in this comma-separated set "
                         "(e.g. --only_dataset dental, or --only_dataset BUSI,BUS-UCLM). Applied alongside "
                         "--task/--exclude_*. Backward-compatible: unset = keep all datasets.")
    ap.add_argument("--cap_total", type=int, default=None,
                    help="volume-match cap: after all filtering (and rebalance), if more than N ShareGPT "
                         "rows were built, randomly subsample down to N (seeded by --cap_seed). unset = no cap.")
    ap.add_argument("--keys_out", default=None,
                    help="optional path: dump per-row provenance keys (task/dataset/answer) as JSONL, "
                         "row-aligned to --out_jsonl. Diagnostic only; does not change the deck.")
    ap.add_argument("--cap_seed", type=int, default=1337,
                    help="seed for the --cap_total subsample (default 1337).")
    ap.add_argument("--drop_flagged", action="store_true", default=True,
                    help="exclude samples with flagged_bug=True (default on)")
    ap.add_argument("--keep_flagged", dest="drop_flagged", action="store_false",
                    help="override: keep flagged samples")
    ap.add_argument("--image_token", default="<image>")
    # --- Veronika's CoT prompt variant (vendored prompt_variants.py / format_templates.py) ----------
    # ADOPTED for the COUNTING adapter only (Modeling Report §2: -3.6 +/- 1.8 MAE, 3 seeds, both
    # directions). Keep 'none' (bare) for every other task. Train<->infer MUST match:
    # infer_single_adapter.py --prompt_variant <same>. NOTE: measured on InternVL3 -- re-run the
    # Phase-0 byte-identity smoke on Qwen3-VL before trusting it (its CoT system prompt is non-empty,
    # which interacts with Qwen3's omit_system path).
    ap.add_argument("--prompt_variant", default="none",
                    choices=["none", *list_variants()],
                    help="per-task format instruction + system prompt from the vendored registry. "
                         "'none'/'bare_baseline' = bare (control). 'cot' = the CoT package (adopt for counting). "
                         "'pixel_grounded_multi_adapter' = Veronika's pixel-grounding (adopt for classification).")
    # --- K2a moderate class rebalancing (Classification_Stats §3; ADOPTED, +0.0086 cls macro) --------
    ap.add_argument("--rebalance", default="none", choices=["none", "k2a"],
                    help="oversample rare CLASSIFICATION classes keyed on (dataset,answer): "
                         "k2a = 6x cap with a ~250 floor (the only positive arm). measured on InternVL3 "
                         "-> confirm with a Qwen3 A<->B flip before locking.")
    args = ap.parse_args()

    if args.max_images is not None and args.max_images < 1:
        ap.error("--max_images must be >= 1")
    dataset_k = parse_dataset_k(args.dataset_k)
    if args.apply_policy:
        # policy fills in any dataset NOT explicitly overridden on the CLI; global K defaults to 1 (flatten)
        for name, k in POLICY_DATASET_K.items():
            dataset_k.setdefault(name, k)
        if args.max_images is None:
            args.max_images = 1
    # instance_detection is a dead task (not scored in FLARE 2026) -- ALWAYS dropped, plus any
    # user-supplied exclusions on top. (To train on it for an ablation, pass --keep_instance_detection.)
    exclude = set(args.exclude_task or [])
    if not args.keep_instance_detection:
        exclude.add("instance_detection")
    exclude_ds = set(args.exclude_dataset or [])
    only_ds = set(d.strip() for d in args.only_dataset.split(",")) if args.only_dataset else None

    # Veronika's prompt variant (lazy import so the bare path needs no extra module)
    variant = None
    if args.prompt_variant != "none":
        from prompt_variants import get_variant
        variant = get_variant(args.prompt_variant)

    os.makedirs(os.path.dirname(os.path.abspath(args.out_jsonl)), exist_ok=True)

    n_in = 0
    n_drop_flag = n_drop_task = n_drop_excl = n_missing_img = 0
    rows = []  # list of (key_dict, sharegpt_dict); accumulated so --rebalance can oversample by group
    for in_path in args.in_jsonl:
        for line in open(in_path):
            line = line.strip()
            if not line:
                continue
            n_in += 1
            rec = json.loads(line)
            if args.drop_flagged and rec.get("flagged_bug"):
                n_drop_flag += 1
                continue
            task = rec.get("task")
            if args.task and task not in [t.strip() for t in args.task.split(",")]:
                n_drop_task += 1
                continue
            if task in exclude:
                n_drop_excl += 1
                continue
            if rec.get("dataset") in exclude_ds:
                n_drop_excl += 1
                continue
            if only_ds is not None and rec.get("dataset") not in only_ds:
                n_drop_excl += 1
                continue
            images = list(rec.get("image", []))
            missing = [p for p in images if not os.path.exists(p)]
            if missing:
                n_missing_img += 1
                continue
            # CoT layer: per-task format instruction + system prompt (empty when variant off)
            format_instr = system = ""
            if variant is not None:
                format_instr = variant.get_format_instruction(task or "", rec["question"])
                system = variant.system_prompt
            k_used = dataset_k.get(rec.get("dataset"), args.max_images)
            chunks = chunk_images(images, k_used)
            if args.mix and len(images) > 1 and k_used not in (None, 1):
                # MANTIS mix: grouped row(s) + the individual per-image rows (skip if already flattened)
                chunks = chunks + chunk_images(images, 1)
            key = {"task": task, "dataset": rec.get("dataset"), "answer": rec.get("answer", "")}
            for chunk in chunks:
                rows.append((key, render(rec["question"], rec.get("answer", ""), chunk,
                                         args.image_token, args.markers, format_instr, system)))

    if args.rebalance == "k2a":
        rows = rebalance_k2a(rows)

    n_before_cap = len(rows)
    if args.cap_total is not None and len(rows) > args.cap_total:
        import random
        rows = random.Random(args.cap_seed).sample(rows, args.cap_total)
    n_after_cap = len(rows)
    if args.cap_total is not None:
        print(f"  cap_total={args.cap_total} (seed={args.cap_seed}): {n_before_cap} -> {n_after_cap} rows")

    with open(args.out_jsonl, "w") as fout:
        for _key, row in rows:
            fout.write(json.dumps(row) + "\n")
    n_rows_out = len(rows)

    if args.keys_out:
        with open(args.keys_out, "w") as fk:
            for key, _row in rows:
                fk.write(json.dumps(key) + "\n")
        print(f"  wrote {n_rows_out} provenance keys -> {args.keys_out}")

    n_kept = n_in - n_drop_flag - n_drop_task - n_drop_excl - n_missing_img
    policy_note = " [policy applied]" if args.apply_policy else ""
    print(f"read {n_in} records -> kept {n_kept} -> wrote {n_rows_out} ShareGPT rows "
          f"(K={args.max_images}, dataset_k={dataset_k or '{}'}, markers={args.markers}, "
          f"prompt_variant={args.prompt_variant}, rebalance={args.rebalance}){policy_note}")
    print(f"  dropped: flagged={n_drop_flag}, task-filter={n_drop_task}, "
          f"excluded-task={n_drop_excl} ({sorted(exclude)}), missing-image={n_missing_img}")
    print(f"  out: {args.out_jsonl}")


if __name__ == "__main__":
    main()
