#!/usr/bin/env python3
"""CRIMSON scoring step for report generation (the official FLARE 2026 Task-3 report-gen metric).

CRIMSON is what the FLARE 2026 EVAL page actually scores report generation with (the EVAL spec lists
"Report Generation -> CRIMSON score"; arxiv 2603.06183). GREEN was only ever used internally as a
feasible proxy because the `crimson-score` package CANNOT be installed into the InternVL env
`flare25-internvl` -- it pins torch>=2.10 / transformers>=5.3 / pandas>=3.0 (needs Python>=3.11),
which hard-conflicts with the pinned InternVL3 stack (torch 2.3.1, transformers 4.x, Python 3.10).

So this scorer is DECOUPLED: it takes a predictions.jsonl of {gt, pred} and runs CRIMSON offline over
saved predictions. Run it from a SEPARATE Python>=3.11 env that has `crimson-score` installed -- it
never imports the model and never touches the InternVL env.

CRIMSON API (mirrors how Previous_work/FLARE26-MedGemma/mle/engine/evaluate.py calls it):
  from CRIMSON import CRIMSONScore
  scorer = CRIMSONScore(api="hf", model_name=<optional local path or HF id>)
  results = scorer.evaluate_batch(refs, hyps, patient_contexts=..., include_guidelines=True, batch_size=B)
Each per-sample result is a dict with a "crimson_score" (and error_counts / metrics breakdowns). We
report the MEAN crimson_score over samples.

Patient context (Age / Indication) is optional; our report-gen data (e.g. IU_XRay) has none, so we
pass None per sample. Pass --context_keys later if a dataset ships those fields.

Usage
-----
  python score_crimson.py \
      --pred results/reportgen/k2_s1337/predictions.jsonl \
      --crimson_api hf \
      --crimson_model $HOME/models/CRIMSON-model \
      --out results/reportgen/k2_s1337/crimson_metrics.json
"""

import argparse
import json
import math
import os


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", required=True, help="predictions.jsonl with {gt, pred, sample_key} per line")
    ap.add_argument("--crimson_api", default=os.environ.get("CRIMSON_API", "hf"),
                    help="CRIMSON backend api: hf | huggingface | vllm | (other -> per-sample evaluate())")
    ap.add_argument("--crimson_model", default=os.environ.get("CRIMSON_MODEL", ""),
                    help="path to (or HF id of) the CRIMSON model. Stage it locally to avoid a download. "
                         "Empty -> let CRIMSONScore pick its default.")
    ap.add_argument("--batch_size", type=int, default=int(os.environ.get("CRIMSON_BATCH_SIZE", "1")))
    ap.add_argument("--no_guidelines", action="store_true", default=False,
                    help="disable CRIMSON's clinical-guideline context (include_guidelines=False)")
    ap.add_argument("--out", required=True, help="metrics.json output")
    return ap.parse_args()


def load_pairs(path):
    refs, hyps, keys = [], [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            refs.append(str(r.get("gt", "") or ""))
            hyps.append(str(r.get("pred", "") or ""))
            keys.append(r.get("sample_key"))
    return refs, hyps, keys


def run_crimson(refs, hyps, *, api, model_name, batch_size, include_guidelines):
    try:
        from CRIMSON import CRIMSONScore
    except ImportError as exc:  # pragma: no cover - environment guard
        raise ImportError(
            "CRIMSON scoring requires the `crimson-score` package (imported as `CRIMSON`). "
            "Install it in a SEPARATE Python>=3.11 env -- it cannot live in flare25-internvl."
        ) from exc

    scorer_kwargs = {"api": api}
    if model_name:
        scorer_kwargs["model_name"] = model_name
    scorer = CRIMSONScore(**scorer_kwargs)

    contexts = [None] * len(refs)  # our report-gen data ships no Age/Indication context
    if api in {"hf", "huggingface", "vllm"} and hasattr(scorer, "evaluate_batch"):
        return scorer.evaluate_batch(
            list(refs), list(hyps),
            patient_contexts=contexts,
            include_guidelines=include_guidelines,
            batch_size=max(1, batch_size),
        )
    # fallback: per-sample
    out = []
    for ref, hyp, ctx in zip(refs, hyps, contexts):
        out.append(scorer.evaluate(
            reference_findings=ref,
            predicted_findings=hyp,
            patient_context=ctx,
            include_guidelines=include_guidelines,
        ))
    return out


def main():
    args = parse_args()
    refs, hyps, keys = load_pairs(args.pred)
    print(f"scoring {len(refs)} report pairs with CRIMSON (api={args.crimson_api}, "
          f"model={args.crimson_model or 'default'})")

    results = run_crimson(
        refs, hyps,
        api=args.crimson_api,
        model_name=args.crimson_model,
        batch_size=args.batch_size,
        include_guidelines=not args.no_guidelines,
    )

    per_sample_scores = []
    for res in results:
        s = res.get("crimson_score") if isinstance(res, dict) else None
        per_sample_scores.append(float(s) if s is not None else None)

    valid = [s for s in per_sample_scores if s is not None and not math.isnan(s)]
    mean = sum(valid) / len(valid) if valid else None

    metrics = {
        "task": "report_generation",
        "metric": "crimson_score",
        "crimson_api": args.crimson_api,
        "crimson_model": os.path.basename(str(args.crimson_model)) if args.crimson_model else "default",
        "n": len(refs),
        "n_scored": len(valid),
        "n_none": len(per_sample_scores) - len(valid),
        "crimson_score_mean": round(float(mean), 4) if mean is not None else None,
    }

    out_dir = os.path.dirname(os.path.abspath(args.out)) or "."
    os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(metrics, f, indent=2)
    # per-sample scores next to metrics, for error analysis
    per_sample_path = os.path.join(out_dir, "crimson_per_sample.jsonl")
    with open(per_sample_path, "w") as f:
        for k, s in zip(keys, per_sample_scores):
            f.write(json.dumps({"sample_key": k, "crimson_score": s}) + "\n")

    print(f"\n=== CRIMSON ({metrics['crimson_model']}) ===")
    print(f"  n={metrics['n']} scored={metrics['n_scored']} none={metrics['n_none']}")
    print(f"  mean crimson_score = {metrics['crimson_score_mean']}")
    print(f"  -> {args.out}")


if __name__ == "__main__":
    main()
