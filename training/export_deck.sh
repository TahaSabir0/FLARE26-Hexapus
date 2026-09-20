#!/bin/bash
# Build the ShareGPT training deck for one route from the frozen split JSONLs and register it in
# training/data/dataset_info.json (LLaMA-Factory reads decks from `dataset_dir`).
#
#   SPLITS=/path/to/internal_splits_v2 bash training/export_deck.sh <route> [TRAIN_SPLITS] [OUT_TAG]
#
#   <route>      cls_rebal | multilabel_noavg | counting | detection | regmix_A5 | reportgen | general
#   TRAIN_SPLITS default "A B" (development pool). The submitted classification, multi-label,
#                counting and detection adapters used "A B true_test validation-public" with OUT_TAG=_full.
#   OUT_TAG      suffix folded into the deck name ("" for A∪B, "_full" for the all-data decks).
#
# Common to every deck: K=1 flattening of multi-image rows (--max_images 1, general uses the flatten
# policy), per-image "Image-i:" markers, endoscopy excluded, instance detection dropped, flagged
# (annotation-bug) rows dropped.
set -euo pipefail
ROUTE="${1:?route}"; TRAIN_SPLITS="${2:-A B}"; OUT_TAG="${3:-}"
SPLITS="${SPLITS:?set SPLITS=<dir with A.jsonl B.jsonl true_test.jsonl validation-public.jsonl>}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$HERE/../scripts:${PYTHONPATH:-}"
DATA="$HERE/data"; mkdir -p "$DATA"
IN=(); for s in $TRAIN_SPLITS; do IN+=("$SPLITS/$s.jsonl"); done
COMMON=(--markers --exclude_dataset endo)
TASK=""; VARIANT="none"; REBAL="none"; POLICY=no; HAS_SYSTEM=no
case "$ROUTE" in
  cls_rebal)        TASK="classification"; REBAL="k2a" ;;
  multilabel_noavg) TASK="multi-label classification"; VARIANT="medical_cot"; HAS_SYSTEM=yes ;;
  counting)         TASK="counting"; VARIANT="cot"; HAS_SYSTEM=yes ;;
  detection)        TASK="detection" ;;
  reportgen)        TASK="report_generation" ;;
  general)          POLICY=yes ;;
  regmix_A5)        ;;  # handled below
  *) echo "unknown route $ROUTE"; exit 1 ;;
esac

register() {  # register NAME HAS_SYSTEM
python - "$DATA/dataset_info.json" "$1" "$2" <<'PY'
import json, sys, os
p, name, has_sys = sys.argv[1], sys.argv[2], sys.argv[3] == "yes"
d = json.load(open(p)) if os.path.exists(p) else {}
cols = {"messages": "conversations", "images": "images"}
if has_sys: cols["system"] = "system"
d[name] = {"file_name": name + ".jsonl", "formatting": "sharegpt", "columns": cols}
json.dump(d, open(p, "w"), indent=2); print("registered", name)
PY
}

if [ "$ROUTE" = "regmix_A5" ]; then
  # Regression donor mixture: all regression rows + a 5,000-row capped slice of bcn20000
  # (dermatology classification), both bare-prompt, concatenated. cap_seed 1337 reproduces the slice.
  NAME="flare_regmix_A5_dermacls${OUT_TAG:-_AB}"
  REG="$DATA/flare_regmix_regbase${OUT_TAG}.jsonl"; ADD="$DATA/flare_regmix_A5_added${OUT_TAG}.jsonl"
  python "$HERE/../scripts/export_sharegpt.py" --in_jsonl "${IN[@]}" --out_jsonl "$REG" \
      --task regression --max_images 1 --prompt_variant none "${COMMON[@]}"
  python "$HERE/../scripts/export_sharegpt.py" --in_jsonl "${IN[@]}" --out_jsonl "$ADD" \
      --only_dataset bcn20000 --cap_total 5000 --cap_seed 1337 --max_images 1 --prompt_variant none "${COMMON[@]}"
  cat "$REG" "$ADD" > "$DATA/$NAME.jsonl"
  register "$NAME" no
  echo "deck $NAME: $(wc -l < "$REG") regression + $(wc -l < "$ADD") bcn20000 = $(wc -l < "$DATA/$NAME.jsonl") rows"
  exit 0
fi

NAME="flare_final_${ROUTE}${OUT_TAG}"
ARGS=(--in_jsonl "${IN[@]}" --out_jsonl "$DATA/$NAME.jsonl" --prompt_variant "$VARIANT" --rebalance "$REBAL" "${COMMON[@]}")
[ -n "$TASK" ] && ARGS+=(--task "$TASK")
if [ "$POLICY" = yes ]; then ARGS+=(--apply_policy); else ARGS+=(--max_images 1); fi
python "$HERE/../scripts/export_sharegpt.py" "${ARGS[@]}"
register "$NAME" "$HAS_SYSTEM"
echo "deck $NAME: $(wc -l < "$DATA/$NAME.jsonl") rows"
