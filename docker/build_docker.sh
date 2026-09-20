#!/bin/bash
# Assemble the Docker build context (base model + router + 7 adapters) and build the image.
#
#   bash build_docker.sh            # the submitted route set
#
# Produces image tag  flare2026-task3-qwen35:final.
#
# Modeled on docker_qwen3/build_docker.sh (2026-08-20 rewrite) and keeps its two hard lessons:
#   (1) adapters listed ONE BY ONE as dest_name=absolute_source_path -- no tag templating
#       (the _full_s1337 name collision
#   (2) stage with readlink -f + cp -aL and hard-fail on dangling symlinks (the router
#       dangling-symlink bug that built a valid-looking image that died at startup).
set -euo pipefail

if [ "$#" -gt 0 ]; then
  echo "ERROR: build_docker.sh takes no arguments. Override a single route with an env var, e.g." >&2
  echo "         CLS_SRC=/path/to/qwen35_cls_rebal_full_2048_s1337 bash build_docker.sh" >&2
  echo "       (if the replacement was trained at a different pixel cap, ALSO set the matching" >&2
  echo "        <ROUTE>_PIXELS at docker run time -- resolution is part of the route)" >&2
  exit 2
fi

DOCKER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS_DIR="${SCRIPTS_DIR:-$DOCKER_DIR/../scripts}"
MODELS_SRC="${MODELS_SRC:-$DOCKER_DIR/../models}"
BASE_MODEL_DIR="$MODELS_SRC/Qwen3.5-9B"
ROUTER_DIR="$MODELS_SRC/FLARE-gliclass-small-v1.0"
DOCKER="${DOCKER:-$(command -v docker || command -v podman)}"
[ -n "$DOCKER" ] || { echo "no docker/podman found"; exit 1; }
IMAGE="${IMAGE:-flare2026-task3-qwen35:final}"

# THE 9B ROUTE SET -- dest_name = absolute source path. Dest names MUST match the defaults in
# docker_qwen35/inference.py + the ARG defaults in the Dockerfile. Each is env-overridable so a
# single route can be swapped without editing this file (e.g. a
# late-swap candidate). Default source = models/qwen35_final/ next to this repo.
SAVES="${SAVES:-$MODELS_SRC/qwen35_final}"

ADAPTERS=(
  "qwen35_cls_rebal_full_2048_s1337=${CLS_SRC:-$SAVES/qwen35_cls_rebal_full_2048_s1337}"
  "qwen35_multilabel_noavg_medcot_full_2048_s1337=${MULTILABEL_SRC:-$SAVES/qwen35_multilabel_noavg_medcot_full_2048_s1337}"
  "qwen35_detection_full_2048_s1337=${DETECTION_SRC:-$SAVES/qwen35_detection_full_2048_s1337}"
  "qwen35_counting_full_2048_s1337=${COUNTING_SRC:-$SAVES/qwen35_counting_full_2048_s1337}"
  "qwen35_regmix_A5_dermacls_AB_2048_s1337=${REGRESSION_SRC:-$SAVES/qwen35_regmix_A5_dermacls_AB_2048_s1337}"
  "qwen35_reportgen_AB_768_s1337=${REPORTGEN_SRC:-$SAVES/qwen35_reportgen_AB_768_s1337}"
  "qwen35_general_AB_768_s1337=${GENERAL_SRC:-$SAVES/qwen35_general_AB_768_s1337}"
)

# Stage under TMPDIR (needs ~25 GB free); always clean up on exit.
CTX_BASE="${TMPDIR:-/tmp}"
mkdir -p "$CTX_BASE"
CTX="$(mktemp -d "$CTX_BASE/flare_docker35_ctx.XXXXXX")"
cleanup_ctx() {
  local rc=$?
  if [ -n "${KEEP_CTX:-}" ]; then
    echo "=== KEEP_CTX set -- leaving build context at $CTX ==="
  else
    rm -rf "$CTX"
  fi
  return $rc
}
trap cleanup_ctx EXIT
echo "=== staging build context at $CTX (base: $CTX_BASE) ==="
df -h "$CTX_BASE" /var/lib/docker 2>/dev/null | sed 's/^/    /' || true

stage() {  # stage SRC DST : dereference symlinks, hardlink-tree if possible, else full copy
  # readlink -f is LOAD-BEARING: cp -al PRESERVES symlinks; a symlinked source becomes a
  # dangling absolute symlink inside the image (built silently, died at startup on 08-20).
  local src="$1" dst="$2"
  [ -e "$src" ] || { echo "MISSING: $src"; exit 1; }
  src="$(readlink -f "$src")"
  mkdir -p "$(dirname "$dst")"
  cp -aL "$src" "$dst" 2>/dev/null || cp -alL "$src" "$dst" 2>/dev/null || cp -a "$src" "$dst"
}

# 1) code: Dockerfile/requirements/predict.sh/inference.py + the 3 vendored helpers
#    (NO _deepstack_average.py -- no 9B route uses DeepStack averaging)
cp "$DOCKER_DIR/Dockerfile" "$DOCKER_DIR/requirements.txt" "$DOCKER_DIR/predict.sh" \
   "$DOCKER_DIR/inference.py" "$CTX/"
for h in vlm_prompt.py prompt_variants.py format_templates.py; do
  cp "$SCRIPTS_DIR/$h" "$CTX/$h"
done

# 2) models: base + router + the 7 adapters under models/qwen35_final/
stage "$BASE_MODEL_DIR" "$CTX/models/Qwen3.5-9B"
stage "$ROUTER_DIR"     "$CTX/models/FLARE-gliclass-small-v1.0"
for entry in "${ADAPTERS[@]}"; do
  dest="${entry%%=*}"; src="${entry#*=}"
  [ -d "$src" ] || { echo "MISSING adapter source for $dest: $src"; exit 1; }
  [ -f "$src/adapter_model.safetensors" ] || {
    echo "NOT AN ADAPTER (no adapter_model.safetensors): $src"; exit 1; }
  echo "  stage $dest  <-  $src"
  # stage ONLY the final adapter files -- exclude intermediate checkpoint-*/ training snapshots
  mkdir -p "$CTX/models/qwen35_final/$dest"
  for item in "$src"/*; do
    case "$(basename "$item")" in checkpoint-*) continue ;; esac
    stage "$item" "$CTX/models/qwen35_final/$dest/$(basename "$item")"
  done
done

# GUARD: fail loudly on anything that is not real content (dangling symlink => broken image).
dangling="$(find "$CTX" -xtype l 2>/dev/null || true)"
if [ -n "$dangling" ]; then
  echo "ERROR: dangling symlinks in the build context -- the image would be broken:" >&2
  echo "$dangling" >&2
  exit 1
fi
[ -s "$CTX/models/FLARE-gliclass-small-v1.0/config.json" ] || {
  echo "ERROR: router did not stage as real files" >&2; exit 1; }
[ -s "$CTX/models/Qwen3.5-9B/config.json" ] || {
  echo "ERROR: base model did not stage as real files" >&2; exit 1; }
for entry in "${ADAPTERS[@]}"; do
  dest="${entry%%=*}"
  [ -s "$CTX/models/qwen35_final/$dest/adapter_model.safetensors" ] || {
    echo "ERROR: adapter did not stage as real files: $dest" >&2; exit 1; }
done
echo "  guard OK: no dangling symlinks; base + router + 7 adapters are real files"

echo "=== context staged ==="
du -sh "$CTX" 2>/dev/null || true

# 3) build -- pass every route explicitly; no tag templating anywhere
"$DOCKER" build \
  --build-arg CLS_ADAPTER=qwen35_cls_rebal_full_2048_s1337 \
  --build-arg MULTILABEL_ADAPTER=qwen35_multilabel_noavg_medcot_full_2048_s1337 \
  --build-arg DETECTION_ADAPTER=qwen35_detection_full_2048_s1337 \
  --build-arg COUNTING_ADAPTER=qwen35_counting_full_2048_s1337 \
  --build-arg REGRESSION_ADAPTER=qwen35_regmix_A5_dermacls_AB_2048_s1337 \
  --build-arg REPORTGEN_ADAPTER=qwen35_reportgen_AB_768_s1337 \
  --build-arg GENERAL_ADAPTER=qwen35_general_AB_768_s1337 \
  -t "$IMAGE" "$CTX"

echo "=== built $IMAGE. image size (limit 35 GB): ==="
"$DOCKER" images "$IMAGE" --format '{{.Repository}}:{{.Tag}}  {{.Size}}'
