#!/bin/bash
# Train ONE LoRA adapter with LLaMA-Factory v0.9.5 (patched, see README "Training").
#   GPU=0 LF_DIR=/path/to/LLaMA-Factory bash training/train_adapter.sh training/configs/<route>.yaml
# The YAML holds every hyper-parameter; edit its three path fields first (model_name_or_path,
# dataset_dir, output_dir). Effective batch size is 8 in all shipped configs:
#   96 GB GPU  -> per_device_train_batch_size 2 / gradient_accumulation_steps 4 (as shipped)
#   24 GB GPU  -> per_device_train_batch_size 1 / gradient_accumulation_steps 8 (same gradient)
set -euo pipefail
CFG="${1:?usage: GPU=<n> LF_DIR=<LLaMA-Factory dir> train_adapter.sh <config.yaml>}"
GPU="${GPU:?set GPU=<card index>}"
LF_DIR="${LF_DIR:?set LF_DIR=<LLaMA-Factory checkout with lf_v095_tf5_compat.patch applied>}"
CFG="$(readlink -f "$CFG")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export DISABLE_VERSION_CHECK=1
cd "$LF_DIR"
CUDA_VISIBLE_DEVICES="$GPU" llamafactory-cli train "$CFG"
