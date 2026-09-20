#!/bin/bash
# Entry point invoked by the organizer:
#   docker run --gpus "device=1" -m 48G --rm \
#     -v $PWD/FLARE_Test/:/workspace/inputs/ -v $PWD/teamname_outputs/:/workspace/outputs/ \
#     teamname:latest /bin/bash -c "sh predict.sh"
# Input  mount : /workspace/inputs   (dataset folders, each with imagesTr/ + *_questions_*.json)
# Output mount : /workspace/outputs  (we write predictions.json here)
set -e
python /app/inference.py \
    --base_dataset_path /workspace/inputs \
    --output_dir /workspace/outputs \
    --output_filename predictions.json \
    --max_new_tokens 512 \
    --device cuda:0
