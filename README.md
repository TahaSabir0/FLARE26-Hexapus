# Multimodal Medical Image Parsing with Task-Specific Low-Rank Adapters

This repository is the official implementation of *Multimodal Medical Image Parsing with
Task-Specific Low-Rank Adapters* (FLARE 2026 Task 3, team **Hexapus**, MBZUAI BioMedIA).

One frozen 4-bit **Qwen3.5-9B** backbone answers every question. A question-only **GLiClass**
router picks one of six task-specific **QLoRA adapters** (classification, multi-label
classification, counting, detection, regression, report generation) or a general fallback
adapter. Each route fixes its own adapter, prompt variant and image-area budget.

The submitted container is public: [`TahaSabir0/FLARE26-Hexapus`](https://huggingface.co/TahaSabir0/FLARE26-Hexapus)
(`hexapus.tar.gz`, 21.97 GB, MD5 `466a96787cb69543b12cc5048880e624`). It contains the base model,
the seven adapters, the router and this inference code; see [Inference](#inference).

## Environments and Requirements

| Component | Setting |
| :--- | :--- |
| System | Ubuntu 22.04.5 LTS (training, A5000 workstation) / Ubuntu 25.10 (training, Blackwell node) |
| Programming language | Python 3.12 |
| Framework | PyTorch 2.11.0 + CUDA 12.8 |
| Dependencies | transformers 5.16.1, peft 0.18.1, bitsandbytes 0.50.2, accelerate 1.14.0, gliclass 0.1.20 |
| Training library | LLaMA-Factory 0.9.5 (+ `training/lf_v095_tf5_compat.patch`) |
| Training GPUs | 1x NVIDIA RTX PRO 6000 Blackwell 96 GB (2048² adapters) or 1x RTX A5000 24 GB (768² adapters); one GPU per run |
| Inference GPU | one 24 GB+ card (the challenge evaluates on an RTX A6000 Ada 48 GB); 4-bit inference needs ~8 GB at 768² |

Install the inference/evaluation stack:

```bash
conda create -n hexapus python=3.12 -y && conda activate hexapus
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
pip install gliclass==0.1.20            # first: it pins its own transformers
pip install -r requirements.txt         # then re-pins transformers==5.16.1 etc. on top
```

Training additionally needs LLaMA-Factory v0.9.5 with one small compatibility patch for
transformers 5.16 (see [Training](#training)).

## Dataset

- **Source:** [FLARE-MedFM/FLARE-MLLM-2D](https://huggingface.co/datasets/FLARE-MedFM/FLARE-MLLM-2D)
  (19 datasets, 8 modalities, 50,996 images, 58,112 question-answer pairs; 2D track).
- **Structure after download** (each dataset folder holds `imagesTr/` and a `*_questions_*.json`):

```
FLARE-MLLM-2D/
├── training/<Modality>/<dataset>/
├── validation-public/<Modality>/<dataset>/
└── validation-hidden/<Modality>/<dataset>/      # answers withheld
```

- **Download:**

```bash
huggingface-cli download FLARE-MedFM/FLARE-MLLM-2D --repo-type dataset --local-dir ./FLARE-MLLM-2D
find FLARE-MLLM-2D -name "*.zip" -exec unzip -o "{}" -d "$(dirname "{}")" \;
```

## Preprocessing

Data cleaning and the internal splits are built once from `training/` and then frozen:

1. **Audit** the raw JSON for annotation bugs (invalid boxes, duplicate rows with conflicting
   labels, endoscopy option-letter errors, an extreme regression outlier). 73 rows are flagged.
2. **Split** at image level (all questions on an image stay together), stratified by
   dataset × task, seed 1337: `true_test` (10%, internal holdout), `A` and `B` (45% each,
   development halves). Instance detection (unscored) and endoscopy are excluded from training.
3. **Export** a ShareGPT deck per route: multi-image rows are flattened to K=1 (one row per
   image, same question and answer), images get `Image-i:` markers, and the classification deck
   applies **K2a rebalancing** (a dataset-answer group with n < 250 examples is oversampled to
   min(6n, 250) rows, deterministically). The regression deck is all regression rows plus a
   5,000-row capped slice of `bcn20000` dermatology classification rows (the "donor mixture").
   Images are only ever resized by an aspect-preserving pixel-area cap (2048² or 768², per route);
   no augmentation, no normalization, no cropping.

```bash
python data/audit_data.py  --data_root FLARE-MLLM-2D --out_dir audit --check-leakage
python data/build_splits.py --data_root FLARE-MLLM-2D --out_dir internal_splits_v2 --seed 1337 \
       --audit_issues audit/audit_issues.csv --leaked_images audit/leaked_images.json
SPLITS=$PWD/internal_splits_v2 bash training/export_deck.sh cls_rebal "A B"                               # -> flare_final_cls_rebal (A∪B)
SPLITS=$PWD/internal_splits_v2 bash training/export_deck.sh cls_rebal "A B true_test validation-public" _full   # -> flare_final_cls_rebal_full
# routes: cls_rebal | multilabel_noavg | counting | detection | regmix_A5 | reportgen | general
```

At inference the same cap is applied after routing; up to four images per case are kept (evenly
spaced if more are supplied). No labels, task types or modality fields are used at inference.

## Training

Each adapter is one LLaMA-Factory QLoRA run. The seven submitted configurations are in
`training/configs/` (paths to edit are marked `/path/to/...`).

```bash
git clone --branch v0.9.5 https://github.com/hiyouga/LLaMA-Factory && cd LLaMA-Factory
git apply ../training/lf_v095_tf5_compat.patch && pip install -e . --no-deps && cd ..
GPU=0 LF_DIR=$PWD/LLaMA-Factory bash training/train_adapter.sh training/configs/qwen35_cls_rebal_full_2048_s1337.yaml
```

**Shared recipe**

| Component | Setting |
| :--- | :--- |
| Base model | Qwen3.5-9B, 4-bit NF4 (bitsandbytes, no double quantization), frozen |
| Method | QLoRA supervised fine-tuning, loss on answer tokens only |
| LoRA target | all linear layers of the language model (`lora_target: all`) |
| LoRA rank / alpha | 8 / 16 (default); 32 / 64 classification; 64 / 128 multi-label |
| Epochs | 3 |
| Effective batch size | 8 (2 × 4 on 96 GB; 1 × 8 on 24 GB) |
| Optimizer / LR | AdamW, 2e-4, linear decay, ~3% warm-up |
| Sequence length | 8,192 tokens |
| Precision | BF16 compute, gradient checkpointing |
| Chat template | `qwen3_5`, thinking disabled |
| Seed | 1337 |

**Per-route configuration (as submitted)**

| Route | Adapter | Training data | Pixels | r/α | Prompt |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Classification | `qwen35_cls_rebal_full_2048_s1337` | all labeled data, K2a | 2048² | 32/64 | bare |
| Multi-label | `qwen35_multilabel_noavg_medcot_full_2048_s1337` | all labeled data | 2048² | 64/128 | medical CoT |
| Counting | `qwen35_counting_full_2048_s1337` | all labeled data | 2048² | 8/16 | counting CoT |
| Detection | `qwen35_detection_full_2048_s1337` | all labeled data | 2048² | 8/16 | bare |
| Regression | `qwen35_regmix_A5_dermacls_AB_2048_s1337` | A∪B regression + 5k bcn20000 | 2048² | 8/16 | bare |
| Report generation | `qwen35_reportgen_AB_768_s1337` | A∪B | 768² | 8/16 | bare |
| General (fallback) | `qwen35_general_AB_768_s1337` | A∪B, all tasks | 768² | 8/16 | bare |

"All labeled data" = A ∪ B ∪ true_test ∪ validation-public. The held-out numbers in the paper come
from the A∪B siblings of these adapters, never from the all-data ones.

**Router.** The GLiClass task classifier
[`MaiAShaaban/flare-gliclass-small-v1.0`](https://huggingface.co/MaiAShaaban/flare-gliclass-small-v1.0)
(DeBERTa-v3-small encoder) is reused without retraining. It reads the question text only;
confidence below 0.3 falls back to the general adapter.

Trained models: the seven adapters, the base model and the router are packaged in the public
container (link above). Train and inference resolution must match per route.

Before evaluating a new backbone or template, run the byte-identity gate that compares the
LLaMA-Factory training render with the inference prompt: `python scripts/template_gate.py`.

## Inference

1. **From this repository** (base model, adapters and router under `models/`, layout as in
   `docker/build_docker.sh`):

```bash
export FLARE_SCRIPTS_DIR=$PWD/scripts MODELS_DIR=$PWD/models
python docker/inference.py --base_dataset_path <path to test dataset> \
       --output_dir <output folder> --output_filename predictions.json --max_new_tokens 512 --device cuda:0
```

Decoding is greedy (`do_sample=False`), EOS pinned, thinking mode off, 512 new tokens max. Output
is one `predictions.json` with every input field preserved and `Answer` filled.

2. **Docker** (the organizer contract; `hexapus.tar.gz` from the Hugging Face link above):

```bash
docker load -i hexapus.tar.gz
docker run --gpus "device=0" -m 48G --name hexapus --rm \
  -v $PWD/FLARE_Test/:/workspace/inputs/ -v $PWD/hexapus_outputs/:/workspace/outputs/ \
  hexapus:latest /bin/bash -c "sh predict.sh"
```

Rebuild the image yourself with `bash docker/build_docker.sh` after placing `models/Qwen3.5-9B`,
`models/FLARE-gliclass-small-v1.0` and `models/qwen35_final/<adapter>` next to the repo.

3. **Single adapter, no router** (the evaluation path used for the paper's per-task tables):

```bash
PYTHONPATH=scripts python evaluation/eval_adapter.py --base_model models/Qwen3.5-9B \
  --adapter_path saves/qwen35_regmix_A5_dermacls_AB_2048_s1337 --data_path internal_splits_v2/true_test.jsonl \
  --only_task regression --prompt_variant none --image_max_pixels 4194304 --output_file preds_regression.json
```

## Evaluation

Official metrics (balanced accuracy, micro-F1, detection F1 at IoU > 0.5, MAE) with the
organizers' scoring rules, grouped per task and per dataset:

```bash
python evaluation/evaluation.py --base_dataset_path FLARE-MLLM-2D --prediction_file predictions.json \
       --output_dir eval --output_filename metrics.json
python evaluation/score_single.py --pred_file preds_regression.json --out_file metrics_regression.json
```

`score_single.py` reports the per-dataset scores and their unweighted (macro) mean; the paper
reports the macro. Report generation is scored offline with CRIMSON:

```bash
CRIMSON_MODEL=rajpurkarlab/medgemma-4b-it-crimson python evaluation/score_crimson.py \
       --pred preds_reportgen.jsonl --out metrics_reportgen.json
```

## Results

**Official hidden validation** (organizers' run of the submitted container; pooled per task,
4,335 cases; counting and report generation have no cases in this set):

| Task | Metric | n | Score |
| :--- | :--- | ---: | ---: |
| Classification | Balanced accuracy ↑ | 2,610 | 0.8585 |
| Multi-label classification | F1 ↑ | 923 | 0.5310 |
| Detection | F1 @ IoU > 0.5 ↑ | 600 | 0.9733 |
| Regression | MAE ↓ | 202 | 18.53 |

**Internal holdout** (`true_test`, image-disjoint; A∪B adapters at the submitted route
resolution; dataset-macro where a task has several datasets):

| Task | Metric | Datasets | Score |
| :--- | :--- | ---: | ---: |
| Classification | Balanced accuracy ↑ | 10 | 0.7791 |
| Multi-label classification | F1 ↑ | 2 | 0.6286 |
| Counting | MAE ↓ | 1 | 11.73 |
| Detection | F1 @ IoU > 0.5 ↑ | 3 | 0.8914 |
| Regression | MAE ↓ | 2 | 9.78 |
| Report generation | CRIMSON ↑ | 1 | 0.8204 |

Per-dataset numbers, the general-versus-specialist comparison and the 768² versus 2048²
comparison are in the paper. Testing-phase results will be added when released.

**Efficiency** (final image, 32-case gauntlet across all six tasks): 3.1 s/case including model
loading; image 30.1 GB (limit 35 GB); two independent runs byte-identical.

## Citation

```bibtex
@inproceedings{sabir2026hexapus,
  title     = {Multimodal Medical Image Parsing with Task-Specific Low-Rank Adapters},
  author    = {Sabir, Taha and Montero Molina, Alberto and Ly, Thinh Quang and Dmitrenko, Veronika
               and Saleem, Tausifa Jan and Yaqub, Mohammad},
  booktitle = {MICCAI 2026 FLARE Challenge},
  year      = {2026}
}
```

## Contributing

This project is licensed under the Apache License 2.0; see [LICENSE](LICENSE). Issues and pull
requests are welcome.

## Acknowledgement

We thank the FLARE 2026 organizers and the contributors of the public datasets, and the
maintainers of [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory),
[GLiClass](https://github.com/Knowledgator/GLiClass) and [CRIMSON](https://arxiv.org/abs/2603.06183).
