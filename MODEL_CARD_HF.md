---
license: apache-2.0
base_model: Qwen/Qwen3.5-9B
tags:
  - medical
  - vision-language
  - qlora
  - peft
  - flare-2026
  - docker
language:
  - en
pipeline_tag: image-text-to-text
---

# FLARE26-Hexapus: submitted container for MICCAI FLARE 2026 Task 3 (2D)

This repository hosts the exact Docker image submitted by team **Hexapus** (MBZUAI BioMedIA) to
the testing phase of [FLARE 2026 Task 3, Multimodal Model for Medical Image Parsing](https://www.codabench.org/competitions/7151/).
Paper: *Multimodal Medical Image Parsing with Task-Specific Low-Rank Adapters* (MICCAI 2026 FLARE
Challenge proceedings). Code: **[CODE_REPO_URL]**.

| File | Size | MD5 |
| :--- | ---: | :--- |
| `hexapus.tar.gz` | 21,974,243,004 bytes | `466a96787cb69543b12cc5048880e624` |
| `hexapus.tar.gz.md5` | 49 bytes | checksum sidecar |

Image tag inside the tarball: `hexapus:latest` (image ID `101cdd2ee4c6`, 30.1 GB uncompressed).

## What is inside

One frozen 4-bit (NF4) **Qwen3.5-9B** backbone, a question-only **GLiClass** task router
([`MaiAShaaban/flare-gliclass-small-v1.0`](https://huggingface.co/MaiAShaaban/flare-gliclass-small-v1.0)),
seven **QLoRA adapters** (classification, multi-label classification, counting, detection,
regression, report generation, general fallback) and the inference code. Everything runs offline.

| Route | Adapter | Image budget | LoRA r/α |
| :--- | :--- | :--- | :--- |
| Classification | `qwen35_cls_rebal_full_2048_s1337` | 2048² | 32/64 |
| Multi-label classification | `qwen35_multilabel_noavg_medcot_full_2048_s1337` | 2048² | 64/128 |
| Counting | `qwen35_counting_full_2048_s1337` | 2048² | 8/16 |
| Detection | `qwen35_detection_full_2048_s1337` | 2048² | 8/16 |
| Regression | `qwen35_regmix_A5_dermacls_AB_2048_s1337` | 2048² | 8/16 |
| Report generation | `qwen35_reportgen_AB_768_s1337` | 768² | 8/16 |
| General (fallback, router confidence < 0.3) | `qwen35_general_AB_768_s1337` | 768² | 8/16 |

Input is an image (up to four) plus a question; output is a text answer whose format follows the
task (label, label set, integer, nested `[x_min, y_min, x_max, y_max]` boxes, number, or free
text). Decoding is greedy with EOS pinned and thinking mode disabled.

## Run

```bash
huggingface-cli download TahaSabir0/FLARE26-Hexapus hexapus.tar.gz --local-dir .
md5sum -c hexapus.tar.gz.md5
docker load -i hexapus.tar.gz
docker run --gpus "device=0" -m 48G --name hexapus --rm \
  -v $PWD/FLARE_Test/:/workspace/inputs/ -v $PWD/hexapus_outputs/:/workspace/outputs/ \
  hexapus:latest /bin/bash -c "sh predict.sh"
```

Input mount: one folder per dataset, each with `imagesTr/` and a `*_questions_*.json` in the
FLARE 2026 format. Output: `/workspace/outputs/predictions.json`. Measured on an RTX A5000:
3.1 s per case including model loading; two independent runs produce byte-identical outputs.

## Results

Organizer-scored hidden validation (pooled per task, 4,335 cases): classification balanced
accuracy **0.8585** (n=2,610), multi-label F1 **0.5310** (n=923), detection F1 at IoU>0.5
**0.9733** (n=600), regression MAE **18.53** (n=202). Counting and report generation have no
cases in that set. Internal held-out results are in the paper.

## Training data and intended use

Adapters were trained on the [FLARE-MedFM/FLARE-MLLM-2D](https://huggingface.co/datasets/FLARE-MedFM/FLARE-MLLM-2D)
training and public-validation data (2D: clinical photography, dermatology, mammography,
microscopy, retinography, ultrasound, X-ray). This is a challenge submission for research use
only. It is not a medical device and must not be used for clinical decision making.

## Licenses

The image redistributes Qwen3.5-9B (Apache 2.0, Alibaba Cloud), the GLiClass router from ME-VLIP,
and our adapters and code (Apache 2.0). Dataset licenses are those of the FLARE 2026 organizers.

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
