# FLARE-2026 Task-3 -- Qwen3.5-9B routed submission image (v5 stack).
# Build with the staging context assembled by build_docker.sh (models/ + helpers beside this file).
#
# Base = cu128 to MATCH the verified inference env (torch 2.11.0+cu128 / torchvision 0.26.0).
# Eval GPU A6000 Ada (sm_89) is fully supported by cu128 wheels.
FROM pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TOKENIZERS_PARALLELISM=false \
    FLARE_SCRIPTS_DIR=/app \
    MODELS_DIR=/app/models

# EVERY ADAPTER IS PINNED BY EXPLICIT NAME, one build-arg per route (the ADAPTER_TAG
# name-collision lesson -- never a shared tag template).
# Defaults = the submitted route set (see the paper). Override any single route at
# build time; names must match the dirs staged into /app/models/qwen35_final/ by build_docker.sh.
# NOTE the RESOLUTION IS PART OF THE ROUTE: if you swap an adapter to one trained at a different
# pixel cap, set the matching <ROUTE>_PIXELS env too (see inference.py TASK_CONFIG).
ARG CLS_ADAPTER=qwen35_cls_rebal_full_2048_s1337
ARG MULTILABEL_ADAPTER=qwen35_multilabel_noavg_medcot_full_2048_s1337
ARG DETECTION_ADAPTER=qwen35_detection_full_2048_s1337
ARG COUNTING_ADAPTER=qwen35_counting_full_2048_s1337
ARG REGRESSION_ADAPTER=qwen35_regmix_A5_dermacls_AB_2048_s1337
ARG REPORTGEN_ADAPTER=qwen35_reportgen_AB_768_s1337
ARG GENERAL_ADAPTER=qwen35_general_AB_768_s1337
ENV CLS_ADAPTER=${CLS_ADAPTER}
ENV MULTILABEL_ADAPTER=${MULTILABEL_ADAPTER}
ENV DETECTION_ADAPTER=${DETECTION_ADAPTER}
ENV COUNTING_ADAPTER=${COUNTING_ADAPTER}
ENV REGRESSION_ADAPTER=${REGRESSION_ADAPTER}
ENV REPORTGEN_ADAPTER=${REPORTGEN_ADAPTER}
ENV GENERAL_ADAPTER=${GENERAL_ADAPTER}

# OpenGL/glib runtime libs that Pillow-style image loads need (no compilers -- keep image small).
RUN echo 'APT::Sandbox::User "root";' > /etc/apt/apt.conf.d/no-sandbox \
    && apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

WORKDIR /app

# gliclass FIRST (it pulls its own transformers); the pins in requirements.txt then land
# transformers==5.16.1 etc. on top. gliclass must be ==0.1.20: 0.1.11 does NOT load under
# transformers v5 (pad_token_id + tie_weights API breaks); 0.1.20 verified routing-identical
# to the 4.57.1 reference (probe 2026-09-05).
COPY requirements.txt /tmp/requirements.txt
# PEP 668: base image (Ubuntu 24.04) marks system python externally-managed; /usr/bin/python IS the
# torch 2.11.0+cu128 interpreter (no conda in this image) and is what predict.sh invokes.
ENV PIP_BREAK_SYSTEM_PACKAGES=1
RUN pip install --no-cache-dir gliclass==0.1.20 \
    && pip install --no-cache-dir -r /tmp/requirements.txt

# inference + the vendored FLARE helper modules (FLARE_SCRIPTS_DIR=/app picks them up) + entry
COPY inference.py \
     vlm_prompt.py \
     prompt_variants.py \
     format_templates.py \
     predict.sh \
     /app/

# pre-downloaded models (staged into the build context by build_docker.sh):
#   Qwen3.5-9B base  +  qwen35_final/<7 adapters>  +  FLARE-gliclass-small-v1.0 router
COPY models /app/models

RUN chmod +x /app/predict.sh

# organizer runs: /bin/bash -c "sh predict.sh"
CMD ["/bin/bash", "-c", "sh predict.sh"]
