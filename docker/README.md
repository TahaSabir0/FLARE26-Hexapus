# Submission container

The FLARE 2026 Task 3 testing-phase image (`hexapus:latest`). Organizer contract:

```bash
docker run --gpus "device=0" -m 48G --name hexapus --rm \
  -v $PWD/FLARE_Test/:/workspace/inputs/ -v $PWD/hexapus_outputs/:/workspace/outputs/ \
  hexapus:latest /bin/bash -c "sh predict.sh"
```

Input mount: dataset folders, each with `imagesTr/` and `*_questions_*.json`. Output mount:
`predictions.json` (all input fields preserved, `Answer` filled).

**Prebuilt image:** [`TahaSabir0/FLARE26-Hexapus`](https://huggingface.co/TahaSabir0/FLARE26-Hexapus)
(`hexapus.tar.gz`, MD5 `466a96787cb69543b12cc5048880e624`), then `docker load -i hexapus.tar.gz`.

**Build it yourself:** place `Qwen3.5-9B/`, `FLARE-gliclass-small-v1.0/` and
`qwen35_final/<the seven adapters>/` under `../models/` and run `bash build_docker.sh`
(needs ~25 GB of staging space under `$TMPDIR`). Adapters are pinned by explicit name, one
build-arg per route; a swapped adapter trained at a different pixel cap also needs the matching
`<ROUTE>_PIXELS` environment variable at run time, because the image budget is part of the route.

**Design notes** (details in the header of `inference.py`):

- base image `pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime`; `gliclass==0.1.20` installed first,
  then `requirements.txt` re-pins `transformers==5.16.1`;
- one 4-bit NF4 backbone, seven LoRA adapters loaded once, `set_adapter` per case;
- prompt rendered with `enable_thinking=False`; byte-identical to the LLaMA-Factory `qwen3_5`
  training render (checked by `../scripts/template_gate.py`);
- greedy decoding, EOS pinned at every `generate` call (Qwen3.5-9B ships no `generation_config.json`);
- per-route image cap applied after routing: 2048² for classification, multi-label, counting,
  detection and regression; 768² for report generation and the general adapter;
- the build script dereferences symlinks and fails on any dangling link, so a broken image cannot
  build silently.
