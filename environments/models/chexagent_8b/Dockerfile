# syntax=docker/dockerfile:1.7
# Generated from environments/models.yaml; do not edit by hand.
FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04@sha256:46cb48a4abfbc40c836fe57bc05a07101b6458fffc63bbdfd6a50db98c9358bd
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    git python3.10 python3.10-venv python3-pip ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN python3.10 -m venv /opt/medumm/venv
ENV PATH=/opt/medumm/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/cache/huggingface \
    MEDUMM_MODEL_ROOT=/models \
    MEDUMM_OUTPUT_ROOT=/outputs
COPY lock.txt /tmp/medumm-lock.txt
RUN python -m pip install --upgrade pip==25.1.1 \
    && python -m pip install --extra-index-url https://download.pytorch.org/whl/cu121 -r /tmp/medumm-lock.txt
WORKDIR /workspace
LABEL org.medumm.model=chexagent_8b \
      org.medumm.profile=internvl \
      org.medumm.contract-sha256=9230002d1ddcec5221f8ef5391ba05ecae462ad46a9dcb2f94745dbb1fc33f38
CMD ["python", "-m", "medumm.cli.main", "--help"]
