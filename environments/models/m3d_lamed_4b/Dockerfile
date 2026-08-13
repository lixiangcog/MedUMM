# syntax=docker/dockerfile:1.7
# Generated from environments/models.yaml; do not edit by hand.
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04@sha256:f6913f3c02f297877f6859d12ff330043c0be668fdad86868c29a239a5a82151
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
    && python -m pip install --extra-index-url https://download.pytorch.org/whl/cu118 -r /tmp/medumm-lock.txt
WORKDIR /workspace
LABEL org.medumm.model=m3d_lamed_4b \
      org.medumm.profile=m3d \
      org.medumm.contract-sha256=b038973a3715580b59eed607a46322e38128f9d3d5461613e007fa10aa9852f8
CMD ["python", "-m", "medumm.cli.main", "--help"]
