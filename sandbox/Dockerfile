# Sandbox image. Build context is allowlisted by .dockerignore (only wsp/ and sandbox/).
# TODO(VM): pin the base image by digest once built on the VM (python:3.11-slim@sha256:...).
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    HF_HUB_DISABLE_PROGRESS_BARS=1 \
    TRANSFORMERS_VERBOSITY=error \
    PYTHONPATH=/opt

# Everything is baked in here, at build time (the container itself has no network).
RUN pip install --no-cache-dir \
        torch==2.10.0 transformers==5.2.0 safetensors==0.7.0 numpy==2.4.2 pyarrow==23.0.1 \
        accelerate==1.15.0

COPY wsp/ /opt/wsp/
COPY sandbox/executor.py /opt/executor.py

# Non-root, no home directory, no shell login.
RUN useradd --uid 10001 --no-create-home --home-dir /nonexistent --shell /usr/sbin/nologin sandbox \
 && mkdir -p /work && chown 10001:10001 /work

USER 10001:10001
WORKDIR /work
# Weights are mounted read-only at /model by the harness (neutral names: shard_00.safetensors ...).
ENTRYPOINT ["python", "-u", "/opt/executor.py"]
