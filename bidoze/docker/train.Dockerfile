# Training image: the data image plus the package and the harness.
#
# Built by scripts/build_train_image.sh. The data image arrives as a named build
# context; pisces-lite and its senpy backend are installed from PyPI:
#
#   --build-context data=docker-image://<data image pinned by digest>
#
# A public build can point `data` at a synthetic fixture instead.
# The `base` tag, not `-cudnn-runtime`: jax[cuda12] ships its own pinned
# nvidia-*-cu12 wheels (cuDNN, cuBLAS, cuSOLVER, NCCL and ptxas) and loads
# them by RPATH, ignoring /usr/local/cuda. A -cudnn-runtime base therefore
# adds ~3GB that nothing ever opens, and doesn't even carry ptxas, which XLA
# needs. Keeping CUDA on the pip side also ties its version to jax's rather
# than to whatever this tag froze.
FROM nvidia/cuda:12.6.2-base-ubuntu24.04
ARG JAX_VERSION=0.11.2
ARG PISCES_LITE_VERSION=4.1.0
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    PATH=/opt/venv/bin:$PATH

# No compiler: every wheel installed below is prebuilt, including arcascope-senpy
# (the one compiled dependency). senpy lives behind pisces-lite's [proc] extra,
# and proc is the data-prep path -- training reads a prebuilt feature cache, so
# [proc] is not installed here.
RUN apt-get update \
    && apt-get install -y --no-install-recommends python3.12 python3.12-venv \
    && rm -rf /var/lib/apt/lists/* \
    && python3.12 -m venv /opt/venv

# jax is pinned exactly: the jax[cuda12] extra drags the whole CUDA stack
# (cuDNN, cuBLAS, NCCL, ptxas) in as pinned wheels, so an unpinned jax silently
# changes the CUDA runtime under a rebuild. Bump this deliberately.
RUN pip install --no-cache-dir "pisces-lite==${PISCES_LITE_VERSION}" \
    && pip install --no-cache-dir "jax[cuda12]==${JAX_VERSION}"

WORKDIR /opt/bidoze
COPY pyproject.toml README.md ./
COPY src ./src
COPY config ./config
COPY scripts ./scripts
# [aws] as well as [train]: the harness uploads the finished run to S3, and
# boto3 is pure Python -- ~15MB, no compiler, no CUDA.
RUN pip install --no-cache-dir ".[train,aws]"

COPY --from=data /data/feature_cache ./feature_cache
COPY --from=data /data/manifest.jsonl ./build/manifest/manifest.jsonl

# The code half of the run fingerprint. There is no git and no .git dir in
# this image, so the commit has to be baked in at build time; build_train_image.sh
# marks it -dirty when the tree it built from had uncommitted changes.
ARG GIT_COMMIT=unknown
ENV FEATURE_CACHE_DIR=/opt/bidoze/feature_cache \
    BIDOZE_RESULTS_DIR=/opt/bidoze/results \
    BIDOZE_MANIFEST_PATH=/opt/bidoze/build/manifest/manifest.jsonl \
    GIT_COMMIT=${GIT_COMMIT}

ENTRYPOINT ["bash", "scripts/harness.sh"]
