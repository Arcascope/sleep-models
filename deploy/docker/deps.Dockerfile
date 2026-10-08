# Dependency base for every training and prep image: Python, pisces-lite and
# the CUDA build of jax. These are the slow, multi-GB layers; family images
# start FROM this one and add only code, config and data.
#
# Built by deploy/scripts/build_deps_image.sh, which tags it by the pinned
# versions and this file's digest, so the same inputs always give the same tag.
#
# The `base` tag, not `-cudnn-runtime`: jax[cuda12] ships its own pinned
# nvidia-*-cu12 wheels (cuDNN, cuBLAS, cuSOLVER, NCCL and ptxas) and loads
# them by RPATH, ignoring /usr/local/cuda. A -cudnn-runtime base therefore
# adds ~3GB that nothing ever opens, and doesn't even carry ptxas, which XLA
# needs. Keeping CUDA on the pip side also ties its version to jax's rather
# than to whatever this tag froze.
FROM nvidia/cuda:12.6.2-base-ubuntu24.04
# No defaults: build_deps_image.sh passes the versions pinned in pyproject.toml
# and its own JAX_VERSION.
ARG JAX_VERSION
ARG PISCES_LITE_VERSION
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    PYTHONUNBUFFERED=1 \
    PATH=/opt/venv/bin:$PATH

# No compiler: every wheel installed below is prebuilt, including
# arcascope-senpy (the one compiled dependency, behind pisces-lite's [proc]
# extra, which the prep image adds).
RUN apt-get update \
    && apt-get install -y --no-install-recommends python3.12 python3.12-venv \
    && rm -rf /var/lib/apt/lists/* \
    && python3.12 -m venv /opt/venv

# jax is pinned exactly: the jax[cuda12] extra drags the whole CUDA stack
# (cuDNN, cuBLAS, NCCL, ptxas) in as pinned wheels, so an unpinned jax silently
# changes the CUDA runtime under a rebuild. Bump this deliberately.
RUN pip install --no-cache-dir "pisces-lite==${PISCES_LITE_VERSION}" \
    && pip install --no-cache-dir "jax[cuda12]==${JAX_VERSION}"

LABEL org.opencontainers.image.title="arcasleep-deps" \
      io.arcascope.arcasleep.jax_version="${JAX_VERSION}" \
      io.arcascope.arcasleep.pisces_lite_version="${PISCES_LITE_VERSION}"
