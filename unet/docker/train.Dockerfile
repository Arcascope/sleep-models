# UNet training image: the data image plus the package and the harness.
#
# The same build as bidoze/docker/train.Dockerfile; only the family differs.
#
# Built by deploy/scripts/build_train_image.sh on the dependency image
# (deploy/docker/deps.Dockerfile), which carries Python, pisces-lite and the
# CUDA build of jax. The data image arrives as a named build context:
#
#   --build-arg DEPS_IMAGE=<deps image>
#   --build-context data=docker-image://<data image pinned by digest>
#
# A public build can point `data` at a synthetic fixture instead.
ARG DEPS_IMAGE
FROM ${DEPS_IMAGE}

# The build context is the repository root: the arcasleep distribution holds
# the shared code and every model family. The image has no .git, so the
# version setuptools-scm would read from a tag is supplied instead.
ARG PACKAGE_VERSION=0.0.0
WORKDIR /opt/arcasleep
COPY pyproject.toml README.md LICENSE MANIFEST.in ./
COPY core/src ./core/src
COPY bidoze/src ./bidoze/src
COPY unet/src ./unet/src
# [aws] as well as [train]: the harness uploads the finished run to S3, and
# boto3 is pure Python -- ~15MB, no compiler, no CUDA.
RUN SETUPTOOLS_SCM_PRETEND_VERSION=${PACKAGE_VERSION} \
    pip install --no-cache-dir ".[unet,train,aws]"

WORKDIR /opt/unet
COPY unet/config ./config
COPY unet/scripts ./scripts

COPY --from=data /data/feature_cache ./feature_cache
COPY --from=data /data/manifest.jsonl ./build/manifest/manifest.jsonl

# The code half of the run fingerprint. There is no git and no .git dir in
# this image, so the commit has to be baked in at build time; build_train_image.sh
# marks it -dirty when the tree it built from had uncommitted changes.
ARG GIT_COMMIT=unknown
# The shared provenance code reads the manifest from BIDOZE_MANIFEST_PATH,
# whichever family is training.
ENV FEATURE_CACHE_DIR=/opt/unet/feature_cache \
    UNET_RESULTS_DIR=/opt/unet/results \
    BIDOZE_MANIFEST_PATH=/opt/unet/build/manifest/manifest.jsonl \
    GIT_COMMIT=${GIT_COMMIT}

ENTRYPOINT ["bash", "scripts/harness.sh"]
