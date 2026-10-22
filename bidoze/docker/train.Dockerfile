# Training image: the data image plus the package and the harness.
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

# The feature cache comes first: it is the largest layer and changes least, so a
# code or config change rebuilds and pushes only the small layers above it.
WORKDIR /opt/bidoze
COPY --from=data /data/feature_cache ./feature_cache
COPY --from=data /data/manifest.jsonl ./build/manifest/manifest.jsonl

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
# boto3 is pure Python -- ~15MB, no compiler, no CUDA. [publish] is for the release
# bundle the harness writes after a release's final phase (safetensors, pandas).
RUN SETUPTOOLS_SCM_PRETEND_VERSION=${PACKAGE_VERSION} \
    pip install --no-cache-dir ".[bidoze,train,aws,publish]"

WORKDIR /opt/bidoze
COPY bidoze/config ./config
COPY bidoze/scripts ./scripts
COPY bidoze/experiments ./experiments

# The code half of the run fingerprint. There is no git and no .git dir in
# this image, so the commit has to be baked in at build time; build_train_image.sh
# marks it -dirty when the tree it built from had uncommitted changes.
ARG GIT_COMMIT=unknown
ENV FEATURE_CACHE_DIR=/opt/bidoze/feature_cache \
    BIDOZE_RESULTS_DIR=/opt/bidoze/results \
    BIDOZE_MANIFEST_PATH=/opt/bidoze/build/manifest/manifest.jsonl \
    GIT_COMMIT=${GIT_COMMIT}

ENTRYPOINT ["bash", "scripts/harness.sh"]
