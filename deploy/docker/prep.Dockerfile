# Feature-cache build image: the dependency image plus pisces-lite's [proc]
# extra (the senpy NUFFT), the AWS CLI and every family's processing config.
#
# deploy/scripts/prep_harness.sh is the entrypoint: it syncs the named datasets
# from S3, runs `arcasleep process` on the GPU and syncs the cache back to S3,
# where a data image is built from it.
#
#   docker buildx build -f deploy/docker/prep.Dockerfile \
#       --build-arg DEPS_IMAGE=<deps image> .
ARG DEPS_IMAGE
FROM ${DEPS_IMAGE}

# The image has no .git, so the version setuptools-scm would read from a tag is
# supplied instead. awscli syncs whole dataset directories, which boto3 alone
# does not.
ARG PACKAGE_VERSION=0.0.0
WORKDIR /opt/arcasleep
COPY pyproject.toml README.md LICENSE MANIFEST.in ./
COPY core/src ./core/src
COPY bidoze/src ./bidoze/src
COPY unet/src ./unet/src
RUN SETUPTOOLS_SCM_PRETEND_VERSION=${PACKAGE_VERSION} \
    pip install --no-cache-dir ".[prep]" "pisces-lite[jax]" awscli

WORKDIR /opt/prep
COPY bidoze/config ./bidoze/config
COPY unet/config ./unet/config
COPY deploy/scripts/prep_harness.sh ./prep_harness.sh

ARG GIT_COMMIT=unknown
ENV GIT_COMMIT=${GIT_COMMIT} \
    PREP_WORK_DIR=/opt/prep/work

ENTRYPOINT ["bash", "prep_harness.sh"]
