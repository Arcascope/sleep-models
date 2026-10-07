# A feature cache as an immutable, content-addressed image layer, shared by
# every model family that reads the same cache prefix.
#
# This image is *only* a COPY --from source. It has no shell, no libc and no
# entrypoint, so it cannot be run; that is deliberate. It carries bytes and a
# manifest, nothing that could drift.
#
# Build context is the cache *prefix* directory, not the repo. The manifest
# arrives as a second, named context so that it never has to be written into
# the cache directory -- build_local_manifest hashes every regular file it
# finds, so a manifest stored alongside the data would change the very hash it
# records.
#
# deploy/scripts/build_data_image.sh runs both steps:
#
#   arcasleep manifest-local \
#       --root <cache>/12Hz_log_PSD_aligned \
#       --output build/manifest/12Hz_log_PSD_aligned/manifest.jsonl
#
#   docker buildx build \
#       -f deploy/docker/data.Dockerfile \
#       --build-context manifest=build/manifest/12Hz_log_PSD_aligned \
#       --build-arg CACHE_PREFIX=12Hz_log_PSD_aligned \
#       -t arcasleep-data:12Hz_log_PSD_aligned-<digest[:12]> \
#       <cache>/12Hz_log_PSD_aligned
#
# BuildKit reads data.Dockerfile.dockerignore next to this file,
# so nothing has to be dropped inside the cache directory itself either.

# CACHE_PREFIX names the subdirectory training looks in. It is the same string
# as "cache_prefix" in config/inputs.json, and `arcasleep <family> train` derives
# the path from that config, so a mismatch here surfaces as a missing-cache
# error rather than a silently empty run.
ARG CACHE_PREFIX=12Hz_log_PSD_aligned

FROM scratch

ARG CACHE_PREFIX
ARG SOURCE_DATE_EPOCH

# All cache NPZs go in one layer.
#
# Note: under BuildKit a wildcard matching nothing is a silent no-op, not an
# error. A data image built from a
# half-populated cache therefore builds cleanly. Completeness is checked before
# the build, against the manifest, where the per-dataset counts are already
# enumerated.
COPY *_features.npz  /data/feature_cache/${CACHE_PREFIX}/

# Last, and in its own layer, so regenerating the manifest never perturbs the
# data layers' digests. A run reads this rather than trusting the image tag:
# the tag is a claim about the bytes, the manifest is a checkable record of them.
COPY --from=manifest manifest.jsonl /data/manifest.jsonl

LABEL org.opencontainers.image.title="arcasleep-data" \
      org.opencontainers.image.description="Arcasleep NUFFT feature cache, data only" \
      io.arcascope.arcasleep.cache_prefix="${CACHE_PREFIX}"
