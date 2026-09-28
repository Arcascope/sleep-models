# The Autofish feature cache as an immutable, content-addressed image layer.
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
# Both commands run from the autofish/ directory, as harness.sh does.
#
#   autofish manifest-local \
#       --root feature_cache/12Hz_log_PSD \
#       --output build/manifest/manifest.jsonl
#
#   docker buildx build \
#       -f docker/data.Dockerfile \
#       --build-context manifest=build/manifest \
#       --build-arg CACHE_PREFIX=12Hz_log_PSD \
#       -t autofish-data:12Hz_log_PSD-<digest[:12]> \
#       feature_cache/12Hz_log_PSD
#
# BuildKit reads autofish/docker/data.Dockerfile.dockerignore next to this file,
# so nothing has to be dropped inside the cache directory itself either.

# CACHE_PREFIX names the subdirectory training looks in. It is the same string
# as "cache_prefix" in config/inputs.json, and autofish train derives the path
# from that config, so a mismatch here surfaces as a missing-cache error rather
# than a silently empty run.
ARG CACHE_PREFIX=12Hz_log_PSD

FROM scratch

ARG CACHE_PREFIX
ARG SOURCE_DATE_EPOCH

# One COPY per dataset, ordered least-churning first. Each is its own layer, so
# re-processing weaver leaves the Foundry and SleepAccel layers byte-identical
# and ECR stores and ships them once.
#
# Note: under BuildKit a wildcard matching nothing is a silent no-op, not an
# error (verified: it emits an empty layer). A data image built from a
# half-populated cache therefore builds cleanly. Completeness is checked before
# the build, against the manifest, where the per-dataset counts are already
# enumerated.
COPY *_features.npz  /data/feature_cache/${CACHE_PREFIX}/

# Last, and in its own layer, so regenerating the manifest never perturbs the
# data layers' digests. A run reads this rather than trusting the image tag:
# the tag is a claim about the bytes, the manifest is a checkable record of them.
COPY --from=manifest manifest.jsonl /data/manifest.jsonl

LABEL org.opencontainers.image.title="autofish-data" \
      org.opencontainers.image.description="Autofish NUFFT feature cache, data only" \
      io.arcascope.autofish.cache_prefix="${CACHE_PREFIX}"
