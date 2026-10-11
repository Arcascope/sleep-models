ARG BASE_IMAGE
FROM ${BASE_IMAGE}
COPY core/src /opt/arcasleep/core/src
COPY bidoze/src /opt/arcasleep/bidoze/src
COPY unet/src /opt/arcasleep/unet/src
COPY bidoze/scripts /opt/bidoze/scripts
COPY bidoze/experiments/boat_evidence /opt/bidoze/experiments/boat_evidence
COPY --from=evidence / /opt/boat-evidence-data/
ARG GIT_COMMIT
ENV GIT_COMMIT=${GIT_COMMIT} BIDOZE_EXPERIMENT=experiments/boat_evidence
WORKDIR /opt/bidoze
