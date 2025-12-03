ARG BASE_IMAGE=ghcr.io/ptb-mr/mrpro_py313:latest
FROM ${BASE_IMAGE} AS base

USER root
ARG DEBIAN_FRONTEND=noninteractive

WORKDIR /workdir
COPY reco_scripts/ reco_scripts/

COPY mr_recon.sh .
RUN chmod +x mr_recon.sh



