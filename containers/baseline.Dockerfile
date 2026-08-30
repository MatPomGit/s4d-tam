FROM python:3.12.11-slim-bookworm@sha256:27f90d79cc85e9b7b2560063ef44fa0e9eaae7a7c3f5a9f74563065c5477cc24
WORKDIR /opt/s4dtam
COPY containers/requirements.lock /tmp/requirements.lock
RUN python -m pip install --no-cache-dir --require-hashes -r /tmp/requirements.lock
COPY . .
RUN python -m pip install --no-cache-dir --no-deps .
LABEL org.opencontainers.image.version="0.1.0" org.opencontainers.image.licenses="Apache-2.0" io.s4dtam.role="baseline"
ENTRYPOINT ["s4dtam-bench"]
