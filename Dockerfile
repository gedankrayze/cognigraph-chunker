# Stage 1: Build
FROM rust:1.93-bookworm AS builder

WORKDIR /app

# Copy workspace manifests
COPY Cargo.toml Cargo.lock ./
COPY packages/python/Cargo.toml packages/python/Cargo.toml

# Stub out sources so cargo fetch resolves with just manifests
RUN mkdir -p src packages/python/src benches && \
    echo 'fn main() {}' > src/main.rs && \
    echo '' > src/lib.rs && \
    echo 'fn main() {}' > benches/chunking.rs && \
    printf 'use pyo3::prelude::*;\n#[pymodule]\nfn cognigraph_chunker(_py: Python, _m: &Bound<PyModule>) -> PyResult<()> { Ok(()) }' \
      > packages/python/src/lib.rs && \
    cargo fetch

# Copy real source (overwrites stubs)
COPY src/ src/
COPY benches/ benches/

RUN cargo build --release --package cognigraph-chunker --bin cognigraph-chunker

# Download ONNX Runtime shared library for linux-x64
ARG ORT_VERSION=1.22.0
ADD https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-x64-${ORT_VERSION}.tgz /tmp/ort.tgz
RUN tar -xzf /tmp/ort.tgz -C /tmp && \
    mkdir -p /opt/onnxruntime && \
    cp /tmp/onnxruntime-linux-x64-${ORT_VERSION}/lib/libonnxruntime.so* /opt/onnxruntime/

# Stage 2: Runtime
FROM debian:bookworm-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates curl && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/target/release/cognigraph-chunker /usr/local/bin/cognigraph-chunker
COPY --from=builder /opt/onnxruntime/ /usr/local/lib/
RUN ldconfig

# Optional: bundle ONNX models for local embeddings
# COPY models/ /app/models/

# Railway, Render, Fly.io inject PORT env var
ENV PORT=3000

EXPOSE ${PORT}

# Environment variables for runtime configuration:
#   PORT                   - server port (default: 3000)
#   API_KEY                - Bearer token for authentication
#   NO_AUTH                - set to "1" to disable auth
#   CORS_ORIGINS           - comma-separated allowed origins
#   OPENAI_API_KEY         - for OpenAI embedding provider
#   CLOUDFLARE_AUTH_TOKEN  - for Cloudflare Workers AI embedding provider
#   CLOUDFLARE_ACCOUNT_ID  - Cloudflare account ID
#   CLOUDFLARE_AI_GATEWAY  - Cloudflare AI Gateway name (optional)
#   ORT_DYLIB_PATH         - path to libonnxruntime.so (default: system lib path)

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:${PORT}/api/v1/health || exit 1

ENTRYPOINT ["sh", "-c", \
  "exec cognigraph-chunker serve \
    --host 0.0.0.0 \
    --port ${PORT} \
    ${API_KEY:+--api-key \"$API_KEY\"} \
    ${NO_AUTH:+--no-auth} \
    ${CORS_ORIGINS:+--cors-origin \"$CORS_ORIGINS\"}"]
