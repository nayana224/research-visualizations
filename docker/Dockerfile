FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PATH="/root/.local/bin:${PATH}" \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/workspace/.venv

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl git \
    build-essential pkg-config python3-dev \
    libcairo2-dev libpango1.0-dev \
    ffmpeg dvisvgm \
    texlive-latex-base texlive-latex-extra texlive-fonts-recommended \
    fonts-noto-cjk \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

WORKDIR /workspace
CMD ["bash"]
