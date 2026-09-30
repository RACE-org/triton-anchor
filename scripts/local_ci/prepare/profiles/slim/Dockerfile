ARG BASE_IMAGE=ubuntu@sha256:cdb5fd928fced577cfecf12c8966e830fcdf42ee481fb0b91904eeddc2fe5eff
FROM ${BASE_IMAGE} AS toolchain

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential ca-certificates cmake curl git \
    libxml2-dev libz-dev libzstd-dev ninja-build pkg-config \
    python3 python3-dev python3-venv unzip xz-utils \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv /opt/venv

FROM toolchain AS foundation
RUN --mount=type=bind,source=wheels,target=/wheels,readonly \
    cd /wheels && sha256sum --check SHA256SUMS \
    && python -m pip install --no-cache-dir --no-compile --no-index --find-links=/wheels /wheels/*.whl \
    && python -m pip check \
    && python -I -c "import build,setuptools,wheel,pybind11,yaml,pytest,torch; assert torch.__version__ == '2.8.0+cpu'; assert torch.version.cuda is None"

RUN --mount=type=bind,source=codex,target=/build-codex,readonly \
    install -m 0755 /build-codex /usr/local/bin/codex \
    && codex --version

WORKDIR /workspace
