FROM nvidia/cuda:12.6.3-runtime-ubuntu24.04
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv git curl ca-certificates gnupg && \
    rm -rf /var/lib/apt/lists/*
# ubuntu 24.04 has no modern node in its main repos — nodesource BEFORE installing nodejs
# (the nodesource package bundles npm).
RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - && \
    apt-get install -y --no-install-recommends nodejs && \
    rm -rf /var/lib/apt/lists/*
# UV_INSTALL_DIR=/usr/local/bin: the default ($HOME/.local/bin) lands in /root, which
# sandbox_cmd mounts the per-run HOME over — a symlink from /usr/local/bin dangles at runtime
# and every leg re-discovers "uv missing" and self-installs.
RUN curl -LsSf https://astral.sh/uv/install.sh | UV_INSTALL_DIR=/usr/local/bin UV_NO_MODIFY_PATH=1 sh
# The 3 CLIs, versions pinned at build time
# PINNED, because the CLI is the experiment's subject: an unpinned rebuild changes it silently.
# A patch release is enough — resume-flag parsing has changed between adjacent versions, which
# breaks a leg the driver cannot then land. Bump deliberately, one at a time, and smoke after.
RUN npm install -g @anthropic-ai/claude-code@2.1.221 @openai/codex@0.146.0 @google/gemini-cli@0.53.1
# Kimi Code is not on npm. KIMI_INSTALL_DIR=/usr/local: the default ($HOME/.kimi-code) would be
# hidden at runtime — sandbox_cmd mounts the per-run HOME over /root.
RUN curl -fsSL https://code.kimi.com/kimi-code/install.sh | KIMI_INSTALL_DIR=/usr/local KIMI_NO_MODIFY_PATH=1 bash -s -- --version 0.32.0
# The workspace is bind-mounted with the HOST owner's uid while the container runs as root, so
# git refuses every command with "dubious ownership" (CVE-2022-24765 guard). System-level config:
# /etc is baked into the image, unlike /root which is mounted over per run. Without this, every
# leg of every campaign burns turns rediscovering and re-fixing it.
RUN git config --system --add safe.directory /workspace
WORKDIR /workspace
ENTRYPOINT ["/bin/bash", "-lc"]
