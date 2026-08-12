# PROVISIONING — the machine a campaign runs on

Cloud-agnostic. A campaign needs one box, sized so that a sandboxed agent can run at full tilt while
the host still has room to poll the proxy and write artifacts.

## Sizing

| Resource | Recommendation | Why |
| --- | --- | --- |
| vCPU | ≥ 16 | Each sandbox takes 12 by default (`sandbox_cpus`); the host needs the rest. |
| RAM | ≥ 64 GB | Sandbox default is 12 GB, campaigns often raise it to 48 for GPU work. |
| Disk | **≥ 200 GB** | Every leg builds its own venv (torch alone is multi-GB), keeps a prediction set, and may commit model weights. A campaign that fills the disk does not fail cleanly — writes start failing inside agents and containers, so budget generously and watch it. |
| GPU | one 24 GB card (L4 / A10 class) if `gpu: true` | 24 GB and bf16 support are what a modern time-series model expects. A 16 GB card without bf16 works but silently changes what agents can attempt. |

State the actual shape in the campaign's `hardware:` field — it is interpolated into the agent's
constitution, and it is how the agent decides how large a job to launch.

**Preemptible / spot instances distort the experiment.** A preempted leg resumes with a fresh wall
clock, so it quietly receives more time than its envelope allows. Use on-demand for real campaigns.

## Install

```bash
# Docker + compose v2, then add yourself to the docker group (log out and back in after)
sudo usermod -aG docker "$USER"

# NVIDIA container toolkit, for gpu: true campaigns
sudo apt-get install -y nvidia-container-toolkit && sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# uv (Python 3.12+) and git
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Prefer a base image with the NVIDIA driver **preinstalled**. Building a driver via DKMS on a Secure
Boot machine requires enrolling a signing key through firmware, which cannot be done headless.

## Verify the GPU in three layers

Each layer can pass while the next fails, so check all three:

```bash
nvidia-smi                                                   # 1. host driver
nvidia-ctk --version                                         # 2. container toolkit
docker run --rm --gpus all nvidia/cuda:12.6.3-runtime-ubuntu24.04 nvidia-smi   # 3. containers
docker run --rm --gpus all synth-bench-sandbox "nvidia-smi -L"                 # 4. our image
```

## Network

- **Never expose the proxy port (4000).** It holds every provider key behind one master key. Bind it
  to localhost, keep the box without a public IP, and reach it through an SSH tunnel when you need
  the ledger.
- Egress is required: model providers, PyPI/GitHub (the sandbox builds its venv), the public Synth
  API. Route it through NAT rather than giving the box a public address.
- SSH through a bastion or your cloud's identity-aware proxy; no inbound rules otherwise.

Next: [DEPLOY.md](DEPLOY.md).
