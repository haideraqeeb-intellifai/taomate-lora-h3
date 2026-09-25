# TaoMate LoRA · MiniMax H3

Code, workflows, prompts, measurements, and generated video/audio for the **185.15s BF16** and **approximately 88.53s optimized** TaoMate H3 configurations on an NVIDIA RTX 5090.

**Experiment date: 16 September 2026** for all three configurations, verified from the saved `measured_1_gpu_samples.json` timestamps.

| Launcher profile | Base / text encoder | Diffusion attention | Sampler | Measured generation |
| --- | --- | --- | --- | ---: |
| `bf16` | Full BF16 / BF16 | PyTorch SDPA | Euler, native sigma shifts | 185.145862s |
| `int8-turbo` | Full INT8 ConvRot / NVFP4-AWQ | SageAttention 2.2.0 | MiniMaxH3TurboSampler | 88.524651s |
| `int8-euler` | Full INT8 ConvRot / NVFP4-AWQ | SageAttention 2.2.0 | Euler, native sigma shifts | 88.686618s |

The requested approximately 88.53s result is the archived **Turbo-sampler** run. Its exact time rounds directly to 88.52s at two decimals; rounding the displayed 88.525s intermediate value yields 88.53s. The later Euler run is included separately so the two configurations cannot be confused. These are single measured runs, not statistical averages or an isolated comparison of quantization alone; the sampler and ComfyUI revision also differ.

All three use the same podcast prompt, seed **42**, TaoMate LoRA strength **1.0**, **3 denoising steps**, the `simple` scheduler, **736 × 1280**, **243 frames**, **24 FPS**, and native generated stereo audio. Native duration is **10.125 seconds**. Both Euler profiles use video/audio sigma shifts **12 / 3**. Graph caching and explicit compilation are disabled. Dynamic CPU/GPU offloading is enabled.

Timing includes prompt encoding, denoising, decoding, and MP4 saving, after one complete warm-up in the same process. Initialization and uploads are excluded. CUDA synchronization is used at generation/component boundaries; nested component timings are not additive. See each `report.json` for the original environment, GPU samples, forward counts, and output metadata.

## Saved results

- [185.15s BF16 report](runs/base-h3-bf16-taomate-3step-seed42-v2/index.html) · [JSON](runs/base-h3-bf16-taomate-3step-seed42-v2/report.json) · [video](runs/base-h3-bf16-taomate-3step-seed42-v2/runtime/output/measured_1/podcast_00001_.mp4)
- [88.525s INT8 Turbo report](runs/base-h3-taomate-3step-seed42-larry-sampler-archive/index.html) · [JSON](runs/base-h3-taomate-3step-seed42-larry-sampler-archive/report.json) · [video](runs/base-h3-taomate-3step-seed42-larry-sampler-archive/runtime/output/measured_1/podcast_00001_.mp4)
- [88.69s INT8 Euler report](runs/base-h3-taomate-3step-seed42/index.html) · [JSON](runs/base-h3-taomate-3step-seed42/report.json) · [video](runs/base-h3-taomate-3step-seed42/runtime/output/measured_1/podcast_00001_.mp4)

Open the HTML files locally after cloning, or serve this directory with `python3 -m http.server 8000`. Each run includes its warm-up video, measured video, preview, prompt, workflows, telemetry, and original runner snapshot. Absolute paths inside historical JSON describe the original machine; use the relative links above to find the published files.

## Setup

The recorded GPU environment was Linux, Python **3.12.13**, PyTorch **2.11.0+cu130**, CUDA **13.0**, and NVIDIA driver **595.84**, with approximately **32 GB VRAM**. The original host had about 186 GiB RAM; BF16 weights alone total approximately 126 GB on disk, and inference offloads to host RAM. A minimum host RAM requirement was not established by these runs.

Use separate environments/checkouts for the two ComfyUI revisions. `environment/requirements-*.txt` preserves the corresponding upstream requirements; the Turbo run's JSON also contains its observed package inventory. These are not complete reproducibility lockfiles for the BF16/Euler environment.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r environment/requirements-gpu.txt

mkdir -p upstream
git clone https://github.com/kijai/ComfyUI.git upstream/ComfyUI-euler
git -C upstream/ComfyUI-euler checkout 10febb01d7be73d1491cf5e5347b5ab8b6c2c09e
python -m pip install -r environment/requirements-bf16-euler.txt
```

Install `ffmpeg`/`ffprobe` and ensure `nvidia-smi` is available. The optimized profiles require a working **SageAttention 2.2.0** installation built for the selected CUDA/PyTorch/GPU environment; follow the [upstream build instructions](https://github.com/thu-ml/SageAttention). They require the QK INT8 / PV FP8 CUDA path and apply it only to diffusion; the text encoder and VAEs keep PyTorch attention. The Euler optimized profile also requires the CUDA extension in `comfy-kitchen` to load successfully. The portable launcher uses the installed package, or a built source checkout supplied with `--comfy-kitchen-root`.

For the archived Turbo profile, create a separate environment using `environment/requirements-gpu.txt` and `environment/requirements-int8-turbo.txt`, and prepare these checkouts:

```bash
git clone https://github.com/Comfy-Org/ComfyUI.git upstream/ComfyUI-turbo
git -C upstream/ComfyUI-turbo checkout e377e263049f9338b4d12a3dd417b36ae62948ff
git clone https://github.com/Larryvrh/ComfyUI-MiniMax-H3-Turbo.git upstream/ComfyUI-MiniMax-H3-Turbo
git -C upstream/ComfyUI-MiniMax-H3-Turbo checkout 4274783a23afcfdbea3b4876cb79effd6c510785
```

The Turbo profile uses the external **sampler**, with the **TaoMate** adapter loaded by `LoraLoaderModelOnly`. It does not use the Larryvrh LoRA weights. The archived shared engine retains generic Larryvrh defaults; the included wrapper supplies the TaoMate graph and adapter. Run through `benchmark.py` below.

## Models

No model weights are stored in this repository, Git LFS, or releases. [MODEL_REFERENCES.md](MODEL_REFERENCES.md) and [model-manifest.json](model-manifest.json) provide immutable download URLs, sizes, SHA-256 hashes, access information, and license references.

```bash
# Print the download plan; no weights are fetched by default.
python download_models.py --profile bf16
python download_models.py --profile int8-turbo

# Explicitly download and verify the required weights.
python download_models.py --profile bf16 --download
python download_models.py --profile int8-turbo --download
```

Both optimized profiles use the same weight files. Downloads do not resume partial files. Existing files are checksummed and never overwritten when they disagree with the manifest.

## Run

Run each generation in a fresh process, with the GPU idle and a new output directory. Defaults select the archived prompt, seed, dimensions, and frame count.

```bash
# No GPU or model dependencies: generate the exact workflow for inspection.
python benchmark.py --profile bf16 --dry-run --output outputs/bf16-plan
python benchmark.py --profile int8-turbo --dry-run --output outputs/turbo-plan
python benchmark.py --profile int8-euler --dry-run --output outputs/euler-plan

# BF16 baseline: warm-up plus measured generation.
python benchmark.py --profile bf16 \
  --comfy-root upstream/ComfyUI-euler --model-root models --output outputs/bf16

# The approximately 88.53s historical configuration.
python benchmark.py --profile int8-turbo \
  --comfy-root upstream/ComfyUI-turbo \
  --turbo-root upstream/ComfyUI-MiniMax-H3-Turbo \
  --model-root models --output outputs/int8-turbo

# The later 88.69s Euler comparison.
python benchmark.py --profile int8-euler \
  --comfy-root upstream/ComfyUI-euler --model-root models --output outputs/int8-euler
```

`--prompt PATH` and `--seed NUMBER` allow new experiments; changing them no longer reproduces the recorded workload. The original snapshots enforce their ComfyUI revisions. `BenchmarkSage` is registered by the runner, so optimized workflow JSON is an API graph for this harness rather than a standalone drag-and-drop UI workflow.

## Publication provenance

[SOURCE_INVENTORY.json](SOURCE_INVENTORY.json) records the selected source files and their original checksums. [PUBLISHING_LOG.md](PUBLISHING_LOG.md) records publication changes and verification. Original runner snapshots, measurements, and media are preserved; the top-level launcher adapts machine-specific paths. One historical HTML label was corrected from SageAttention 3 to the recorded 2.2.0 implementation.

Scope is the requested short podcast benchmarks. Other local Ref2VA, long-video, and Nsight experiments are separate work and are not part of this repository. No license has been assigned to the user's original project code; upstream code and model terms remain applicable. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
