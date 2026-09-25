# Publication log

## 2026-09-25 21:38:45 UTC — Initial public publication preparation

Destination: `haideraqeeb-intellifai/taomate-lora-h3` (public).
Source: `/home/raqeeb/code/h3-optimization/taomate-lora`; the source workspace has no Git revision. Original source file hashes and sizes are recorded in SOURCE_INVENTORY.json. ComfyUI revisions are `10febb01d7be73d1491cf5e5347b5ab8b6c2c09e` (BF16/Euler) and `e377e263049f9338b4d12a3dd417b36ae62948ff` (Turbo); Turbo node revision is `4274783a23afcfdbea3b4876cb79effd6c510785`.

Inventory: three complete short podcast run directories; 39 original source files covering runner snapshots, prompts, workflows, JSON measurements, GPU telemetry, HTML reports, previews, six distinct generated MP4 outputs plus one duplicate warm-up MP4; portable launcher and model-download script; environment requirement snapshots; model manifest, source inventory, notices, and reproduction documentation. All selected source files are included. There are no submodules or publication symlinks. All files fit ordinary Git; no LFS objects are needed.

Intentional exclusions: all model weights (including adapter weights and weight symlinks), with stable download references supplied instead; disposable Python/download caches and application metadata; unrelated Ref2VA, long-video, creator-promo, and Nsight experiments outside the requested configurations. Nothing in the selected run directories was excluded because of file type or size. The original workspace is unchanged; publication was prepared in its `publication/` subdirectory.

Publication adaptations:

- Added `benchmark.py` to select profiles and supply portable runtime paths without modifying archived runner snapshots.
- Redirected the top-level `run_taomate.py` wrapper to the included archived measurement engine instead of a sibling workspace path.
- The optimized Euler launcher imports an installed/built comfy-kitchen CUDA package instead of borrowing a binary from the original machine's virtualenv. A fresh GPU rerun is required to measure the adapted environment's timing.
- Corrected only the archived Turbo HTML attention label from SageAttention 3 NVFP4 to SageAttention 2.2.0 QK INT8 / PV FP8 CUDA, consistent with its JSON and code.
- Kept the approximately 88.53s Turbo configuration distinct from the later 88.69s Euler run; exact raw timing is retained.

Validation before push:

- Python syntax and JSON parsing passed.
- All three CPU-only launcher dry runs exactly matched the archived warm-up workflow JSON.
- Original runner snapshots matched the recorded SHA-256 values for all three runs.
- ffprobe verified every saved MP4: 736 × 1280, 243 frames, 24 FPS, 32 kHz stereo audio.
- Local HTML links resolved; credential/private-key/signed-URL pattern scans found no matches.
- Model revisions, file sizes, and SHA-256 values were checked against public Hugging Face metadata. No weights were downloaded or uploaded during publication.
- The authenticated GitHub API account was verified as `haideraqeeb-intellifai`, and the requested repository name was unused.
- GPU inference was not rerun for publication; the committed measurements are the historical observations, not new performance claims.
