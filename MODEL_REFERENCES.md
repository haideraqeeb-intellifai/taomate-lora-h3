# Model references

Weights are excluded from this public Git repository, Git LFS, and release assets. The downloader uses the immutable revisions below and verifies both size and SHA-256. Public access was checked at publication; no Hugging Face gate was active.

## Sources and terms

- Base diffusion, text encoders, and VAEs: [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3). See its [model card and license references](https://huggingface.co/Comfy-Org/MiniMax-H3) and the original [MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) terms.
- ComfyUI-converted three-step adapter: [Robert1212star/TaoMate-H3-3Step-ComfyUI](https://huggingface.co/Robert1212star/TaoMate-H3-3Step-ComfyUI). Its model card states that the original [TaoMate-H3](https://huggingface.co/TaoLiveAIGC/TaoMate-H3) and MiniMax H3 terms apply. A separate project license does not grant rights to these weights.

## Immutable files

### `diffusion_models/minimax_h3_fl2va_bf16.safetensors`

- Purpose / profiles: bf16.
- Repository: `Comfy-Org/MiniMax-H3`; revision: `0bd506d2e895983a9663037febda27aa3948cf48`.
- [Download](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/0bd506d2e895983a9663037febda27aa3948cf48/diffusion_models/minimax_h3_fl2va_bf16.safetensors) — 66,280,487,368 bytes.
- SHA-256: `907d4add438438ec1544f5240c3b38532ed934fe6be75677a6bbda2a6fdd6182`.

### `diffusion_models/minimax_h3_fl2va_int8_convrot.safetensors`

- Purpose / profiles: int8-turbo, int8-euler.
- Repository: `Comfy-Org/MiniMax-H3`; revision: `0bd506d2e895983a9663037febda27aa3948cf48`.
- [Download](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/0bd506d2e895983a9663037febda27aa3948cf48/diffusion_models/minimax_h3_fl2va_int8_convrot.safetensors) — 34,038,892,334 bytes.
- SHA-256: `7ad4c73e6e378b822ffd1629f27f632d3787d95f5e468e3af958f98c58df96a5`.

### `text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors`

- Purpose / profiles: bf16.
- Repository: `Comfy-Org/MiniMax-H3`; revision: `0bd506d2e895983a9663037febda27aa3948cf48`.
- [Download](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/0bd506d2e895983a9663037febda27aa3948cf48/text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors) — 51,506,295,256 bytes.
- SHA-256: `600d567f6a9629c8574e8e7041b199bdd9c59a986afa7906910a81919610607d`.

### `text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors`

- Purpose / profiles: int8-turbo, int8-euler.
- Repository: `Comfy-Org/MiniMax-H3`; revision: `0bd506d2e895983a9663037febda27aa3948cf48`.
- [Download](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/0bd506d2e895983a9663037febda27aa3948cf48/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors) — 15,687,142,551 bytes.
- SHA-256: `35a88d51044231fe332301d7a62aa81e3f2cba62febeb446e2c1e3e0ef76f2c6`.

### `vae/minimax_h3_audio_vae_fp32.safetensors`

- Purpose / profiles: bf16, int8-turbo, int8-euler.
- Repository: `Comfy-Org/MiniMax-H3`; revision: `0bd506d2e895983a9663037febda27aa3948cf48`.
- [Download](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/0bd506d2e895983a9663037febda27aa3948cf48/vae/minimax_h3_audio_vae_fp32.safetensors) — 605,254,808 bytes.
- SHA-256: `8e505d95dd1561d47abd43d4238fd40d9bb1ae9e147ed0a4cba778d76ae4db48`.

### `vae/minimax_h3_video_vae_fp16.safetensors`

- Purpose / profiles: bf16, int8-turbo, int8-euler.
- Repository: `Comfy-Org/MiniMax-H3`; revision: `0bd506d2e895983a9663037febda27aa3948cf48`.
- [Download](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/0bd506d2e895983a9663037febda27aa3948cf48/vae/minimax_h3_video_vae_fp16.safetensors) — 5,207,808,496 bytes.
- SHA-256: `7c1f131492e7eddacaac9069a61b81bdd39de5cc96561e677c5eab1cdce5e522`.

### `loras/taomate_h3_3step_comfy.safetensors`

- Purpose / profiles: bf16, int8-turbo, int8-euler.
- Repository: `Robert1212star/TaoMate-H3-3Step-ComfyUI`; revision: `6897eea8f92ca8a1d511612dbf3ea51a63399cc4`.
- [Download](https://huggingface.co/Robert1212star/TaoMate-H3-3Step-ComfyUI/resolve/6897eea8f92ca8a1d511612dbf3ea51a63399cc4/taomate_h3_3step_comfy.safetensors) — 2,481,007,456 bytes.
- SHA-256: `c1c057121a5ebf77d708b8a5c331ebb78416b90775df465c02a4fa48688315cb`.

## Retrieval and provenance

Run `python download_models.py --profile bf16 --download` or `python download_models.py --profile int8-turbo --download`. Files are stored beneath `models/` in the directory structure expected by the runner. The latter also supplies `int8-euler`. Without `--download`, only the retrieval plan is printed.

Pinned remote metadata verified at publication. Original BF16 diffusion download revision was 7e75982b97cd5a41d2dcfa1904ee88d0686d6fd1; the manifest revision contains the same SHA-256. The archived Turbo run did not independently hash its base/text/VAE files; these hashes identify retrieval targets, not retroactive validation.

The LoRA upload revision was resolved at publication using its matching SHA-256. The benchmark did not record the LoRA repository revision at execution time. VAE checksums above come from immutable upstream metadata; the original run did not independently verify them.

The Turbo node repository includes `h3_silu_temb_grid.safetensors` for its pruned-base adapter path. This publication uses a full, non-pruned base and the native TaoMate LoRA loader, so that tensor is not used or vendored here. Its upstream location, if inspecting that separate path, is [the pinned Turbo repository](https://github.com/Larryvrh/ComfyUI-MiniMax-H3-Turbo/tree/4274783a23afcfdbea3b4876cb79effd6c510785).
