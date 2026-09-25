# Third-party notices

This repository does not assign a license to the original benchmark code or outputs. Referenced dependencies retain their own licenses; install them from their pinned upstream sources rather than treating this repository as a license grant.

- ComfyUI: https://github.com/Comfy-Org/ComfyUI and https://github.com/kijai/ComfyUI — upstream LICENSE applies. Revisions: `e377e263049f9338b4d12a3dd417b36ae62948ff` for the Turbo profile, `10febb01d7be73d1491cf5e5347b5ab8b6c2c09e` for BF16/Euler. The environment requirements files are copied from those revisions; their upstream license is preserved in [licenses/ComfyUI-LICENSE.txt](licenses/ComfyUI-LICENSE.txt).
- ComfyUI-MiniMax-H3-Turbo: https://github.com/Larryvrh/ComfyUI-MiniMax-H3-Turbo at `4274783a23afcfdbea3b4876cb79effd6c510785` — upstream LICENSE applies. Only the external sampler is used in `int8-turbo`; its code and weights are not vendored here.
- comfy-kitchen: https://github.com/Comfy-Org/comfy-kitchen — upstream LICENSE applies. The optimized Euler snapshot originally borrowed an installed CUDA extension into a source checkout; the portable launcher instead imports a properly installed/built package.
- SageAttention: https://github.com/thu-ml/SageAttention — upstream license applies; the two optimized run records identify version 2.2.0.
- PyTorch: https://github.com/pytorch/pytorch — upstream LICENSE and notices apply.
- Base models, encoder, VAEs, and TaoMate adapter: see MODEL_REFERENCES.md for authoritative model and license references.

Archived `benchmark.py` is the user's local measurement harness, with a generic Larryvrh graph that the TaoMate wrapper replaces. It is distinct from the third-party Turbo node implementation.
