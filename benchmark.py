"""Portable entry point for the preserved TaoMate H3 benchmark configurations."""

import argparse
import importlib.util
import json
from pathlib import Path
import sys
import sysconfig

ROOT = Path(__file__).resolve().parent
RUNS = {
    "bf16": "base-h3-bf16-taomate-3step-seed42-v2",
    "int8-turbo": "base-h3-taomate-3step-seed42-larry-sampler-archive",
    "int8-euler": "base-h3-taomate-3step-seed42",
}


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=RUNS, required=True)
    parser.add_argument("--comfy-root", type=Path)
    parser.add_argument("--comfy-kitchen-root", type=Path,
                        help="Optional built source checkout; otherwise use the installed package")
    parser.add_argument("--turbo-root", type=Path,
                        help="Required only for int8-turbo")
    parser.add_argument("--model-root", type=Path, default=ROOT / "models")
    parser.add_argument("--prompt", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true",
                        help="Write the workflow without importing CUDA dependencies or loading weights")
    args = parser.parse_args()
    if not 0 <= args.seed < 2**64:
        parser.error("seed must be an unsigned 64-bit integer")
    if not args.dry_run and args.comfy_root is None:
        parser.error("--comfy-root is required for generation")
    if not args.dry_run and args.profile == "int8-turbo" and args.turbo_root is None:
        parser.error("--turbo-root is required for int8-turbo")

    record = ROOT / "runs" / RUNS[args.profile]
    prompt_path = (args.prompt or record / "prompt.txt").resolve()
    if args.profile == "int8-turbo":
        module = load(ROOT / "run_taomate.py", "taomate_wrapper")
        module.SOURCE = record / "benchmark.py"
        engine = module.load_benchmark()
        engine.MODELS["loras"] = ["taomate_h3_3step_comfy.safetensors"]
        workflow = module.graph_factory(engine)(prompt_path.read_text().strip(), args.seed, "warmup", 3, "sage")
    else:
        module = load(record / "runner.py", "taomate_runner")
        if args.profile == "int8-euler":
            module.MODELS["diffusion"] = "minimax_h3_fl2va_int8_convrot.safetensors"
            module.MODELS["text"] = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
            module.USE_SAGE = True
        workflow = module.graph(prompt_path.read_text().strip(), args.seed, "warmup")
    if args.dry_run:
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output / "workflow.json").write_text(json.dumps(workflow, indent=2) + "\n")
        print(args.output / "workflow.json")
        return

    cli = [str(Path(__file__).resolve()), "--model-root", str(args.model_root.resolve()),
           "--prompt", str(prompt_path), "--seed", str(args.seed),
           "--output", str(args.output.resolve())]
    if args.profile == "int8-turbo":
        cli += ["--comfy-root", str(args.comfy_root.resolve()),
                "--turbo-root", str(args.turbo_root.resolve()),
                "--attention", "sage", "--steps", "3", "--runs", "1"]
    else:
        module.COMFY_ROOT = args.comfy_root.resolve()
        module.COMFY_KITCHEN_ROOT = (args.comfy_kitchen_root.resolve()
                                    if args.comfy_kitchen_root else Path(sysconfig.get_path("purelib")))
        if args.profile == "int8-euler":
            # The archived runner borrowed a CUDA binary from a machine-specific
            # virtualenv. Use the properly installed/built package instead.
            def preload_installed_kitchen():
                import comfy_kitchen  # noqa: F401
            module.preload_comfy_kitchen_cuda_extension = preload_installed_kitchen
            cli += ["--optimized"]
    sys.argv = cli
    module.main()


if __name__ == "__main__":
    main()
