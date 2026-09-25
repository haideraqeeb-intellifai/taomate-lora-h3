"""Benchmark clean BF16 MiniMax H3 plus the TaoMate 3-step LoRA."""

import argparse
import asyncio
from collections import Counter
from fractions import Fraction
import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import threading
import time


ROOT = Path(__file__).resolve().parent
COMFY_ROOT = Path("/home/raqeeb/fasth3-5090/ComfyUI")
COMFY_KITCHEN_ROOT = Path("/home/raqeeb/fasth3-5090/comfy-kitchen")
COMFY_COMMIT = "10febb01d7be73d1491cf5e5347b5ab8b6c2c09e"
MODELS = {
    "diffusion": "minimax_h3_fl2va_bf16.safetensors",
    "text": "qwen3vl_32b_minimax_h3_bf16.safetensors",
    "video_vae": "minimax_h3_video_vae_fp16.safetensors",
    "audio_vae": "minimax_h3_audio_vae_fp32.safetensors",
    "lora": "taomate_h3_3step_comfy.safetensors",
}
USE_SAGE = False


def preload_comfy_kitchen_cuda_extension():
    """Use the installed CUDA binary with the newer local comfy-kitchen Python API."""
    missing = COMFY_KITCHEN_ROOT / "comfy_kitchen/backends/cuda/_C.abi3.so"
    installed = Path(
        "/home/raqeeb/uv/envs/h3_5090/lib/python3.12/site-packages/"
        "comfy_kitchen/backends/cuda/_C.abi3.so"
    )
    if missing.is_file():
        import comfy_kitchen  # noqa: F401
        return
    if not installed.is_file():
        raise FileNotFoundError(f"No comfy-kitchen CUDA extension at {missing} or {installed}")
    real_exists = os.path.exists
    real_spec = importlib.util.spec_from_file_location

    def redirected_exists(path):
        return True if os.fspath(path) == str(missing) else real_exists(path)

    def redirected_spec(name, location, *values, **options):
        if name == "comfy_kitchen.backends.cuda._C" and os.fspath(location) == str(missing):
            location = installed
        return real_spec(name, location, *values, **options)

    os.path.exists = redirected_exists
    importlib.util.spec_from_file_location = redirected_spec
    try:
        import comfy_kitchen  # noqa: F401
    finally:
        os.path.exists = real_exists
        importlib.util.spec_from_file_location = real_spec


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def graph(prompt, seed, phase):
    workflow = {}

    def node(name, kind, **inputs):
        workflow[name] = {"class_type": kind, "inputs": inputs}
        return [name, 0]

    model = node("diffusion", "UNETLoader", unet_name=MODELS["diffusion"], weight_dtype="default")
    clip = node("text", "CLIPLoader", clip_name=MODELS["text"], type="minimax", device="default")
    vae = node("video_vae", "VAELoader", vae_name=MODELS["video_vae"])
    audio_vae = node("audio_vae", "VAELoader", vae_name=MODELS["audio_vae"])
    conditioning = node("conditioning", "MiniMaxH3ImageToVideo", clip=clip, vae=vae,
                        prompt=prompt, width=736, height=1280, length=243)
    model = node("adapter", "LoraLoaderModelOnly", model=model,
                 lora_name=MODELS["lora"], strength_model=1.0)
    model = node("sigma_shift", "MiniMaxH3SigmaShift", model=model,
                 shift_video=12.0, shift_audio=3.0)
    if USE_SAGE:
        model = node("attention", "BenchmarkSage", model=model)
    noise = node("noise", "RandomNoise", noise_seed=seed)
    guider = node("guider", "BasicGuider", model=model, conditioning=conditioning)
    sampler = node("sampler", "KSamplerSelect", sampler_name="euler")
    sigmas = node("schedule", "BasicScheduler", model=model, scheduler="simple", steps=3, denoise=1.0)
    samples = node("denoise", "SamplerCustomAdvanced", noise=noise, guider=guider,
                   sampler=sampler, sigmas=sigmas, latent_image=["conditioning", 1])
    images = node("video_decode", "VAEDecode", samples=samples, vae=vae)
    audio = node("audio_decode", "VAEDecodeAudio", samples=samples, vae=audio_vae)
    video = node("video", "CreateVideo", images=images, audio=audio, fps=24.0, bit_depth=8)
    node("save", "SaveVideo", video=video, filename_prefix=phase + "/podcast", format="mp4", codec="auto")
    return workflow


class Progress:
    client_id = None
    last_node_id = None

    def send_sync(self, event, data, sid=None):
        if event == "execution_error":
            print(json.dumps(data), flush=True)


class Measurements:
    def __init__(self, torch):
        self.torch = torch
        self.calls = []
        self.originals = []

    def wrap(self, owner, method, label):
        original = inspect.getattr_static(owner, method)
        function = original.__func__ if isinstance(original, (classmethod, staticmethod)) else original

        def measured(*args, **kwargs):
            self.torch.cuda.synchronize()
            started = time.perf_counter()
            try:
                return function(*args, **kwargs)
            finally:
                self.torch.cuda.synchronize()
                elapsed = time.perf_counter() - started
                self.calls.append({"component": label, "seconds": elapsed})
                print(f"{label}: {elapsed:.3f} seconds", flush=True)

        replacement = type(original)(measured) if isinstance(original, (classmethod, staticmethod)) else measured
        setattr(owner, method, replacement)
        self.originals.append((owner, method, original))

    def close(self):
        for owner, method, original in reversed(self.originals):
            setattr(owner, method, original)


def main():
    global USE_SAGE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, default=ROOT / "models-base-lora-only")
    parser.add_argument("--prompt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--optimized", action="store_true",
                        help="Use INT8 ConvRot base, NVFP4-AWQ text encoder, and SageAttention 2.2.0")
    args = parser.parse_args()
    USE_SAGE = args.optimized
    if args.optimized:
        MODELS["diffusion"] = "minimax_h3_fl2va_int8_convrot.safetensors"
        MODELS["text"] = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    prompt = args.prompt.read_text().strip()
    workflow = graph(prompt, args.seed, "warmup")
    save(output / "workflow.json", workflow)
    (output / "prompt.txt").write_text(prompt + "\n")
    (output / "runner.py").write_bytes(Path(__file__).read_bytes())

    report = {
        "status": "preflight", "runs": [], "seed": args.seed,
        "width": 736, "height": 1280, "frames": 243, "fps": 24,
        "duration_seconds": 10.125, "steps": 3, "strength": 1.0,
        "base_model": ("MiniMax H3 FL2VA full INT8 ConvRot (non-pruned)" if args.optimized
                       else "MiniMax H3 FL2VA full BF16 (non-pruned, non-quantized)"),
        "text_encoder": ("Qwen3-VL 32B NVFP4-AWQ" if args.optimized
                         else "Qwen3-VL 32B BF16 (non-quantized)"),
        "adapter": "Robert1212star/TaoMate-H3-3Step-ComfyUI",
        "attention": ("SageAttention 2.2.0 QK INT8 / PV FP8 CUDA for diffusion; PyTorch elsewhere"
                      if args.optimized else "PyTorch scaled dot-product attention"),
        "sampler": "Euler", "scheduler": "simple",
        "sigma_shift_video": 12.0, "sigma_shift_audio": 3.0,
        "optimizations_enabled": (["INT8 ConvRot diffusion checkpoint", "NVFP4-AWQ text encoder", "SageAttention 2.2.0", "dynamic VRAM offloading"]
                                  if args.optimized else ["dynamic VRAM offloading"]),
        "optimizations_excluded": (["Larryvrh MiniMaxH3TurboSampler", "VSA", "Spectrum", "EasyCache", "graph compilation"]
                                   if args.optimized else ["SageAttention", "INT8/FP8/NVFP4 model quantization", "VSA", "Spectrum", "EasyCache", "graph compilation"]),
        "memory_management": "Dynamic CPU/GPU weight offloading",
        "comfy_commit": COMFY_COMMIT,
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "artifact_sha256": {
            MODELS["diffusion"]: ("7ad4c73e6e378b822ffd1629f27f632d3787d95f5e468e3af958f98c58df96a5" if args.optimized else "907d4add438438ec1544f5240c3b38532ed934fe6be75677a6bbda2a6fdd6182"),
            MODELS["text"]: ("35a88d51044231fe332301d7a62aa81e3f2cba62febeb446e2c1e3e0ef76f2c6" if args.optimized else "600d567f6a9629c8574e8e7041b199bdd9c59a986afa7906910a81919610607d"),
            MODELS["lora"]: "c1c057121a5ebf77d708b8a5c331ebb78416b90775df465c02a4fa48688315cb",
        },
        "timing_method": "One complete warm-up followed by one measured run in the same process. CUDA synchronized at full-generation and component boundaries. Full time includes prompt encoding, denoising, decoding, and MP4 saving; initialization is excluded.",
    }
    save(output / "report.json", report)
    meter = None
    try:
        busy = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True).strip()
        if busy:
            raise RuntimeError(f"GPU occupied by {busy}")
        actual_commit = subprocess.check_output(["git", "-C", str(COMFY_ROOT), "rev-parse", "HEAD"], text=True).strip()
        if actual_commit != COMFY_COMMIT:
            raise RuntimeError(f"Unexpected ComfyUI commit {actual_commit}")
        paths = {
            "diffusion": args.model_root / "diffusion_models" / MODELS["diffusion"],
            "text": args.model_root / "text_encoders" / MODELS["text"],
            "video_vae": args.model_root / "vae" / MODELS["video_vae"],
            "audio_vae": args.model_root / "vae" / MODELS["audio_vae"],
            "lora": args.model_root / "loras" / MODELS["lora"],
        }
        for path in paths.values():
            if not path.is_file():
                raise FileNotFoundError(path)
        report["files"] = {name: {"path": str(path.resolve()), "bytes": path.stat().st_size} for name, path in paths.items()}
        report["hardware"] = subprocess.check_output(["nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total", "--format=csv"], text=True)
        report["python"] = sys.version
        report["platform"] = platform.platform()

        sys.path.insert(0, str(COMFY_ROOT))
        sys.path.insert(0, str(COMFY_KITCHEN_ROOT))
        sys.argv = [sys.argv[0], "--use-pytorch-cross-attention", "--enable-dynamic-vram"]
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        if args.optimized:
            preload_comfy_kitchen_cuda_extension()
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        report["torch"] = torch.__version__
        report["cuda"] = torch.version.cuda
        initialized = time.perf_counter()
        from comfy.cli_args import args as comfy_args
        comfy_args.models_directory = str(args.model_root.resolve())
        comfy_args.disable_all_custom_nodes = True
        comfy_args.cache_none = True
        for name in ("input", "output", "temp", "user"):
            directory = output / "runtime" / name
            directory.mkdir(parents=True)
            setattr(comfy_args, name + "_directory", str(directory))
        import main as comfy_main
        import nodes
        import execution
        import comfy.sd
        import comfy.samplers
        import comfy.ldm.modules.attention as attention
        import comfy.memory_management
        awaitable = nodes.init_extra_nodes(init_custom_nodes=False, init_api_nodes=False)
        asyncio.run(awaitable)
        if attention.optimized_attention.__name__ != "attention_pytorch":
            raise RuntimeError(f"Expected PyTorch attention, got {attention.optimized_attention.__name__}")

        counts = Counter()
        if args.optimized:
            if not attention.SAGE_ATTENTION_IS_AVAILABLE:
                raise RuntimeError("SageAttention is unavailable")
            sage_version = importlib.metadata.version("sageattention")
            if sage_version != "2.2.0":
                raise RuntimeError(f"Expected SageAttention 2.2.0, got {sage_version}")
            import sageattention.core as sage_core
            original_sage = attention.sageattn
            original_cuda_sage = sage_core.sageattn_qk_int8_pv_fp8_cuda

            def counted_sage(*values, **options):
                counts["sage_attempts"] += 1
                try:
                    result = original_sage(*values, **options)
                    counts["sage_completed"] += 1
                    return result
                except Exception:
                    counts["sage_errors"] += 1
                    raise

            def counted_cuda_sage(*values, **options):
                result = original_cuda_sage(*values, **options)
                counts["sage_qk_int8_pv_fp8_cuda_completed"] += 1
                return result

            attention.sageattn = counted_sage
            sage_core.sageattn_qk_int8_pv_fp8_cuda = counted_cuda_sage

            class BenchmarkSage:
                @classmethod
                def INPUT_TYPES(cls):
                    return {"required": {"model": ("MODEL",)}}

                RETURN_TYPES = ("MODEL",)
                FUNCTION = "patch"
                CATEGORY = "benchmark"

                def patch(self, model):
                    clone = model.clone()

                    def override(original, *values, **options):
                        counts["sage_dispatches"] += 1
                        completed_before = counts["sage_completed"]
                        result = attention.attention_sage(*values, **options)
                        if counts["sage_completed"] == completed_before:
                            counts["sage_pytorch_fallbacks"] += 1
                        return result

                    clone.model_options["transformer_options"]["optimized_attention_override"] = override
                    return (clone,)

            nodes.NODE_CLASS_MAPPINGS["BenchmarkSage"] = BenchmarkSage
            report["sageattention_version"] = sage_version
            report["gpu_compute_capability"] = list(torch.cuda.get_device_capability())
            report["attention_scope"] = "diffusion model only; text encoder and VAEs use PyTorch attention"

        from comfy.ldm.minimax.model import MiniMaxH3Model
        original_forward = MiniMaxH3Model.forward
        def counted_forward(self, *values, **options):
            counts["transformer_forwards"] += 1
            return original_forward(self, *values, **options)
        MiniMaxH3Model.forward = counted_forward

        executor = execution.PromptExecutor(Progress(), cache_type=execution.CacheType.NONE,
                                            cache_args={"ram": 0, "ram_inactive": 0, "lru": 0})
        torch.cuda.synchronize()
        report["initialization_seconds"] = time.perf_counter() - initialized
        report["dynamic_vram"] = comfy.memory_management.aimdo_enabled
        meter = Measurements(torch)
        for kind in {item["class_type"] for item in workflow.values()}:
            cls = nodes.NODE_CLASS_MAPPINGS[kind]
            method = "execute" if "execute" in cls.__dict__ else cls.FUNCTION
            meter.wrap(cls, method, "node/" + kind)
        meter.wrap(comfy.sd.CLIP, "tokenize", "text_tokenization")
        meter.wrap(comfy.sd.CLIP, "encode_from_tokens_scheduled", "text_encoder")
        meter.wrap(comfy.samplers.KSAMPLER, "sample", "denoising_loop")

        for index, phase in enumerate(("warmup", "measured_1")):
            workflow = graph(prompt, args.seed, phase)
            save(output / f"{phase}_workflow.json", workflow)
            valid, error, outputs, details = asyncio.run(execution.validate_prompt(phase, workflow, None))
            if not valid:
                raise RuntimeError(f"Invalid workflow: {error}; {details}")
            meter.calls.clear()
            counts.clear()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            report["status"] = phase
            save(output / "report.json", report)
            samples = []
            stop = threading.Event()
            def monitor():
                while not stop.is_set():
                    try:
                        row = subprocess.check_output(["nvidia-smi", "--query-gpu=timestamp,memory.used,utilization.gpu,power.draw", "--format=csv,noheader,nounits"], text=True).strip()
                        samples.append(row)
                    except Exception as exc:
                        samples.append(repr(exc))
                    stop.wait(1)
            thread = threading.Thread(target=monitor, daemon=True)
            thread.start()
            print(f"START {phase}", flush=True)
            try:
                started = time.perf_counter()
                executor.execute(workflow, phase, execute_outputs=outputs)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - started
            finally:
                stop.set(); thread.join()
                save(output / f"{phase}_gpu_samples.json", samples)
            if not executor.success:
                raise RuntimeError(str(executor.status_messages))
            if counts["transformer_forwards"] != 3:
                raise RuntimeError(f"Expected 3 transformer forwards, got {counts}")
            if args.optimized and counts["sage_qk_int8_pv_fp8_cuda_completed"] == 0:
                raise RuntimeError(f"SageAttention CUDA kernel did not execute: {counts}")
            if args.optimized and counts["sage_pytorch_fallbacks"]:
                raise RuntimeError(f"Unexpected SageAttention fallback: {counts}")
            videos = list((output / "runtime/output" / phase).glob("*.mp4"))
            if len(videos) != 1:
                raise RuntimeError(f"Expected one MP4, got {videos}")
            video = videos[0]
            probe = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(video)], text=True))
            stream = next(item for item in probe["streams"] if item["codec_type"] == "video")
            if (stream["width"], stream["height"], int(stream["nb_frames"]), Fraction(stream["avg_frame_rate"])) != (736, 1280, 243, 24):
                raise RuntimeError("Unexpected output geometry")
            if not any(item["codec_type"] == "audio" for item in probe["streams"]):
                raise RuntimeError("Audio missing")
            totals = Counter()
            for call in meter.calls:
                totals[call["component"]] += call["seconds"]
            report["runs"].append({
                "phase": phase, "generation_seconds": elapsed,
                "components_seconds": dict(totals), "calls": list(meter.calls),
                "forward_counts": dict(counts),
                "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
                "sampled_peak_device_memory_mib": max((int(row.split(",")[1]) for row in samples if "," in row), default=None),
                "video": str(video), "ffprobe": probe,
            })
            save(output / "report.json", report)
            print(f"DONE {phase}: {elapsed:.3f} seconds; {video}", flush=True)
            comfy_main.hook_breaker_ac10a0.restore_functions()
        report["status"] = "complete"
    except BaseException as exc:
        report["status"] = "failed"
        report["error"] = repr(exc)
        raise
    finally:
        if meter:
            meter.close()
        save(output / "report.json", report)


if __name__ == "__main__":
    main()
