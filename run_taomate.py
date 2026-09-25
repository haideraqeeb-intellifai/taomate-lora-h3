"""Run the report prompt on MiniMax H3 with the TaoMate 3-step ComfyUI LoRA."""

import argparse
import importlib.util
import json
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "runs/base-h3-taomate-3step-seed42-larry-sampler-archive/benchmark.py"


def load_benchmark():
    spec = importlib.util.spec_from_file_location("h3_report_benchmark", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def graph_factory(benchmark):
    def graph(prompt, seed, phase, steps=3, attention_backend="sage"):
        workflow = {}

        def node(name, kind, **inputs):
            workflow[name] = {"class_type": kind, "inputs": inputs}
            return [name, 0]

        model = node("diffusion", "UNETLoader", unet_name=benchmark.MODELS["diffusion_models"][0], weight_dtype="default")
        clip = node("text", "CLIPLoader", clip_name=benchmark.MODELS["text_encoders"][0], type="minimax", device="default")
        vae = node("video_vae", "VAELoader", vae_name=benchmark.MODELS["vae"][0])
        audio_vae = node("audio_vae", "VAELoader", vae_name=benchmark.MODELS["vae"][1])
        conditioning = node("conditioning", "MiniMaxH3ImageToVideo", clip=clip, vae=vae,
                            prompt=prompt, width=736, height=1280, length=243)
        model = node("adapter", "LoraLoaderModelOnly", model=model,
                     lora_name=benchmark.MODELS["loras"][0], strength_model=1.0)
        if attention_backend == "sage":
            model = node("attention", "BenchmarkSage", model=model)
        noise = node("noise", "RandomNoise", noise_seed=seed)
        guider = node("guider", "BasicGuider", model=model, conditioning=conditioning)
        sampler = node("sampler", "MiniMaxH3TurboSampler")
        sigmas = node("schedule", "BasicScheduler", model=model, scheduler="simple", steps=steps, denoise=1.0)
        samples = node("denoise", "SamplerCustomAdvanced", noise=noise, guider=guider,
                       sampler=sampler, sigmas=sigmas, latent_image=["conditioning", 1])
        images = node("video_decode", "VAEDecode", samples=samples, vae=vae)
        audio = node("audio_decode", "VAEDecodeAudio", samples=samples, vae=audio_vae)
        video = node("video", "CreateVideo", images=images, audio=audio, fps=24.0, bit_depth=8)
        node("save", "SaveVideo", video=video, filename_prefix=phase + "/podcast", format="mp4", codec="auto")
        return workflow

    return graph


def main():
    benchmark = load_benchmark()
    benchmark.MODELS["loras"] = ["taomate_h3_3step_comfy.safetensors"]
    benchmark.graph = graph_factory(benchmark)

    original_add_argument = argparse.ArgumentParser.add_argument

    def patched_add_argument(parser, *names, **options):
        if "--steps" in names:
            options["choices"] = [3]
            options["default"] = 3
        elif "--lora-name" in names:
            options["choices"] = ["taomate_h3_3step_comfy.safetensors"]
            options["default"] = "taomate_h3_3step_comfy.safetensors"
        return original_add_argument(parser, *names, **options)

    original_argv = list(sys.argv)
    argparse.ArgumentParser.add_argument = patched_add_argument
    try:
        benchmark.main()
    finally:
        argparse.ArgumentParser.add_argument = original_add_argument

    output_index = original_argv.index("--output") + 1
    report_path = Path(original_argv[output_index]).resolve() / "report.json"
    report = json.loads(report_path.read_text())
    report["adapter"] = {
        "repo": "Robert1212star/TaoMate-H3-3Step-ComfyUI",
        "file": "taomate_h3_3step_comfy.safetensors",
        "strength": 1.0,
        "low_vram": False,
        "sha256": "c1c057121a5ebf77d708b8a5c331ebb78416b90775df465c02a4fa48688315cb",
    }
    report["model_change"] = "Larryvrh Turbo LoRA replaced by TaoMate-H3 3-step LoRA; full MiniMax H3 FL2VA base retained"
    report_path.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
