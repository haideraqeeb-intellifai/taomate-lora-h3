"""List pinned model URLs, or download and checksum the selected profile's weights."""

import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parent


def checksum(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["bf16", "int8-turbo", "int8-euler"], required=True)
    parser.add_argument("--model-root", type=Path, default=ROOT / "models")
    parser.add_argument("--download", action="store_true", help="Download roughly 58–126 GB, depending on profile")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "model-manifest.json").read_text())
    for item in manifest["files"]:
        if args.profile not in item["profiles"]:
            continue
        target = args.model_root / item["local_path"]
        print(f"{item['bytes']} bytes -> {target}\n{item['url']}", flush=True)
        if not args.download:
            continue
        if target.exists():
            if target.stat().st_size != item["bytes"] or checksum(target) != item["sha256"]:
                raise RuntimeError(f"Existing model differs from manifest: {target}")
            print("Verified existing file", flush=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".part")
        with urllib.request.urlopen(item["url"], timeout=120) as response, temporary.open("wb") as output:
            for chunk in iter(lambda: response.read(8 * 1024 * 1024), b""):
                output.write(chunk)
        if temporary.stat().st_size != item["bytes"] or checksum(temporary) != item["sha256"]:
            raise RuntimeError(f"Download checksum or size mismatch: {temporary}")
        temporary.replace(target)
        print("Verified download", flush=True)


if __name__ == "__main__":
    main()
