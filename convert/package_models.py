"""Collect the browser-ready models into models/ and write models/manifest.json (size + SHA-256 per model).

    python convert/package_models.py            # after convert/roformer.py made the two fp16 RoFormer files

The MDX-Net models are already ONNX and run as they are (kept in float32 so they also run on the CPU).
"""
import hashlib
import json
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "models"
SRC = Path(os.environ.get("SPLITWAVE_SOURCE_MODELS", Path.home() / "Library/Application Support/UtauCover/models"))
MODELS = {
    "mdx_voc_ft": ("mdx_voc_ft.onnx", SRC / "UVR-MDX-NET-Voc_FT.onnx"),
    "mdx_kara_2": ("mdx_kara_2.onnx", SRC / "UVR_MDXNET_KARA_2.onnx"),
    "bs_roformer": ("bs_roformer_fp16.onnx", None),            # from convert/roformer.py
    "mel_karaoke": ("mel_roformer_karaoke_fp16.onnx", None),
}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    OUT.mkdir(exist_ok=True)
    manifest = {}
    for mid, (name, src) in MODELS.items():
        dst = OUT / name
        if src is not None and not dst.exists():
            shutil.copy2(src, dst)
        if not dst.exists():
            raise SystemExit(f"missing {dst}: run convert/roformer.py first")
        manifest[mid] = {"file": name, "bytes": dst.stat().st_size, "sha256": sha256(dst)}
        print(f"{mid:12} {name:34} {dst.stat().st_size / 1e6:6.0f} MB")
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {OUT / 'manifest.json'}")


if __name__ == "__main__":
    main()
