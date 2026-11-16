"""Publish models/ to a Hugging Face model repo: the four .onnx files, manifest.json and a model card.

    hf auth login                                  # once, with a token that can write
    python convert/upload_hf.py <user>/<repo>      # creates the (public) repo if needed

Then set VITE_MODELS_URL in web/.env to the URL it prints and rebuild the app.
"""
import json
import sys
import urllib.request
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
CARD = Path(__file__).with_name("hf_model_card.md")


def main():
    if len(sys.argv) != 2 or "/" not in sys.argv[1]:
        sys.exit("usage: python convert/upload_hf.py <user>/<repo>")
    repo = sys.argv[1]
    manifest = json.loads((MODELS / "manifest.json").read_text())
    for entry in manifest.values():
        f = MODELS / entry["file"]
        if not f.exists() or f.stat().st_size != entry["bytes"]:
            sys.exit(f"{f} is missing or doesn't match manifest.json: run convert/package_models.py")
    (MODELS / "README.md").write_text(CARD.read_text().replace("{repo}", repo))
    api = HfApi()
    print(f"signed in as {api.whoami()['name']}")
    api.create_repo(repo, repo_type="model", exist_ok=True, private=False)
    print(f"uploading {sum(e['bytes'] for e in manifest.values()) / 1e6:.0f} MB to {repo} ...")
    api.upload_folder(folder_path=str(MODELS), repo_id=repo, repo_type="model",
                      allow_patterns=["*.onnx", "manifest.json", "README.md"],
                      commit_message="Splitwave models (ONNX, browser-ready)")
    base = f"https://huggingface.co/{repo}/resolve/main/"
    # what a browser does: fetch with an Origin header, follow the redirect to the CDN, need CORS headers
    for entry in [{"file": "manifest.json", "bytes": None}, *manifest.values()]:
        req = urllib.request.Request(base + entry["file"], method="HEAD",
                                     headers={"Origin": "https://splitwave.example", "User-Agent": "splitwave-upload"})
        with urllib.request.urlopen(req) as r:
            size = int(r.headers.get("Content-Length") or 0)
            cors = r.headers.get("Access-Control-Allow-Origin")
        ok = cors in ("*", "https://splitwave.example") and (entry["bytes"] is None or size == entry["bytes"])
        print(f"  {'ok  ' if ok else 'FAIL'} {entry['file']:34} {size / 1e6:7.1f} MB  CORS {cors}")
    print(f"\nSet this in web/.env and rebuild:\nVITE_MODELS_URL={base}")


if __name__ == "__main__":
    main()
