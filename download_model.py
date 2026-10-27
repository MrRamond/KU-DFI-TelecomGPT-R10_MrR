"""Download the complete upstream snapshot; weights remain outside Git."""
import argparse
import json
from pathlib import Path

MODEL_ID = "KU-DFI/TelecomGPT-R1"
DEFAULT_DIR = Path(__file__).resolve().parent / "models" / "TelecomGPT-R1"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", default="main", help="Upstream branch, tag or commit SHA")
    parser.add_argument("--output", type=Path, default=DEFAULT_DIR)
    args = parser.parse_args()
    from huggingface_hub import HfApi, snapshot_download

    # Resolve once to avoid mixing files if main changes during a download.
    revision = HfApi().model_info(MODEL_ID, revision=args.revision).sha
    if not revision:
        raise RuntimeError("Hugging Face did not return a model commit SHA.")
    target = args.output.expanduser().resolve()
    print(f"Downloading {MODEL_ID}@{revision} to {target} (~55 GB)")
    snapshot_download(repo_id=MODEL_ID, revision=revision, local_dir=str(target))
    (target / "integration-source.json").write_text(
        json.dumps({"repo_id": MODEL_ID, "revision": revision}, indent=2) + "\n",
        encoding="utf-8",
    )
    print("Download complete. Use run_model.py with this local directory.")


if __name__ == "__main__":
    main()
