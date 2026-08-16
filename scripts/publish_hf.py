from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish the IMRNNs release bundle to a Hugging Face model repo.")
    parser.add_argument("--repo-id", required=True, help="Target Hugging Face repo id, e.g. YashSaxena21/IMRNNs")
    parser.add_argument("--private", action="store_true", help="Create the model repo as private")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Local IMRNNs repository root",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    api = HfApi()
    api.create_repo(repo_id=args.repo_id, repo_type="model", private=args.private, exist_ok=True)

    uploads = [
        (repo_root / "huggingface" / "README.md", "README.md"),
        (repo_root / "huggingface" / "config.json", "config.json"),
        (repo_root / "LICENSE", "LICENSE"),
        (repo_root / "ATTRIBUTION.md", "ATTRIBUTION.md"),
        (repo_root / "CITATION.cff", "CITATION.cff"),
        (repo_root / "requirements.txt", "requirements.txt"),
        (repo_root / "pyproject.toml", "pyproject.toml"),
        (repo_root / "TRAINING_STUDY.md", "TRAINING_STUDY.md"),
        (repo_root / "scripts" / "minimal_eval.py", "scripts/minimal_eval.py"),
    ]

    for local_path, remote_path in uploads:
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=remote_path,
            repo_id=args.repo_id,
            repo_type="model",
        )

    api.upload_folder(
        folder_path=str(repo_root / "checkpoints" / "validated"),
        path_in_repo="checkpoints/validated",
        repo_id=args.repo_id,
        repo_type="model",
    )
    api.upload_folder(
        folder_path=str(repo_root / "src" / "imrnns"),
        path_in_repo="src/imrnns",
        repo_id=args.repo_id,
        repo_type="model",
    )

    print(f"Published Hugging Face model repo: https://huggingface.co/{args.repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
