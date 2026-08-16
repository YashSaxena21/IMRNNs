from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi


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

    uploads: list[tuple[Path, str]] = [
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

    folder_uploads = [
        (repo_root / "assets" / "brand", "assets/brand"),
        (repo_root / "checkpoints" / "validated", "checkpoints/validated"),
        (repo_root / "src" / "imrnns", "src/imrnns"),
    ]
    for folder_path, remote_root in folder_uploads:
        uploads.extend(
            (local_path, f"{remote_root}/{local_path.relative_to(folder_path).as_posix()}")
            for local_path in sorted(folder_path.rglob("*"))
            if local_path.is_file()
        )

    desired_checkpoints = {
        remote_path for _, remote_path in uploads if remote_path.startswith("checkpoints/")
    }
    remote_files = set(api.list_repo_files(repo_id=args.repo_id, repo_type="model"))
    obsolete_checkpoints = sorted(
        remote_path
        for remote_path in remote_files
        if remote_path.startswith("checkpoints/") and remote_path not in desired_checkpoints
    )

    operations = [
        CommitOperationDelete(path_in_repo=remote_path)
        for remote_path in obsolete_checkpoints
    ]
    operations.extend(
        CommitOperationAdd(path_in_repo=remote_path, path_or_fileobj=local_path)
        for local_path, remote_path in uploads
    )

    parent_commit = api.model_info(repo_id=args.repo_id).sha
    api.create_commit(
        repo_id=args.repo_id,
        repo_type="model",
        operations=operations,
        commit_message="Publish the current IMRNNs release",
        commit_description=(
            "Synchronize the model card, package sources, release metadata, and "
            "checkpoint set with the current public release."
        ),
        parent_commit=parent_commit,
    )

    print(f"Published Hugging Face model repo: https://huggingface.co/{args.repo_id}")
    if obsolete_checkpoints:
        print(f"Removed {len(obsolete_checkpoints)} superseded checkpoint file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
