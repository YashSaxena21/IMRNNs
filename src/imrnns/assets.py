from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .encoders import encoder_storage_key, normalize_encoder_name


@dataclass(frozen=True)
class AssetMatch:
    encoder: str
    dataset: str
    path: Path


SPECIAL_CACHE_DIRS = {
    "mini_fiqa": ("mini", "fiqa"),
    "cache_e5": ("e5", "msmarco"),
    "cache_fresh_run": ("mini", "msmarco"),
    "cache_mpnet": ("mpnet", "msmarco"),
}


def default_assets_root() -> Path:
    return Path(__file__).resolve().parents[3]


def package_root() -> Path:
    return Path(__file__).resolve().parents[2]


def discover_cached_embeddings(assets_root: Path) -> list[AssetMatch]:
    assets: list[AssetMatch] = []
    for entry in sorted(assets_root.iterdir()):
        if not entry.is_dir():
            continue

        special = SPECIAL_CACHE_DIRS.get(entry.name)
        if special:
            encoder, dataset = special
            assets.append(AssetMatch(encoder=encoder, dataset=dataset, path=entry))
            continue

        match = re.fullmatch(r"cache_(.+)_(.+)", entry.name)
        if match:
            encoder, dataset = match.groups()
            encoder = encoder_storage_key(encoder)
            assets.append(AssetMatch(encoder=encoder, dataset=dataset, path=entry))
    return assets


def discover_checkpoints(assets_root: Path) -> list[AssetMatch]:
    assets: list[AssetMatch] = []
    for entry in sorted(assets_root.glob("bihypernet_*.pt")):
        match = re.fullmatch(r"bihypernet_(mini|e5|mpnet)(?:_(.+))?\.pt", entry.name)
        if not match:
            continue
        encoder, dataset = match.groups()
        assets.append(AssetMatch(encoder=encoder, dataset=dataset or "msmarco", path=entry))
    return assets


def discover_repo_checkpoints(repo_root: Path) -> list[AssetMatch]:
    assets: list[AssetMatch] = []
    base_dir = repo_root / "checkpoints" / "pretrained"
    if not base_dir.exists():
        return assets
    for entry in sorted(base_dir.rglob("*.pt")):
        encoder = encoder_storage_key(entry.parent.name)
        prefix = f"imrnns-{entry.parent.name}-"
        if not entry.name.startswith(prefix) or not entry.name.endswith(".pt"):
            continue
        dataset = entry.name.removeprefix(prefix).removesuffix(".pt")
        assets.append(AssetMatch(encoder=encoder, dataset=dataset, path=entry))
    return assets


def resolve_cache_dir(assets_root: Path, encoder: str, dataset: str) -> Path:
    encoder = encoder_storage_key(encoder)
    dataset = dataset.lower()
    for asset in discover_cached_embeddings(assets_root):
        if asset.encoder == encoder and asset.dataset.lower() == dataset:
            return asset.path
    direct = assets_root / f"cache_{encoder}_{dataset}"
    if direct.exists():
        return direct
    raise FileNotFoundError(
        f"No cached embeddings found for encoder='{encoder}' dataset='{dataset}' under {assets_root}"
    )


def resolve_checkpoint_path(assets_root: Path, encoder: str, dataset: str) -> Optional[Path]:
    encoder = encoder_storage_key(encoder)
    dataset = dataset.lower()
    for asset in discover_repo_checkpoints(package_root()):
        if asset.encoder == encoder and asset.dataset.lower() == dataset:
            return asset.path
    for asset in discover_checkpoints(assets_root):
        if encoder_storage_key(asset.encoder) == encoder and asset.dataset.lower() == dataset:
            return asset.path
    direct = assets_root / f"imrnns-{encoder}-{dataset}.pt"
    if direct.exists():
        return direct
    return None
