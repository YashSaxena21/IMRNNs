from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from imrnns.checkpoints import convert_legacy_checkpoint
from imrnns.encoders import normalize_encoder_name


def infer_metadata(path: Path) -> dict[str, str]:
    match = re.fullmatch(r"imrnns-(minilm|e5)-(.+)\.pt", path.name)
    if not match:
        raise ValueError(f"Cannot infer metadata from checkpoint name: {path.name}")
    encoder, dataset = match.groups()
    normalized_encoder = normalize_encoder_name(encoder)
    return {
        "encoder": encoder,
        "normalized_encoder": normalized_encoder,
        "dataset": dataset,
    }


def main() -> int:
    converted = []
    for checkpoint_path in sorted((REPO_ROOT / "checkpoints" / "pretrained").rglob("*.pt")):
        metadata = infer_metadata(checkpoint_path)
        temp_path = checkpoint_path.with_suffix(".tmp.pt")
        convert_legacy_checkpoint(checkpoint_path, temp_path, metadata=metadata)
        temp_path.replace(checkpoint_path)
        converted.append(str(checkpoint_path))

    print(json.dumps({"converted": converted}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
