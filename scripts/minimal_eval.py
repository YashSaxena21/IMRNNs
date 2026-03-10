from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from imrnns import evaluate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal IMRNN checkpoint evaluator.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--encoder", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--datasets-dir", type=Path, default=Path("datasets"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--feedback-k", type=int, default=100)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = evaluate(
        encoder=args.encoder,
        dataset=args.dataset,
        cache_dir=args.cache_dir,
        datasets_dir=args.datasets_dir,
        checkpoint_path=args.checkpoint,
        device=args.device,
        feedback_k=args.feedback_k,
        k=args.k,
    )
    payload = dict(result)
    payload["checkpoint"] = str(payload["checkpoint"])
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
