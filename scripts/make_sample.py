"""Create the deterministic balanced validation sample from the original source ZIP."""

from __future__ import annotations

import argparse
from pathlib import Path

from cx_sentiment.data import create_validation_sample


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--seed", type=int, default=20261016)
    args = parser.parse_args()
    result = create_validation_sample(args.archive, args.output_dir, seed=args.seed)
    print(f"Created {result.frame.height} validation records in {args.output_dir}")


if __name__ == "__main__":
    main()
