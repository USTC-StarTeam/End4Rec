"""CLI launcher for END4Rec training."""

from __future__ import annotations

import argparse

from train import main as train_main


def parse_args():
    parser = argparse.ArgumentParser(description="END4Rec trainer")
    parser.add_argument(
        "--config",
        type=str,
        default="",
        help="Path to a JSON config file. If omitted, built-in defaults are used.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    train_main(args.config or None)


if __name__ == "__main__":
    main()
