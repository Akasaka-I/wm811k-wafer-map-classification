"""Command-line interface for the WM-811K project."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="wm811k",
        description="WM-811K data preparation, training, and evaluation",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect", help="Inspect the raw pickle dataset")
    inspect_parser.add_argument("--data", type=Path, default=Path("data/LSWMD.pkl"))
    inspect_parser.add_argument("--output", type=Path, default=Path("outputs/experiments/data_audit.json"))

    preprocess_parser = subparsers.add_parser(
        "preprocess", help="Create resized arrays and group-aware splits"
    )
    preprocess_parser.add_argument("--config", type=Path, default=Path("configs/data.yaml"))

    eda_parser = subparsers.add_parser(
        "eda", help="Generate train-only exploratory data analysis figures"
    )
    eda_parser.add_argument("--config", type=Path, default=Path("configs/data.yaml"))

    train_parser = subparsers.add_parser("train", help="Train a configured PyTorch model")
    train_parser.add_argument("--config", type=Path, required=True)

    baseline_parser = subparsers.add_parser(
        "baseline", help="Train an interpretable traditional-ML baseline"
    )
    baseline_parser.add_argument("--config", type=Path, required=True)

    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate a saved checkpoint")
    evaluate_parser.add_argument("--config", type=Path, required=True)
    evaluate_parser.add_argument("--checkpoint", type=Path, required=True)

    validate_parser = subparsers.add_parser(
        "validate", help="Inspect per-class validation metrics for a saved checkpoint"
    )
    validate_parser.add_argument("--config", type=Path, required=True)
    validate_parser.add_argument("--checkpoint", type=Path, required=True)

    report_parser = subparsers.add_parser(
        "report", help="Build the frozen final comparison and error-analysis report"
    )
    report_parser.add_argument("--config", type=Path, default=Path("configs/report.yaml"))

    subparsers.add_parser("models", help="List registered PyTorch models")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)

    if args.command == "inspect":
        from wm811k.data.inspect import inspect_raw_dataset

        summary = inspect_raw_dataset(args.data, args.output)
        print(summary.to_text())
        return

    if args.command == "preprocess":
        from wm811k.data.preprocess import preprocess_from_config

        preprocess_from_config(args.config)
        return

    if args.command == "eda":
        from wm811k.analysis.eda import run_eda_from_config

        run_eda_from_config(args.config)
        return

    if args.command == "train":
        from wm811k.training.trainer import train_from_config

        train_from_config(args.config)
        return

    if args.command == "baseline":
        from wm811k.baselines.random_forest import train_baseline_from_config

        train_baseline_from_config(args.config)
        return

    if args.command == "evaluate":
        from wm811k.evaluation.evaluator import evaluate_checkpoint

        evaluate_checkpoint(args.config, args.checkpoint)
        return

    if args.command == "validate":
        from wm811k.evaluation.evaluator import evaluate_checkpoint

        evaluate_checkpoint(args.config, args.checkpoint, split_name="val")
        return

    if args.command == "report":
        from wm811k.analysis.final_report import build_final_report_from_config

        build_final_report_from_config(args.config)
        return

    if args.command == "models":
        from wm811k.models.factory import available_models

        print("\n".join(available_models()))
        return

    raise RuntimeError(f"Unhandled command: {args.command}")
