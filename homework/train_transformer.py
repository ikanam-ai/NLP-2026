#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from homework.ticket_routing import (
    ARTIFACTS_DIR,
    CONFIDENCE_COVERAGES,
    CONFIDENCE_GRID,
    TransformerConfig,
    build_prediction_frame,
    calibrate_logits_by_task,
    prepare_ticket_splits,
    run_transformer_experiment,
    selective_metrics_table,
    token_length_quantiles,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train multitask or queue-only transformer for the ITSM homework.")
    parser.add_argument("--model-name", default="FacebookAI/xlm-roberta-base")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.10)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--balanced-queue", action="store_true")
    parser.add_argument("--queue-only", action="store_true")
    parser.add_argument("--smoke-batches", type=int, default=None)
    parser.add_argument("--allow-cpu-full-training", action="store_true")
    parser.add_argument("--save-model-state", action="store_true")
    parser.add_argument("--train-last-n-layers", type=int, default=2)
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    parser.add_argument("--padding-strategy", choices=["auto", "longest", "max_length"], default="auto")
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=ARTIFACTS_DIR / "transformer_runs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    splits = prepare_ticket_splits(cache_dir=args.cache_dir)

    config = TransformerConfig(
        model_name=args.model_name,
        max_length=args.max_length,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum,
        patience=args.patience,
        seed=args.seed,
        smoke_batches=args.smoke_batches,
        output_dir=str(args.output_dir),
        save_model_state=args.save_model_state,
        gradient_checkpointing=not args.no_gradient_checkpointing,
        train_last_n_layers=args.train_last_n_layers,
        padding_strategy=args.padding_strategy,
    )
    tasks = ("queue",) if args.queue_only else ("queue", "priority", "type")

    result = run_transformer_experiment(
        splits=splits,
        config=config,
        tasks=tasks,
        balanced_queue=args.balanced_queue,
        allow_cpu_full_training=args.allow_cpu_full_training,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result["history"].to_csv(args.output_dir / "history.csv", index=False)
    write_json(
        args.output_dir / "metrics.json",
        {
            "device": result["device"],
            "tasks": result["tasks"],
            "balanced_queue": result["balanced_queue"],
            "config": result["config"],
            "encoder_type": result["encoder_type"],
            "encoder_layer_stack_path": result["encoder_layer_stack_path"],
            "train_last_n_layers": result["train_last_n_layers"],
            "trainable_params": result["trainable_params"],
            "val_metrics": result["val_metrics"],
            "test_metrics": result["test_metrics"],
            "label_classes": result["label_classes"],
        },
    )

    tokenizer = result["tokenizer"]
    length_stats = token_length_quantiles(splits["train"]["text"].tolist(), tokenizer)
    write_json(args.output_dir / "token_length_quantiles.json", length_stats)

    if set(tasks) == {"queue", "priority", "type"}:
        calibration = calibrate_logits_by_task(
            val_logits=result["val_outputs"]["logits"],
            val_labels=result["val_outputs"]["labels"],
            test_logits=result["test_outputs"]["logits"],
        )
        calibration["metrics"].to_csv(args.output_dir / "calibration_metrics.csv", index=False)
        write_json(args.output_dir / "temperatures.json", calibration["temperatures"])

        prediction_frame = build_prediction_frame(
            labels=result["test_outputs"]["labels"],
            logits=calibration["test_logits"],
            label_encoders=result["label_encoders"],
        )
        prediction_frame.to_csv(args.output_dir / "prediction_frame.csv", index=False)

        coarse_table = selective_metrics_table(
            prediction_frame=prediction_frame,
            coverages=CONFIDENCE_COVERAGES,
        )
        fine_table = selective_metrics_table(
            prediction_frame=prediction_frame,
            coverages=CONFIDENCE_GRID,
        )
        coarse_table.to_csv(args.output_dir / "coverage_table.csv", index=False)
        fine_table.to_csv(args.output_dir / "coverage_grid.csv", index=False)

    print(f"[OK] Transformer artifacts saved to {args.output_dir}")


if __name__ == "__main__":
    main()
