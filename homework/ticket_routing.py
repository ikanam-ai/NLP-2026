from __future__ import annotations

import copy
import gc
import hashlib
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from datasets import load_dataset
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, Normalizer
from sklearn.svm import LinearSVC
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import (
    Adafactor,
    AutoModel,
    AutoTokenizer,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)


DATASET_NAME = "Tobi-Bueck/customer-support-tickets"
TARGET_COLUMNS = ("queue", "priority", "type")
LOSS_WEIGHTS = {"queue": 0.70, "priority": 0.15, "type": 0.15}
CONFIDENCE_COVERAGES = (0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 1.00)
CONFIDENCE_GRID = tuple(np.round(np.arange(0.10, 1.01, 0.05), 2).tolist())

HOMEWORK_DIR = Path(__file__).resolve().parent
REPO_ROOT = HOMEWORK_DIR.parent
DATA_DIR = HOMEWORK_DIR / "data"
ARTIFACTS_DIR = HOMEWORK_DIR / "artifacts"
SPLIT_FILES = {
    "train": DATA_DIR / "train_idx.txt",
    "val": DATA_DIR / "val_idx.txt",
    "test": DATA_DIR / "test_idx.txt",
}


@dataclass(slots=True)
class TransformerConfig:
    model_name: str = "FacebookAI/xlm-roberta-base"
    max_length: int = 256
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.10
    epochs: int = 3
    batch_size: int = 8
    grad_accum_steps: int = 4
    patience: int = 2
    seed: int = 42
    smoke_batches: int | None = None
    output_dir: str | None = None
    save_model_state: bool = False
    gradient_checkpointing: bool = True
    mps_memory_fraction: float | None = 1.95
    optimizer_name: str = "auto"
    train_last_n_layers: int | None = 2
    padding_strategy: str = "auto"


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device() -> torch.device:
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_idx(path: Path) -> np.ndarray:
    values = np.loadtxt(path, dtype=np.int64)
    if np.isscalar(values):
        return np.array([int(values)], dtype=np.int64)
    return values.astype(np.int64)


def md5_text(text: str) -> str:
    return hashlib.md5(text.encode("utf-8", errors="ignore")).hexdigest()


def combine_text(subjects: pd.Series, bodies: pd.Series) -> pd.Series:
    return (
        subjects.fillna("").astype(str).str.strip()
        + "\n\n"
        + bodies.fillna("").astype(str).str.strip()
    ).str.strip()


def load_ticket_frame(
    dataset_name: str = DATASET_NAME,
    cache_dir: str | Path | None = None,
) -> pd.DataFrame:
    dataset = load_dataset(dataset_name, split="train", cache_dir=str(cache_dir) if cache_dir else None)
    frame = dataset.to_pandas()
    needed = {"subject", "body", "queue", "priority", "type", "language"}
    missing = sorted(needed - set(frame.columns))
    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}")
    frame = frame.copy()
    frame["subject"] = frame["subject"].fillna("")
    frame["body"] = frame["body"].fillna("")
    frame["type"] = frame["type"].fillna("Unknown")
    frame["text"] = combine_text(frame["subject"], frame["body"])
    frame["row_idx"] = np.arange(len(frame))
    return frame


def prepare_ticket_splits(
    dataset_name: str = DATASET_NAME,
    cache_dir: str | Path | None = None,
) -> dict[str, pd.DataFrame]:
    frame = load_ticket_frame(dataset_name=dataset_name, cache_dir=cache_dir)
    splits: dict[str, pd.DataFrame] = {}
    for split_name, idx_path in SPLIT_FILES.items():
        idx = read_idx(idx_path)
        split_frame = frame.iloc[idx].copy()
        split_frame["row_idx"] = idx
        split_frame = split_frame.reset_index(drop=True)
        splits[split_name] = split_frame
    return splits


def validate_splits(splits: dict[str, pd.DataFrame]) -> dict[str, Any]:
    stats: dict[str, Any] = {
        "sizes": {split: int(len(frame)) for split, frame in splits.items()},
        "unique_row_idx": {},
        "cross_split_intersections": {},
        "cross_text_duplicates": {},
    }

    row_sets: dict[str, set[int]] = {}
    hash_sets: dict[str, set[str]] = {}
    for split_name, frame in splits.items():
        row_idx = frame["row_idx"].astype(int).tolist()
        row_sets[split_name] = set(row_idx)
        stats["unique_row_idx"][split_name] = len(row_idx) == len(row_sets[split_name])

        text_hashes = frame["text"].map(md5_text).tolist()
        hash_sets[split_name] = set(text_hashes)
        stats.setdefault("within_split_duplicates", {})[split_name] = round(
            1.0 - len(hash_sets[split_name]) / max(len(frame), 1),
            6,
        )

    pairs = [("train", "val"), ("train", "test"), ("val", "test")]
    for left, right in pairs:
        key = f"{left}__{right}"
        stats["cross_split_intersections"][key] = len(row_sets[left] & row_sets[right])
        stats["cross_text_duplicates"][key] = len(hash_sets[left] & hash_sets[right])

    return stats


def _length_stats(texts: pd.Series) -> dict[str, float]:
    char_lengths = texts.str.len()
    word_lengths = texts.str.split().map(len)
    return {
        "chars_mean": round(float(char_lengths.mean()), 2),
        "chars_p50": round(float(char_lengths.quantile(0.50)), 2),
        "chars_p90": round(float(char_lengths.quantile(0.90)), 2),
        "chars_p99": round(float(char_lengths.quantile(0.99)), 2),
        "words_mean": round(float(word_lengths.mean()), 2),
        "words_p50": round(float(word_lengths.quantile(0.50)), 2),
        "words_p90": round(float(word_lengths.quantile(0.90)), 2),
        "words_p99": round(float(word_lengths.quantile(0.99)), 2),
    }


def compute_eda_summary(splits: dict[str, pd.DataFrame]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "sizes": pd.DataFrame(
            [{"split": split, "rows": len(frame)} for split, frame in splits.items()]
        ),
        "missing_share": pd.DataFrame(
            [
                {
                    "split": split,
                    **{
                        column: round(float(frame[column].isna().mean()), 6)
                        for column in ("subject", "body", "queue", "priority", "type", "language")
                    },
                }
                for split, frame in splits.items()
            ]
        ),
        "language_distribution": pd.concat(
            [
                frame["language"]
                .fillna("NA")
                .value_counts(normalize=True)
                .rename("share")
                .reset_index()
                .rename(columns={"index": "language"})
                .assign(split=split)
                for split, frame in splits.items()
            ],
            ignore_index=True,
        ),
        "length_stats": pd.DataFrame(
            [{"split": split, **_length_stats(frame["text"])} for split, frame in splits.items()]
        ),
        "class_distribution": {},
        "rare_class_tail": {},
    }

    for target in TARGET_COLUMNS:
        summary["class_distribution"][target] = pd.concat(
            [
                frame[target]
                .astype(str)
                .value_counts()
                .rename("count")
                .reset_index()
                .rename(columns={"index": target})
                .assign(split=split)
                for split, frame in splits.items()
            ],
            ignore_index=True,
        )
        train_counts = splits["train"][target].astype(str).value_counts()
        summary["rare_class_tail"][target] = {
            "top_10": train_counts.head(10).to_dict(),
            "bottom_10": train_counts.tail(10).to_dict(),
        }

    summary["split_validation"] = validate_splits(splits)
    return summary


def calculate_final_score(queue_macro_f1: float, priority_acc: float, type_acc: float) -> float:
    return float(
        LOSS_WEIGHTS["queue"] * queue_macro_f1
        + LOSS_WEIGHTS["priority"] * priority_acc
        + LOSS_WEIGHTS["type"] * type_acc
    )


def _task_metrics(y_true: Sequence[Any], y_pred: Sequence[Any], task: str) -> dict[str, float]:
    metrics: dict[str, float] = {f"{task}_acc": float(accuracy_score(y_true, y_pred))}
    if task == "queue":
        metrics[f"{task}_macro_f1"] = float(f1_score(y_true, y_pred, average="macro"))
    return metrics


def _full_metrics_from_predictions(
    labels: dict[str, np.ndarray],
    preds: dict[str, np.ndarray],
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for task in labels:
        metrics.update(_task_metrics(labels[task], preds[task], task))
    if {"queue", "priority", "type"}.issubset(labels):
        metrics["score"] = calculate_final_score(
            metrics["queue_macro_f1"],
            metrics["priority_acc"],
            metrics["type_acc"],
        )
    return metrics


def _make_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        max_features=200_000,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
    )


def run_linear_baselines(splits: dict[str, pd.DataFrame]) -> dict[str, Any]:
    vectorizer = _make_vectorizer()
    train_matrix = vectorizer.fit_transform(splits["train"]["text"])
    matrices = {
        "train": train_matrix,
        "val": vectorizer.transform(splits["val"]["text"]),
        "test": vectorizer.transform(splits["test"]["text"]),
    }

    results: dict[str, Any] = {
        "model_name": "tfidf_linear_svc",
        "targets": {},
        "summary_rows": [],
        "vectorizer": vectorizer,
    }

    for target in TARGET_COLUMNS:
        clf = LinearSVC(C=1.0)
        y_train = splits["train"][target].astype(str).to_numpy()
        clf.fit(matrices["train"], y_train)
        target_result = {"model": clf}
        for split_name in ("val", "test"):
            y_true = splits[split_name][target].astype(str).to_numpy()
            y_pred = clf.predict(matrices[split_name])
            metrics = _task_metrics(y_true, y_pred, target)
            target_result[split_name] = metrics
        results["targets"][target] = target_result

    for split_name in ("val", "test"):
        row = {"split": split_name, "model_name": "tfidf_linear_svc"}
        for target in TARGET_COLUMNS:
            row.update(results["targets"][target][split_name])
        row["score"] = calculate_final_score(
            row["queue_macro_f1"],
            row["priority_acc"],
            row["type_acc"],
        )
        results["summary_rows"].append(row)

    results["summary"] = pd.DataFrame(results["summary_rows"])
    return results


def run_knn_queue_baseline(
    splits: dict[str, pd.DataFrame],
    n_components: int = 300,
    n_neighbors: int = 15,
    seed: int = 42,
) -> dict[str, Any]:
    vectorizer = _make_vectorizer()
    train_matrix = vectorizer.fit_transform(splits["train"]["text"])
    val_matrix = vectorizer.transform(splits["val"]["text"])
    test_matrix = vectorizer.transform(splits["test"]["text"])

    svd = TruncatedSVD(n_components=n_components, random_state=seed)
    normalizer = Normalizer(copy=False)

    train_reduced = normalizer.fit_transform(svd.fit_transform(train_matrix))
    val_reduced = normalizer.transform(svd.transform(val_matrix))
    test_reduced = normalizer.transform(svd.transform(test_matrix))

    model = KNeighborsClassifier(
        n_neighbors=n_neighbors,
        metric="cosine",
        weights="distance",
    )
    y_train = splits["train"]["queue"].astype(str).to_numpy()
    model.fit(train_reduced, y_train)

    result: dict[str, Any] = {
        "model_name": "tfidf_svd_knn_queue",
        "n_components": n_components,
        "n_neighbors": n_neighbors,
        "vectorizer": vectorizer,
        "svd": svd,
        "normalizer": normalizer,
        "model": model,
        "summary_rows": [],
    }

    for split_name, reduced_matrix in (("val", val_reduced), ("test", test_reduced)):
        y_true = splits[split_name]["queue"].astype(str).to_numpy()
        y_pred = model.predict(reduced_matrix)
        metrics = _task_metrics(y_true, y_pred, "queue")
        result[split_name] = metrics
        result["summary_rows"].append({"split": split_name, "model_name": result["model_name"], **metrics})

    result["summary"] = pd.DataFrame(result["summary_rows"])
    return result


def build_label_encoders(
    train_frame: pd.DataFrame,
    tasks: Sequence[str],
) -> dict[str, LabelEncoder]:
    encoders: dict[str, LabelEncoder] = {}
    for task in tasks:
        encoder = LabelEncoder()
        encoder.fit(train_frame[task].astype(str))
        encoders[task] = encoder
    return encoders


class TicketDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        tokenizer: AutoTokenizer,
        label_encoders: dict[str, LabelEncoder],
        tasks: Sequence[str],
        max_length: int,
    ) -> None:
        self.tasks = tuple(tasks)
        self.encodings = tokenizer(
            frame["text"].tolist(),
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        self.labels = {
            task: label_encoders[task].transform(frame[task].astype(str)).astype(np.int64)
            for task in self.tasks
        }

    def __len__(self) -> int:
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = {key: value[idx] for key, value in self.encodings.items()}
        for task in self.tasks:
            item[f"labels_{task}"] = int(self.labels[task][idx])
        return item


class TicketCollator:
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        padding: bool | str = True,
        max_length: int | None = None,
    ) -> None:
        collator_kwargs: dict[str, Any] = {
            "tokenizer": tokenizer,
            "padding": padding,
        }
        if padding == "max_length":
            collator_kwargs["max_length"] = max_length
        self.inner = DataCollatorWithPadding(**collator_kwargs)

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        label_keys = [key for key in features[0].keys() if key.startswith("labels_")]
        label_payload = {key: [feature.pop(key) for feature in features] for key in label_keys}
        batch = self.inner(features)
        for key, values in label_payload.items():
            batch[key] = torch.tensor(values, dtype=torch.long)
        return batch


class TicketRoutingModel(nn.Module):
    def __init__(
        self,
        model_name: str,
        num_labels: dict[str, int],
        loss_weights: dict[str, float] | None = None,
        class_weights: dict[str, torch.Tensor] | None = None,
        dropout: float = 0.1,
        gradient_checkpointing: bool = True,
        train_last_n_layers: int | None = 2,
    ) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.encoder_type = type(self.encoder).__name__
        self.encoder_layer_stack_path: str | None = None
        self.train_last_n_layers = train_last_n_layers
        self._freeze_encoder_layers(train_last_n_layers)
        if gradient_checkpointing and hasattr(self.encoder, "gradient_checkpointing_enable"):
            self.encoder.gradient_checkpointing_enable()
        self.dropout = nn.Dropout(dropout)
        self.tasks = tuple(num_labels.keys())
        self.heads = nn.ModuleDict(
            {
                self._head_key(task): nn.Linear(self.encoder.config.hidden_size, count)
                for task, count in num_labels.items()
            }
        )
        self.loss_weights = loss_weights or {task: 1.0 for task in num_labels}
        self.class_weights = class_weights or {}

    @staticmethod
    def _head_key(task: str) -> str:
        return f"task__{task}"

    def _resolve_encoder_layer_stack(self) -> tuple[Sequence[nn.Module], str]:
        candidate_paths = (
            "encoder.layer",
            "encoder.layers",
            "layers",
            "transformer.layer",
            "transformer.layers",
            "model.layers",
        )

        for path in candidate_paths:
            current: Any = self.encoder
            for part in path.split("."):
                current = getattr(current, part, None)
                if current is None:
                    break
            if current is None:
                continue
            if isinstance(current, (list, tuple, nn.ModuleList)) and len(current) > 0:
                return current, path

        raise ValueError(
            f"The selected encoder '{self.encoder_type}' does not expose a supported layer stack for partial fine-tuning."
        )

    def _freeze_encoder_layers(self, train_last_n_layers: int | None) -> None:
        if train_last_n_layers is None:
            return

        for parameter in self.encoder.parameters():
            parameter.requires_grad = False

        if train_last_n_layers <= 0:
            return

        encoder_stack, stack_path = self._resolve_encoder_layer_stack()
        self.encoder_layer_stack_path = stack_path

        for layer in encoder_stack[-train_last_n_layers:]:
            for parameter in layer.parameters():
                parameter.requires_grad = True

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **labels: torch.Tensor) -> dict[str, Any]:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        pooled = self.dropout(pooled)

        logits: dict[str, torch.Tensor] = {}
        losses: dict[str, torch.Tensor] = {}
        for task in self.tasks:
            head = self.heads[self._head_key(task)]
            task_logits = head(pooled)
            logits[task] = task_logits
            label_key = f"labels_{task}"
            if label_key in labels:
                weight = self.class_weights.get(task)
                losses[task] = F.cross_entropy(task_logits, labels[label_key], weight=weight)

        total_loss = None
        if losses:
            total_loss = sum(self.loss_weights.get(task, 1.0) * loss for task, loss in losses.items())

        return {"loss": total_loss, "logits": logits, "losses": losses}


def _move_batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def _compute_class_weights(frame: pd.DataFrame, encoder: LabelEncoder) -> torch.Tensor:
    encoded = encoder.transform(frame["queue"].astype(str))
    counts = np.bincount(encoded)
    weights = counts.sum() / (len(counts) * counts.astype(np.float32))
    return torch.tensor(weights, dtype=torch.float32)


def _epoch_iterator(loader: DataLoader, total_limit: int | None = None) -> Iterable[tuple[int, Any]]:
    for batch_index, batch in enumerate(loader, start=1):
        if total_limit is not None and batch_index > total_limit:
            break
        yield batch_index, batch


def predict_logits(
    model: TicketRoutingModel,
    loader: DataLoader,
    tasks: Sequence[str],
    device: torch.device,
    max_batches: int | None = None,
) -> dict[str, Any]:
    model.eval()
    logits_store = {task: [] for task in tasks}
    labels_store = {task: [] for task in tasks}
    running_loss = 0.0
    steps = 0

    with torch.inference_mode():
        for _, batch in _epoch_iterator(loader, max_batches):
            batch = _move_batch_to_device(batch, device)
            outputs = model(**batch)
            steps += 1
            if outputs["loss"] is not None:
                running_loss += float(outputs["loss"].detach().cpu().item())
            for task in tasks:
                logits_store[task].append(outputs["logits"][task].detach().cpu().numpy())
                labels_store[task].append(batch[f"labels_{task}"].detach().cpu().numpy())
            del outputs, batch
            if device.type == "mps":
                torch.mps.empty_cache()

    return {
        "loss": running_loss / max(steps, 1),
        "logits": {task: np.concatenate(chunks, axis=0) for task, chunks in logits_store.items()},
        "labels": {task: np.concatenate(chunks, axis=0) for task, chunks in labels_store.items()},
    }


def compute_transformer_metrics(
    labels: dict[str, np.ndarray],
    logits: dict[str, np.ndarray],
    label_encoders: dict[str, LabelEncoder],
) -> dict[str, Any]:
    predictions: dict[str, np.ndarray] = {}
    metrics: dict[str, Any] = {}
    for task, task_logits in logits.items():
        preds = task_logits.argmax(axis=1)
        predictions[task] = preds
        true_labels = label_encoders[task].inverse_transform(labels[task])
        pred_labels = label_encoders[task].inverse_transform(preds)
        metrics.update(_task_metrics(true_labels, pred_labels, task))

    if {"queue", "priority", "type"}.issubset(logits):
        metrics["score"] = calculate_final_score(
            metrics["queue_macro_f1"],
            metrics["priority_acc"],
            metrics["type_acc"],
        )

    metrics["predictions"] = predictions
    return metrics


def _save_checkpoint(
    output_dir: Path,
    model: TicketRoutingModel,
    config: TransformerConfig,
    label_encoders: dict[str, LabelEncoder],
) -> None:
    ensure_dir(output_dir)
    if config.save_model_state:
        torch.save(model.state_dict(), output_dir / "model_state.pt")
    (output_dir / "config.json").write_text(
        json.dumps(asdict(config), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    serializable_classes = {task: encoder.classes_.tolist() for task, encoder in label_encoders.items()}
    (output_dir / "label_classes.json").write_text(
        json.dumps(serializable_classes, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def run_transformer_experiment(
    splits: dict[str, pd.DataFrame],
    config: TransformerConfig | None = None,
    tasks: Sequence[str] = TARGET_COLUMNS,
    balanced_queue: bool = False,
    allow_cpu_full_training: bool = False,
) -> dict[str, Any]:
    config = config or TransformerConfig()
    set_seed(config.seed)

    device = select_device()
    if device.type == "cpu" and config.smoke_batches is None and not allow_cpu_full_training:
        config = copy.deepcopy(config)
        config.smoke_batches = 4
    if (
        device.type == "mps"
        and config.mps_memory_fraction is not None
        and hasattr(torch.mps, "set_per_process_memory_fraction")
    ):
        try:
            torch.mps.set_per_process_memory_fraction(config.mps_memory_fraction)
        except RuntimeError:
            pass

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    label_encoders = build_label_encoders(splits["train"], tasks)
    num_labels = {task: len(label_encoders[task].classes_) for task in tasks}
    padding_strategy = config.padding_strategy
    if padding_strategy == "auto":
        padding_strategy = "max_length" if device.type == "mps" else "longest"

    datasets = {
        split: TicketDataset(
            frame=frame,
            tokenizer=tokenizer,
            label_encoders=label_encoders,
            tasks=tasks,
            max_length=config.max_length,
        )
        for split, frame in splits.items()
    }
    collator = TicketCollator(
        tokenizer=tokenizer,
        padding=padding_strategy,
        max_length=config.max_length,
    )
    loaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=collator,
        ),
        "val": DataLoader(
            datasets["val"],
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=collator,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=collator,
        ),
    }

    class_weights: dict[str, torch.Tensor] = {}
    if balanced_queue and "queue" in tasks:
        class_weights["queue"] = _compute_class_weights(splits["train"], label_encoders["queue"]).to(device)

    experiment_weights = (
        {task: LOSS_WEIGHTS[task] for task in tasks}
        if {"queue", "priority", "type"}.issubset(tasks)
        else {tasks[0]: 1.0}
    )
    model = TicketRoutingModel(
        model_name=config.model_name,
        num_labels=num_labels,
        loss_weights=experiment_weights,
        class_weights=class_weights,
        gradient_checkpointing=config.gradient_checkpointing,
        train_last_n_layers=config.train_last_n_layers,
    ).to(device)
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]

    optimizer_name = config.optimizer_name
    if optimizer_name == "auto":
        optimizer_name = "adafactor" if device.type == "mps" else "adamw"

    if optimizer_name == "adafactor":
        optimizer = Adafactor(
            trainable_parameters,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            scale_parameter=False,
            relative_step=False,
            warmup_init=False,
        )
    else:
        optimizer = AdamW(trainable_parameters, lr=config.learning_rate, weight_decay=config.weight_decay)
    steps_per_epoch = math.ceil(len(loaders["train"]) / config.grad_accum_steps)
    total_steps = max(steps_per_epoch * config.epochs, 1)
    warmup_steps = int(total_steps * config.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    history: list[dict[str, Any]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_val_key = -float("inf")
    best_val_metrics: dict[str, Any] | None = None
    best_val_outputs: dict[str, Any] | None = None
    epochs_without_improvement = 0
    train_limit = config.smoke_batches
    eval_limit = config.smoke_batches

    for epoch in range(1, config.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        step_counter = 0
        disable_progress = os.environ.get("TQDM_DISABLE", "").strip().lower() in {"1", "true", "yes", "on"}

        progress = tqdm(
            _epoch_iterator(loaders["train"], train_limit),
            total=train_limit or len(loaders["train"]),
            desc=f"epoch {epoch}",
            leave=False,
            disable=disable_progress,
        )
        total_train_batches = train_limit or len(loaders["train"])
        for batch_index, batch in progress:
            batch = _move_batch_to_device(batch, device)
            outputs = model(**batch)
            loss = outputs["loss"] / config.grad_accum_steps
            loss.backward()
            running_loss += float(outputs["loss"].detach().cpu().item())
            step_counter += 1

            should_step = batch_index % config.grad_accum_steps == 0
            is_last_batch = batch_index == total_train_batches
            if should_step or is_last_batch:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                if device.type == "mps":
                    torch.mps.empty_cache()

            del outputs, loss, batch
            if device.type == "mps" and batch_index % 50 == 0:
                gc.collect()
                torch.mps.empty_cache()

        val_outputs = predict_logits(model, loaders["val"], tasks, device, max_batches=eval_limit)
        if device.type == "mps":
            torch.mps.empty_cache()
        val_metrics = compute_transformer_metrics(val_outputs["labels"], val_outputs["logits"], label_encoders)
        history_row = {
            "epoch": epoch,
            "train_loss": running_loss / max(step_counter, 1),
            "val_loss": val_outputs["loss"],
            **{key: value for key, value in val_metrics.items() if key != "predictions"},
        }
        history.append(history_row)

        if "queue_macro_f1" in val_metrics:
            current_val_key = float(val_metrics["queue_macro_f1"])
        else:
            current_val_key = float(val_metrics.get("queue_acc", 0.0))

        if current_val_key > best_val_key:
            best_val_key = current_val_key
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_val_metrics = val_metrics
            best_val_outputs = val_outputs
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                break

    if best_state is None:
        raise RuntimeError("Training loop finished without a valid checkpoint.")

    model.load_state_dict(best_state)
    if config.output_dir:
        _save_checkpoint(Path(config.output_dir), model, config, label_encoders)

    test_outputs = predict_logits(model, loaders["test"], tasks, device, max_batches=eval_limit)
    test_metrics = compute_transformer_metrics(test_outputs["labels"], test_outputs["logits"], label_encoders)

    return {
        "config": asdict(config),
        "device": device.type,
        "tasks": list(tasks),
        "balanced_queue": balanced_queue,
        "encoder_type": model.encoder_type,
        "encoder_layer_stack_path": model.encoder_layer_stack_path,
        "train_last_n_layers": config.train_last_n_layers,
        "trainable_params": int(sum(parameter.numel() for parameter in trainable_parameters)),
        "label_classes": {task: label_encoders[task].classes_.tolist() for task in tasks},
        "history": pd.DataFrame(history),
        "val_metrics": {key: value for key, value in (best_val_metrics or {}).items() if key != "predictions"},
        "test_metrics": {key: value for key, value in test_metrics.items() if key != "predictions"},
        "val_outputs": best_val_outputs,
        "test_outputs": test_outputs,
        "test_predictions": test_metrics.get("predictions", {}),
        "label_encoders": label_encoders,
        "model": model,
        "tokenizer": tokenizer,
    }


def softmax_probs(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def fit_temperature(logits: np.ndarray, labels: np.ndarray, max_iter: int = 50) -> float:
    logits_tensor = torch.tensor(logits, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    log_temperature = nn.Parameter(torch.zeros(1, dtype=torch.float32))
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=max_iter)

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        temperature = torch.exp(log_temperature).clamp(min=1e-3)
        loss = F.cross_entropy(logits_tensor / temperature, labels_tensor)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(torch.exp(log_temperature).detach().cpu().item())


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    return logits / max(temperature, 1e-6)


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for left, right in zip(bins[:-1], bins[1:]):
        mask = (confidences > left) & (confidences <= right)
        if not np.any(mask):
            continue
        accuracy = (predictions[mask] == labels[mask]).mean()
        avg_conf = confidences[mask].mean()
        ece += mask.mean() * abs(float(accuracy) - float(avg_conf))
    return float(ece)


def calibrate_logits_by_task(
    val_logits: dict[str, np.ndarray],
    val_labels: dict[str, np.ndarray],
    test_logits: dict[str, np.ndarray],
) -> dict[str, Any]:
    temperatures: dict[str, float] = {}
    metrics_rows = []
    calibrated_val: dict[str, np.ndarray] = {}
    calibrated_test: dict[str, np.ndarray] = {}

    for task, logits in val_logits.items():
        pre_probs = softmax_probs(logits)
        pre_ece = expected_calibration_error(pre_probs, val_labels[task])
        pre_nll = float(log_loss(val_labels[task], pre_probs, labels=list(range(pre_probs.shape[1]))))

        temperature = fit_temperature(logits, val_labels[task])
        temperatures[task] = temperature
        calibrated_val_logits = apply_temperature(logits, temperature)
        calibrated_test_logits = apply_temperature(test_logits[task], temperature)
        calibrated_val[task] = calibrated_val_logits
        calibrated_test[task] = calibrated_test_logits

        post_probs = softmax_probs(calibrated_val_logits)
        post_ece = expected_calibration_error(post_probs, val_labels[task])
        post_nll = float(log_loss(val_labels[task], post_probs, labels=list(range(post_probs.shape[1]))))

        metrics_rows.append(
            {
                "task": task,
                "temperature": temperature,
                "ece_before": pre_ece,
                "ece_after": post_ece,
                "nll_before": pre_nll,
                "nll_after": post_nll,
            }
        )

    return {
        "temperatures": temperatures,
        "metrics": pd.DataFrame(metrics_rows),
        "val_logits": calibrated_val,
        "test_logits": calibrated_test,
    }


def build_prediction_frame(
    labels: dict[str, np.ndarray],
    logits: dict[str, np.ndarray],
    label_encoders: dict[str, LabelEncoder],
    confidence_name: str = "joint_conf",
) -> pd.DataFrame:
    data: dict[str, Any] = {}
    probs_by_task = {task: softmax_probs(task_logits) for task, task_logits in logits.items()}
    max_probs = {}

    for task, probs in probs_by_task.items():
        pred_ids = probs.argmax(axis=1)
        data[f"true_{task}"] = label_encoders[task].inverse_transform(labels[task])
        data[f"pred_{task}"] = label_encoders[task].inverse_transform(pred_ids)
        data[f"conf_{task}"] = probs.max(axis=1)
        max_probs[task] = probs.max(axis=1)

    if {"queue", "priority", "type"}.issubset(probs_by_task):
        joint_conf = (
            np.power(max_probs["queue"], LOSS_WEIGHTS["queue"])
            * np.power(max_probs["priority"], LOSS_WEIGHTS["priority"])
            * np.power(max_probs["type"], LOSS_WEIGHTS["type"])
        )
        data[confidence_name] = joint_conf

    return pd.DataFrame(data)


def _prediction_dict_from_frame(frame: pd.DataFrame, tasks: Sequence[str], prefix: str) -> dict[str, np.ndarray]:
    return {task: frame[f"{prefix}_{task}"].to_numpy() for task in tasks}


def selective_metrics_table(
    prediction_frame: pd.DataFrame,
    coverages: Sequence[float] = CONFIDENCE_COVERAGES,
    confidence_column: str = "joint_conf",
) -> pd.DataFrame:
    if confidence_column not in prediction_frame.columns:
        raise ValueError(f"Column '{confidence_column}' is missing in prediction_frame")

    tasks = [column.removeprefix("true_") for column in prediction_frame.columns if column.startswith("true_")]
    ordered = prediction_frame.sort_values(confidence_column, ascending=False).reset_index(drop=True)
    rows = []

    for coverage in coverages:
        keep_n = max(1, int(round(len(ordered) * coverage)))
        auto = ordered.iloc[:keep_n].copy()
        manual = ordered.iloc[keep_n:].copy()

        auto_labels = _prediction_dict_from_frame(auto, tasks, "true")
        auto_preds = _prediction_dict_from_frame(auto, tasks, "pred")
        auto_metrics = _full_metrics_from_predictions(auto_labels, auto_preds)

        hybrid = ordered.copy()
        for task in tasks:
            hybrid.loc[manual.index, f"pred_{task}"] = hybrid.loc[manual.index, f"true_{task}"]
        hybrid_labels = _prediction_dict_from_frame(hybrid, tasks, "true")
        hybrid_preds = _prediction_dict_from_frame(hybrid, tasks, "pred")
        hybrid_metrics = _full_metrics_from_predictions(hybrid_labels, hybrid_preds)

        joint_correct = np.logical_and.reduce(
            [(auto[f"true_{task}"] == auto[f"pred_{task}"]).to_numpy() for task in tasks]
        )
        row = {
            "coverage": round(float(coverage), 4),
            "auto_rows": int(len(auto)),
            "manual_rows": int(len(manual)),
            "auto_exact_match": float(joint_correct.mean()) if len(joint_correct) else 0.0,
            "auto_queue_macro_f1": auto_metrics.get("queue_macro_f1"),
            "auto_queue_acc": auto_metrics.get("queue_acc"),
            "auto_priority_acc": auto_metrics.get("priority_acc"),
            "auto_type_acc": auto_metrics.get("type_acc"),
            "auto_score": auto_metrics.get("score"),
            "oracle_queue_macro_f1": hybrid_metrics.get("queue_macro_f1"),
            "oracle_queue_acc": hybrid_metrics.get("queue_acc"),
            "oracle_priority_acc": hybrid_metrics.get("priority_acc"),
            "oracle_type_acc": hybrid_metrics.get("type_acc"),
            "oracle_score": hybrid_metrics.get("score"),
        }
        rows.append(row)

    return pd.DataFrame(rows)


def token_length_quantiles(
    texts: Sequence[str],
    tokenizer: AutoTokenizer,
    quantiles: Sequence[float] = (0.50, 0.90, 0.95, 0.99),
    sample_size: int = 10_000,
    batch_size: int = 256,
    seed: int = 42,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    if len(texts) > sample_size:
        sampled = rng.choice(np.asarray(texts, dtype=object), size=sample_size, replace=False).tolist()
    else:
        sampled = list(texts)

    lengths: list[int] = []
    for start in range(0, len(sampled), batch_size):
        batch = sampled[start : start + batch_size]
        encoded = tokenizer(batch, truncation=False, padding=False)
        lengths.extend(len(item) for item in encoded["input_ids"])

    return {f"q{int(q * 100)}": float(np.quantile(lengths, q)) for q in quantiles}


def json_ready(value: Any) -> Any:
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def write_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(json_ready(payload), indent=2, ensure_ascii=False), encoding="utf-8")
