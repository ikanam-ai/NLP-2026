#!/usr/bin/env python3
from __future__ import annotations

import hashlib
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, classification_report


DATASET = "Tobi-Bueck/customer-support-tickets"


def read_idx(path: Path) -> np.ndarray:
    arr = np.loadtxt(path, dtype=np.int64)
    return arr


def md5(s: str) -> str:
    return hashlib.md5(s.encode("utf-8", errors="ignore")).hexdigest()


def split_df(df: pd.DataFrame, repo_root: Path) -> dict[str, pd.DataFrame]:
    train_idx = read_idx(repo_root / "data" / "train_idx.txt")
    val_idx = read_idx(repo_root / "data" / "val_idx.txt")
    test_idx = read_idx(repo_root / "data" / "test_idx.txt")

    out = {
        "train": df.iloc[train_idx].copy(),
        "val": df.iloc[val_idx].copy(),
        "test": df.iloc[test_idx].copy(),
    }
    return out


def basic_eda(splits: dict[str, pd.DataFrame]) -> None:
    print("\n=== SIZES ===")
    for k, d in splits.items():
        print(f"{k}: {len(d)}")

    print("\n=== MISSING VALUES (share) ===")
    cols = ["subject", "body", "queue", "priority", "type", "language"]
    for k, d in splits.items():
        miss = {c: float(d[c].isna().mean()) if c in d.columns else 1.0 for c in cols}
        print(k, {c: round(miss[c], 4) for c in cols})

    # text lengths
    for k, d in splits.items():
        txt = (d["subject"].fillna("") + "\n\n" + d["body"].fillna("")).astype(str)
        lens = txt.str.len()
        ws_tokens = txt.str.split().map(len)
        print(f"\n=== TEXT LENGTHS ({k}) ===")
        print(f"chars:  mean={lens.mean():.1f}  p50={lens.median():.0f}  p90={lens.quantile(0.9):.0f}  p99={lens.quantile(0.99):.0f}")
        print(f"words:  mean={ws_tokens.mean():.1f}  p50={ws_tokens.median():.0f}  p90={ws_tokens.quantile(0.9):.0f}  p99={ws_tokens.quantile(0.99):.0f}")

    # language dist
    print("\n=== LANGUAGE DISTRIBUTION ===")
    for k, d in splits.items():
        c = d["language"].fillna("NA").value_counts(normalize=True)
        print(k, {i: round(float(v), 4) for i, v in c.items()})

    # class coverage + tails
    for target in ["queue", "priority", "type"]:
        print(f"\n=== CLASS COVERAGE: {target} ===")
        train_labels = set(splits["train"][target].astype(str))
        for k, d in splits.items():
            labels = set(d[target].astype(str))
            missing_vs_train = sorted(list(train_labels - labels))
            print(f"{k}: classes={len(labels)}  missing_vs_train={len(missing_vs_train)}")

        # tail in train
        vc = splits["train"][target].astype(str).value_counts()
        print("train top-10:", list(zip(vc.head(10).index.tolist(), vc.head(10).tolist())))
        print("train bottom-10:", list(zip(vc.tail(10).index.tolist(), vc.tail(10).tolist())))


def duplicate_checks(splits: dict[str, pd.DataFrame]) -> None:
    print("\n=== DUPLICATE TEXT CHECKS (MD5 of subject+body) ===")
    hashes = {}
    for k, d in splits.items():
        txt = (d["subject"].fillna("") + "\n\n" + d["body"].fillna("")).astype(str)
        h = txt.map(md5)
        hashes[k] = set(h.tolist())
        dup_within = 1.0 - (len(hashes[k]) / len(d))
        print(f"{k}: unique={len(hashes[k])}/{len(d)}  within-dup-rate={dup_within:.4f}")

    inter_tv = len(hashes["train"] & hashes["val"])
    inter_tt = len(hashes["train"] & hashes["test"])
    inter_vt = len(hashes["val"] & hashes["test"])
    print(f"cross-dup hashes: train∩val={inter_tv}, train∩test={inter_tt}, val∩test={inter_vt}")

    if inter_tv or inter_tt or inter_vt:
        print("[WARN] Есть одинаковые тексты между сплитами (возможна утечка/слишком похожие письма).")
    else:
        print("[OK] Межсплитовых текстовых дублей не найдено.")


def baseline_tfidf_linear_svm(splits: dict[str, pd.DataFrame]) -> None:
    # Build text
    def make_text(d: pd.DataFrame) -> pd.Series:
        return (d["subject"].fillna("") + "\n\n" + d["body"].fillna("")).astype(str)

    X_train = make_text(splits["train"])
    X_test = make_text(splits["test"])

    vec = TfidfVectorizer(
        max_features=200_000,
        ngram_range=(1, 2),
        min_df=2,
    )
    Xtr = vec.fit_transform(X_train)
    Xte = vec.transform(X_test)

    results = {}

    print("\n=== BASELINE: TF-IDF + LinearSVC ===")
    for target in ["queue", "priority", "type"]:
        ytr = splits["train"][target].astype(str).values
        yte = splits["test"][target].astype(str).values

        clf = LinearSVC()
        clf.fit(Xtr, ytr)
        pred = clf.predict(Xte)

        acc = float(accuracy_score(yte, pred))
        if target == "queue":
            mf1 = float(f1_score(yte, pred, average="macro"))
            results["queue_acc"] = acc
            results["queue_macro_f1"] = mf1
            print(f"{target}: acc={acc:.4f}  macro_f1={mf1:.4f}")
        else:
            results[f"{target}_acc"] = acc
            print(f"{target}: acc={acc:.4f}")

    # final score (по вашей формуле)
    score = 0.70 * results["queue_macro_f1"] + 0.15 * results["priority_acc"] + 0.15 * results["type_acc"]
    print("\n=== FINAL SCORE (baseline) ===")
    print(f"Score = {score:.4f}  (0.70*MacroF1(queue) + 0.15*Acc(priority) + 0.15*Acc(type))")


def main():
    repo_root = Path(".").resolve()
    if not (repo_root / "data" / "train_idx.txt").exists():
        raise SystemExit("Запусти из корня репо, где есть папка data/ с *_idx.txt")

    print("Loading dataset:", DATASET)
    ds = load_dataset(DATASET)["train"]
    df = ds.to_pandas()

    # minimal columns check
    needed = {"subject", "body", "queue", "priority", "type", "language"}
    missing = sorted(list(needed - set(df.columns)))
    if missing:
        raise SystemExit(f"Dataset missing columns: {missing}")

    splits = split_df(df, repo_root)

    for k in splits:
        splits[k]["type"] = splits[k]["type"].fillna("Unknown")
    
    basic_eda(splits)
    duplicate_checks(splits)
    baseline_tfidf_linear_svm(splits)

    print("\n[OK] EDA + baseline done.")


if __name__ == "__main__":
    main()