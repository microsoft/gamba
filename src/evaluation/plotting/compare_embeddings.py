#!/usr/bin/env python3
import os
import math
import argparse
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import confusion_matrix

# ---------------- config ----------------

CATEGORIES = [
    "vista_enhancer",
    "UCNE",
    "repeats",
    "exons",
    "introns",
    "noncoding_regions",
    "coding_regions",
    "upstream_TSS",
    "UTR5",
    "UTR3",
    "promoters",
]

SCOPES = ["roi"]  # add "full" if/when needed

# nt / hyena / phyloGPN / caduceus-theirs models
NT_MODELS = [
    "hyenaDNA-random-init",
    "hyenaDNA",
    "phyloGPN",
    "nt-ms",
    "nt-human",
    "phyloGPN-random-init",
    "nt-ms-random-init",
    "nt-human-random-init",
    "caduceus-theirs",
    "caduceus-theirs-random-init",
    "evo2",
]

# ---------- roots for each task ----------

# random: feature vs random (pair-based npz with labels)
RANDOM_PAIRS_ROOT = "/home/mica/NucleotideTransformer/final_representations/random_pairs"

# upstream: feature vs upstream (pair-based npz with labels)
UPSTREAM_PAIRS_ROOT = "/home/mica/NucleotideTransformer/final_representations/upstream_pairs"

# baselines for random/upstream (already combined feature+control per category)
RANDOM_BASELINE_ROOT = (
    "/home/mica/gamba/data_processing/data/240-mammalian/"
    "final_representations/random_pairs/baseline"
)
UPSTREAM_BASELINE_ROOT = (
    "/home/mica/gamba/data_processing/data/240-mammalian/"
    "final_representations/upstream_pairs/baseline"
)


GLOBAL_RANDOM_ROOT = (
    "/home/mica/gamba/data_processing/data/240-mammalian/final_representations/random_pairs"
)
GLOBAL_UPSTREAM_ROOT = (
    "/home/mica/gamba/data_processing/data/240-mammalian/final_representations/upstream_pairs"
)

# multiclass: NT from upstream pair reps, gamba/caduceus + baselines from upstream_pairs global reps
NT_PAIRS_ROOT = "/home/mica/NucleotideTransformer/final_representations/upstream_pairs"
PAIR_GROUP_NAME = "all"
PAIR_LABEL_FILTER = "feature"  # which pair_label to treat as "category-bearing" in multiclass

# gamba / caduceus global models (now under upstream_pairs)
GLOBAL_MODELS = [
    "gamba_cons_only_ALLPOSstep_44000",
    "gamba_dual_ALLPOSstep_44000",
    "gamba_seq_only_ALLPOSstep_44000",
    "gamba_cons_only_step_random_init",
    "gamba_dual_step_random_init",
    "gamba_seq_only_step_random_init",
    "caduceus_cons_only_ALLPOSstep_44000",
    "caduceus_dual_ALLPOSstep_44000",
    "caduceus_seq_only_ALLPOSstep_44000",
    "caduceus_cons_only_step_random_init",
    "caduceus_dual_step_random_init",
    "caduceus_seq_only_step_random_init",
]

# new global root matches:
# /.../final_representations/upstream_pairs/<model>/<split>/<category>/reps_<short>_<split>_<category>_<scope>.npz
GLOBAL_ROOT = (
    "/home/mica/gamba/data_processing/data/240-mammalian/final_representations/upstream_pairs"
)
GLOBAL_SPLIT = "test"  # for non-random-init

# global baselines (multiclass) now under upstream_pairs/baseline with per-category dirs:
# /.../final_representations/upstream_pairs/baseline/kmer6/all/coding_regions/reps_kmer6_all_coding_regions_full_meta.parquet
GLOBAL_BASELINE_ROOT = (
    "/home/mica/gamba/data_processing/data/240-mammalian/final_representations/upstream_pairs/baseline"
)

BASELINE_MODELS = ["kmer6", "phylop"]


# ---------------- core metrics ----------------

def compute_ba_and_se(X: np.ndarray, y: np.ndarray):
    """
    LOO 1-NN balanced accuracy and SE (binary or multiclass).
    X: [N, D], y: [N]
    """
    X = np.asarray(X)
    y = np.asarray(y)

    if X.shape[0] < 3:
        raise ValueError(f"too few samples: {X.shape[0]}")

    nn = NearestNeighbors(n_neighbors=2, metric="cosine")
    nn.fit(X)
    _, idx = nn.kneighbors(X)
    # idx[:, 0] is self, idx[:, 1] is nearest neighbor excluding self
    y_pred = y[idx[:, 1]]

    classes = np.unique(y)
    cm = confusion_matrix(y, y_pred, labels=classes)
    n_per_class = cm.sum(axis=1).astype(float)
    correct_per_class = np.diag(cm).astype(float)

    with np.errstate(divide="ignore", invalid="ignore"):
        recalls = np.divide(
            correct_per_class,
            np.where(n_per_class == 0, 1.0, n_per_class),
        )

    K = len(classes)
    ba = float(np.nanmean(recalls))  # 0–1

    var = np.nansum(
        recalls * (1.0 - recalls) / np.where(n_per_class == 0, np.inf, n_per_class)
    ) / (K ** 2)
    se = math.sqrt(max(var, 0.0))

    return ba * 100.0, se * 100.0  # percent


def compute_ba_and_se_from_npz(npz_path: str):
    """
    baseline npz with 'embeddings' [N,D] and 'labels' [N] (e.g. feature vs random/upstream).
    """
    z = np.load(npz_path, allow_pickle=True)
    X = np.asarray(z["embeddings"])
    y = np.asarray(z["labels"])
    return compute_ba_and_se(X, y)


# ---------------- helpers ----------------

def infer_model_short(model_folder: str) -> str:
    if model_folder.startswith("gamba_"):
        return "gamba"
    if model_folder.startswith("caduceus_"):
        return "caduceus"
    return model_folder.split("_")[0]

def _load_global_pair_binary(
    root: str,
    model_folder: str,
    split: str,
    category: str,
    scope: str,
    pos_label: str,
    neg_label: str,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    generic loader for pair-based binary tasks for global (gamba/caduceus) models.

    expects npz:
      root/<model_folder>/<split>/<category>/
        reps_<short>_<split>_<category>_<scope>.npz

    where:
      embeddings: [N, D]
      labels: [N] with values including pos_label and neg_label
    """
    model_root = os.path.join(root, model_folder, split, category)
    if not os.path.isdir(model_root):
        print(f"[warn] missing global pair dir for model={model_folder}, split={split}, category={category}: {model_root}")
        return None, None

    short = infer_model_short(model_folder)
    base = os.path.join(model_root, f"reps_{short}_{split}_{category}_{scope}")
    npz_path = base + ".npz"
    if not os.path.exists(npz_path):
        print(f"[warn] missing global pair npz: {npz_path}")
        return None, None

    z = np.load(npz_path, allow_pickle=True)
    X = np.asarray(z["embeddings"])
    if "labels" not in z:
        print(f"[warn] global pair npz has no 'labels': {npz_path}")
        return None, None
    labels = np.asarray(z["labels"]).astype(str)

    mask_pos = labels == pos_label
    mask_neg = labels == neg_label

    if not np.any(mask_pos) or not np.any(mask_neg):
        print(
            f"[warn] {npz_path}: pos_label={pos_label} count={mask_pos.sum()}, "
            f"neg_label={neg_label} count={mask_neg.sum()}"
        )
        return None, None

    X_pos = X[mask_pos]
    X_neg = X[mask_neg]

    if X_pos.shape[0] < 3 or X_neg.shape[0] < 3:
        print(
            f"[warn] {npz_path}: too few samples "
            f"(pos={X_pos.shape[0]}, neg={X_neg.shape[0]})"
        )
        return None, None

    return X_pos, X_neg


def _load_pair_binary(
    root: str,
    model: str,
    category: str,
    scope: str,
    group_name: str,
    pos_label: str,
    neg_label: str,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    generic loader for pair-based binary tasks (feature vs random / upstream).

    expects npz:
      root/<model>/reps_<model>_<group_name>_<category>_<scope>.npz

    where:
      embeddings: [N, D]
      labels: [N] with values including pos_label and neg_label
    """
    model_dir = os.path.join(root, model)
    if not os.path.isdir(model_dir):
        print(f"[warn] missing pairs dir for model={model}: {model_dir}")
        return None, None

    base = os.path.join(model_dir, f"reps_{model}_{group_name}_{category}_{scope}")
    npz_path = base + ".npz"
    if not os.path.exists(npz_path):
        print(f"[warn] missing pair npz: {npz_path}")
        return None, None

    z = np.load(npz_path, allow_pickle=True)
    X = np.asarray(z["embeddings"])
    if "labels" not in z:
        print(f"[warn] npz has no 'labels': {npz_path}")
        return None, None
    labels = np.asarray(z["labels"]).astype(str)

    mask_pos = labels == pos_label
    mask_neg = labels == neg_label

    if not np.any(mask_pos) or not np.any(mask_neg):
        print(
            f"[warn] {npz_path}: pos_label={pos_label} count={mask_pos.sum()}, "
            f"neg_label={neg_label} count={mask_neg.sum()}"
        )
        return None, None

    X_pos = X[mask_pos]
    X_neg = X[mask_neg]

    if X_pos.shape[0] < 3 or X_neg.shape[0] < 3:
        print(
            f"[warn] {npz_path}: too few samples "
            f"(pos={X_pos.shape[0]}, neg={X_neg.shape[0]})"
        )
        return None, None

    return X_pos, X_neg


# ---------------- RANDOM: feature vs random ----------------

def collect_random_rows(group_name: str = "all"):
    rows = []

    # ---------- NT-style models, pair-based (NucleotideTransformer random_pairs) ----------
    for model in NT_MODELS:
        for scope in SCOPES:
            for cat in CATEGORIES:
                X_feat, X_rand = _load_pair_binary(
                    RANDOM_PAIRS_ROOT,
                    model,
                    cat,
                    scope,
                    group_name,
                    pos_label="feature",
                    neg_label="random",
                )
                if X_feat is None:
                    continue

                # balance classes
                n = min(X_feat.shape[0], X_rand.shape[0])
                if n < 5:
                    print(
                        f"[warn] RANDOM NT {model} {scope} {cat}: "
                        f"too few after balancing (n_feat={X_feat.shape[0]}, "
                        f"n_rand={X_rand.shape[0]})"
                    )
                    continue

                rng = np.random.default_rng(1337)
                idx_feat = rng.choice(X_feat.shape[0], size=n, replace=False)
                idx_rand = rng.choice(X_rand.shape[0], size=n, replace=False)

                X = np.concatenate([X_feat[idx_feat], X_rand[idx_rand]], axis=0)
                y = np.array(["feature"] * n + ["random"] * n, dtype=object)

                try:
                    ba, se = compute_ba_and_se(X, y)
                except Exception as e:
                    print(f"[skip] RANDOM NT {model} {scope} {cat}: {e}")
                    continue

                rows.append(
                    dict(
                        Model=model,
                        Family="NT_random",
                        Group=group_name,
                        Category=cat,
                        Scope=scope,
                        BA_pct=ba,
                        BA_SE_pct=se,
                        N_pos=int(n),
                        N_neg=int(n),
                    )
                )

    # ---------- gamba/caduceus global pair-based random ----------
    # only contributes if you have random_pairs/<model_folder>/<split>/<category>/...
    for model_folder in GLOBAL_MODELS:
        split = "all" if "random_init" in model_folder else GLOBAL_SPLIT
        for scope in SCOPES:
            for cat in CATEGORIES:
                model_root = os.path.join(GLOBAL_RANDOM_ROOT, model_folder, split, cat)
                if not os.path.isdir(model_root):
                    # no random pairs for this model/category – skip silently
                    continue

                short = infer_model_short(model_folder)
                base = os.path.join(
                    model_root,
                    f"reps_{short}_{split}_{cat}_{scope}",
                )
                npz_path = base + ".npz"
                if not os.path.exists(npz_path):
                    print(f"[warn] missing RANDOM global npz: {npz_path}")
                    continue

                z = np.load(npz_path, allow_pickle=True)
                X = np.asarray(z["embeddings"])
                if "labels" not in z:
                    print(f"[warn] RANDOM global npz has no 'labels': {npz_path}")
                    continue
                labels = np.asarray(z["labels"]).astype(str)

                if "random" not in labels:
                    print(f"[warn] RANDOM global {npz_path}: no 'random' label")
                    continue

                mask_neg = labels == "random"
                mask_pos = ~mask_neg

                if not np.any(mask_pos) or not np.any(mask_neg):
                    print(
                        f"[warn] RANDOM global {npz_path}: "
                        f"pos={mask_pos.sum()} neg={mask_neg.sum()}"
                    )
                    continue

                X_pos = X[mask_pos]
                X_neg = X[mask_neg]

                n = min(X_pos.shape[0], X_neg.shape[0])
                if n < 5:
                    print(
                        f"[warn] RANDOM global {model_folder} {split} {scope} {cat}: "
                        f"too few after balancing (pos={X_pos.shape[0]}, neg={X_neg.shape[0]})"
                    )
                    continue

                rng = np.random.default_rng(1337)
                idx_pos = rng.choice(X_pos.shape[0], size=n, replace=False)
                idx_neg = rng.choice(X_neg.shape[0], size=n, replace=False)

                X_bal = np.concatenate([X_pos[idx_pos], X_neg[idx_neg]], axis=0)
                y_bal = np.array(["feature"] * n + ["random"] * n, dtype=object)

                try:
                    ba, se = compute_ba_and_se(X_bal, y_bal)
                except Exception as e:
                    print(f"[skip] RANDOM global {model_folder} {split} {scope} {cat}: {e}")
                    continue

                rows.append(
                    dict(
                        Model=model_folder,
                        Family="gamba/caduceus_random",
                        Group=split,
                        Category=cat,
                        Scope=scope,
                        BA_pct=ba,
                        BA_SE_pct=se,
                        N_pos=int(n),
                        N_neg=int(n),
                    )
                )

    # ---------- baselines: per-category binary npz with labels ----------
    for model in BASELINE_MODELS:
        for scope in SCOPES:
            for cat in CATEGORIES:
                npz_path = os.path.join(
                    RANDOM_BASELINE_ROOT,
                    model,
                    "all",
                    cat,
                    f"reps_{model}_all_{cat}_{scope}.npz",
                )
                if not os.path.exists(npz_path):
                    print(f"[warn] missing RANDOM baseline npz: {npz_path}")
                    continue

                try:
                    ba, se = compute_ba_and_se_from_npz(npz_path)
                except Exception as e:
                    print(f"[skip] RANDOM baseline {model} {scope} {cat}: {e}")
                    continue

                rows.append(
                    dict(
                        Model=model,
                        Family="baseline_random",
                        Group="all",
                        Category=cat,
                        Scope=scope,
                        BA_pct=ba,
                        BA_SE_pct=se,
                        N_pos=np.nan,
                        N_neg=np.nan,
                    )
                )

    return rows


# ---------------- UPSTREAM: feature vs upstream ----------------

def collect_upstream_rows(group_name: str = "all"):
    rows = []

    # ---------- NT-style models, pair-based (NucleotideTransformer upstream_pairs) ----------
    for model in NT_MODELS:
        for scope in SCOPES:
            for cat in CATEGORIES:
                X_feat, X_up = _load_pair_binary(
                    UPSTREAM_PAIRS_ROOT,
                    model,
                    cat,
                    scope,
                    group_name,
                    pos_label="feature",
                    neg_label="upstream",
                )
                if X_feat is None:
                    continue

                n = min(X_feat.shape[0], X_up.shape[0])
                if n < 5:
                    print(
                        f"[warn] UPSTREAM NT {model} {scope} {cat}: "
                        f"too few after balancing (n_feat={X_feat.shape[0]}, "
                        f"n_up={X_up.shape[0]})"
                    )
                    continue

                rng = np.random.default_rng(1337)
                idx_feat = rng.choice(X_feat.shape[0], size=n, replace=False)
                idx_up = rng.choice(X_up.shape[0], size=n, replace=False)

                X = np.concatenate([X_feat[idx_feat], X_up[idx_up]], axis=0)
                y = np.array(["feature"] * n + ["upstream"] * n, dtype=object)

                try:
                    ba, se = compute_ba_and_se(X, y)
                except Exception as e:
                    print(f"[skip] UPSTREAM NT {model} {scope} {cat}: {e}")
                    continue

                rows.append(
                    dict(
                        Model=model,
                        Family="NT_upstream",
                        Group=group_name,
                        Category=cat,
                        Scope=scope,
                        BA_pct=ba,
                        BA_SE_pct=se,
                        N_pos=int(n),
                        N_neg=int(n),
                    )
                )

    # ---------- gamba/caduceus global pair-based upstream ----------
    for model_folder in GLOBAL_MODELS:
        split = "all" if "random_init" in model_folder else GLOBAL_SPLIT
        for scope in SCOPES:
            for cat in CATEGORIES:
                model_root = os.path.join(GLOBAL_UPSTREAM_ROOT, model_folder, split, cat)
                if not os.path.isdir(model_root):
                    # e.g. you may only have upstream_pairs for some models/categories
                    continue

                short = infer_model_short(model_folder)
                base = os.path.join(
                    model_root,
                    f"reps_{short}_{split}_{cat}_{scope}",
                )
                npz_path = base + ".npz"
                if not os.path.exists(npz_path):
                    print(f"[warn] missing UPSTREAM global npz: {npz_path}")
                    continue

                z = np.load(npz_path, allow_pickle=True)
                X = np.asarray(z["embeddings"])
                if "labels" not in z:
                    print(f"[warn] UPSTREAM global npz has no 'labels': {npz_path}")
                    continue
                labels = np.asarray(z["labels"]).astype(str)

                if "upstream" not in labels:
                    print(f"[warn] UPSTREAM global {npz_path}: no 'upstream' label")
                    continue

                mask_neg = labels == "upstream"
                mask_pos = ~mask_neg  # roi / feature

                if not np.any(mask_pos) or not np.any(mask_neg):
                    print(
                        f"[warn] UPSTREAM global {npz_path}: "
                        f"pos={mask_pos.sum()} neg={mask_neg.sum()}"
                    )
                    continue

                X_pos = X[mask_pos]
                X_neg = X[mask_neg]

                n = min(X_pos.shape[0], X_neg.shape[0])
                if n < 5:
                    print(
                        f"[warn] UPSTREAM global {model_folder} {split} {scope} {cat}: "
                        f"too few after balancing (pos={X_pos.shape[0]}, neg={X_neg.shape[0]})"
                    )
                    continue

                rng = np.random.default_rng(1337)
                idx_pos = rng.choice(X_pos.shape[0], size=n, replace=False)
                idx_neg = rng.choice(X_neg.shape[0], size=n, replace=False)

                X_bal = np.concatenate([X_pos[idx_pos], X_neg[idx_neg]], axis=0)
                y_bal = np.array(["feature"] * n + ["upstream"] * n, dtype=object)

                try:
                    ba, se = compute_ba_and_se(X_bal, y_bal)
                except Exception as e:
                    print(f"[skip] UPSTREAM global {model_folder} {split} {scope} {cat}: {e}")
                    continue

                rows.append(
                    dict(
                        Model=model_folder,
                        Family="gamba/caduceus_upstream",
                        Group=split,
                        Category=cat,
                        Scope=scope,
                        BA_pct=ba,
                        BA_SE_pct=se,
                        N_pos=int(n),
                        N_neg=int(n),
                    )
                )

    # ---------- baselines: per-category binary npz with labels ----------
    for model in BASELINE_MODELS:
        for scope in SCOPES:
            for cat in CATEGORIES:
                npz_path = os.path.join(
                    UPSTREAM_BASELINE_ROOT,
                    model,
                    "all",
                    cat,
                    f"reps_{model}_all_{cat}_{scope}.npz",
                )
                if not os.path.exists(npz_path):
                    print(f"[warn] missing UPSTREAM baseline npz: {npz_path}")
                    continue

                try:
                    ba, se = compute_ba_and_se_from_npz(npz_path)
                except Exception as e:
                    print(f"[skip] UPSTREAM baseline {model} {scope} {cat}: {e}")
                    continue

                rows.append(
                    dict(
                        Model=model,
                        Family="baseline_upstream",
                        Group="all",
                        Category=cat,
                        Scope=scope,
                        BA_pct=ba,
                        BA_SE_pct=se,
                        N_pos=np.nan,
                        N_neg=np.nan,
                    )
                )

    return rows


# ---------------- MULTICLASS: category vs category ----------------

def load_nt_pairs_multiclass(
    model: str,
    scope: str,
    group_name: str = PAIR_GROUP_NAME,
    pair_label_filter: Optional[str] = PAIR_LABEL_FILTER,
):
    """
    NT multiclass from upstream pair reps: filter to pair_label == 'feature'
    and use category as label.
    """
    model_dir = os.path.join(NT_PAIRS_ROOT, model)
    if not os.path.isdir(model_dir):
        print(f"[warn] missing NT pairs dir: {model_dir}")
        return None, None

    X_list = []
    y_list = []

    for cat in CATEGORIES:
        base = os.path.join(
            model_dir,
            f"reps_{model}_{group_name}_{cat}_{scope}",
        )
        npz_path = base + ".npz"
        if not os.path.exists(npz_path):
            print(f"[warn] missing NT pairs npz for {model} {cat} {scope}: {npz_path}")
            continue

        z = np.load(npz_path, allow_pickle=True)
        X_cat = np.asarray(z["embeddings"])

        labels_pair = None
        if "labels" in z:
            labels_pair = np.asarray(z["labels"]).astype(str)

        if labels_pair is not None and pair_label_filter is not None:
            mask = labels_pair == pair_label_filter
            if not np.any(mask):
                print(
                    f"[warn] NT {model} {cat} {scope}: "
                    f"no samples with pair_label == '{pair_label_filter}'"
                )
                continue
            X_cat = X_cat[mask]

        if X_cat.shape[0] == 0:
            continue

        X_list.append(X_cat)
        y_list.append(np.full(X_cat.shape[0], cat, dtype=object))

    if not X_list:
        print(f"[warn] NT {model} {scope}: no categories loaded from pairs")
        return None, None

    X_all = np.concatenate(X_list, axis=0)
    y_all = np.concatenate(y_list, axis=0)

    if X_all.shape[0] < 3:
        print(f"[warn] NT {model} {scope}: too few samples after combining categories")
        return None, None

    return X_all, y_all


def load_global_gamba_caduceus(model_folder: str, scope: str, split: str = "all"):
    """
    global gamba/caduceus multiclass from upstream_pairs layout:

      GLOBAL_ROOT/<model_folder>/<split>/<category>/
        reps_<short>_<split>_<category>_<scope>.npz
        reps_<short>_<split>_<category>_<scope>_meta.parquet
    """
    model_root = os.path.join(GLOBAL_ROOT, model_folder, split)
    if not os.path.isdir(model_root):
        print(f"[warn] missing global model dir: {model_root}")
        return None, None

    short = infer_model_short(model_folder)

    X_list = []
    y_list = []

    for cat in CATEGORIES:
        base = os.path.join(
            model_root,
            cat,
            f"reps_{short}_{split}_{cat}_{scope}",
        )
        npz_path = base + ".npz"
        meta_path = base + "_meta.parquet"

        if not os.path.exists(npz_path) or not os.path.exists(meta_path):
            print(f"[warn] missing global reps for {model_folder} {split} {cat} {scope}: {npz_path} / {meta_path}")
            continue

        z = np.load(npz_path, allow_pickle=True)
        X_cat = np.asarray(z["embeddings"])
        meta = pd.read_parquet(meta_path)

        # meta should all be this category, but we can still guard:
        if "category" in meta.columns:
            mask = meta["category"].isin(CATEGORIES).values
            X_cat = X_cat[mask]

        if X_cat.shape[0] == 0:
            continue

        X_list.append(X_cat)
        y_list.append(np.full(X_cat.shape[0], cat, dtype=object))

    if not X_list:
        print(f"[warn] global {model_folder} {split} {scope}: no categories loaded")
        return None, None

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)

    if X.shape[0] < 3:
        print(f"[warn] global {model_folder} {split} {scope}: too few samples after combining")
        return None, None

    return X, y


def load_global_baseline(model: str, scope: str):
    """
    global baseline multiclass from upstream_pairs layout:

      GLOBAL_BASELINE_ROOT/<model>/all/<category>/
        reps_<model>_all_<category>_<scope>.npz
        reps_<model>_all_<category>_<scope>_meta.parquet
    """
    X_list = []
    y_list = []

    for cat in CATEGORIES:
        base = os.path.join(
            GLOBAL_BASELINE_ROOT,
            model,
            "all",
            cat,
            f"reps_{model}_all_{cat}_{scope}",
        )
        npz_path = base + ".npz"
        meta_path = base + "_meta.parquet"

        if not os.path.exists(npz_path) or not os.path.exists(meta_path):
            print(f"[warn] missing global baseline reps for {model} {cat} {scope}: {npz_path} / {meta_path}")
            continue

        z = np.load(npz_path, allow_pickle=True)
        X_cat = np.asarray(z["embeddings"])
        meta = pd.read_parquet(meta_path)

        if "category" in meta.columns:
            mask = meta["category"].isin(CATEGORIES).values
            X_cat = X_cat[mask]

        if X_cat.shape[0] == 0:
            continue

        X_list.append(X_cat)
        y_list.append(np.full(X_cat.shape[0], cat, dtype=object))

    if not X_list:
        print(f"[warn] global baseline {model} {scope}: no categories loaded")
        return None, None

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)

    if X.shape[0] < 3:
        print(f"[warn] global baseline {model} {scope}: too few samples after combining")
        return None, None

    return X, y


def collect_multiclass_rows():
    rows = []

    # NT-style from pair reps
    for model in NT_MODELS:
        for scope in SCOPES:
            X, y = load_nt_pairs_multiclass(model, scope)
            if X is None:
                continue
            try:
                ba, se = compute_ba_and_se(X, y)
            except Exception as e:
                print(f"[skip] MULTICLASS NT_pairs {model} {scope}: {e}")
                continue

            rows.append(
                dict(
                    Model=model,
                    Family="NT_pairs_multiclass",
                    Group=PAIR_GROUP_NAME,
                    Scope=scope,
                    BA_pct=ba,
                    BA_SE_pct=se,
                )
            )

    # gamba/caduceus global
    for model_folder in GLOBAL_MODELS:
        split = "all" if "random_init" in model_folder else GLOBAL_SPLIT
        for scope in SCOPES:
            X, y = load_global_gamba_caduceus(model_folder, scope, split=split)
            if X is None:
                continue
            try:
                ba, se = compute_ba_and_se(X, y)
            except Exception as e:
                print(f"[skip] MULTICLASS global {model_folder} {split} {scope}: {e}")
                continue

            rows.append(
                dict(
                    Model=model_folder,
                    Family="gamba/caduceus_multiclass",
                    Group=split,
                    Scope=scope,
                    BA_pct=ba,
                    BA_SE_pct=se,
                )
            )

    # global baselines
    for model in BASELINE_MODELS:
        for scope in SCOPES:
            X, y = load_global_baseline(model, scope)
            if X is None:
                continue
            try:
                ba, se = compute_ba_and_se(X, y)
            except Exception as e:
                print(f"[skip] MULTICLASS baseline {model} {scope}: {e}")
                continue

            rows.append(
                dict(
                    Model=model,
                    Family="baseline_multiclass",
                    Group="all",
                    Scope=scope,
                    BA_pct=ba,
                    BA_SE_pct=se,
                )
            )

    return rows


# ---------------- aggregation ----------------

def aggregate_per_category(df: pd.DataFrame) -> pd.DataFrame:
    summaries = []
    for (model, family, group, scope), sub in df.groupby(["Model", "Family", "Group", "Scope"]):
        ba_vals = sub["BA_pct"].to_numpy()
        se_vals = sub["BA_SE_pct"].to_numpy()
        K = len(ba_vals)
        if K == 0:
            continue

        ba_global = float(np.mean(ba_vals))
        var_i = (se_vals / 100.0) ** 2
        var_global = float(np.sum(var_i) / (K ** 2))
        se_global = math.sqrt(max(var_global, 0.0)) * 100.0
        std_across_cats = float(np.std(ba_vals, ddof=1)) if K > 1 else 0.0

        summaries.append(
            dict(
                Model=model,
                Family=family,
                Group=group,
                Scope=scope,
                N_Categories=K,
                GlobalBalancedAccuracyPct=ba_global,
                GlobalBalancedAccuracyStdPct=std_across_cats,
                GlobalBalancedAccuracySEPct=se_global,
            )
        )
    return pd.DataFrame(summaries)


def aggregate_multiclass(df: pd.DataFrame) -> pd.DataFrame:
    summaries = []
    for (model, family, group, scope), sub in df.groupby(["Model", "Family", "Group", "Scope"]):
        ba_vals = sub["BA_pct"].to_numpy()
        se_vals = sub["BA_SE_pct"].to_numpy()
        K = len(ba_vals)
        if K == 0:
            continue

        ba_global = float(np.mean(ba_vals))
        var_i = (se_vals / 100.0) ** 2
        var_global = float(np.sum(var_i) / (K ** 2))
        se_global = math.sqrt(max(var_global, 0.0)) * 100.0
        std_across_runs = float(np.std(ba_vals, ddof=1)) if K > 1 else 0.0

        summaries.append(
            dict(
                Model=model,
                Family=family,
                Group=group,
                Scope=scope,
                N_Runs=K,
                GlobalBalancedAccuracyPct=ba_global,
                GlobalBalancedAccuracyStdPct=std_across_runs,
                GlobalBalancedAccuracySEPct=se_global,
            )
        )
    return pd.DataFrame(summaries)


# ---------------- main ----------------

def main():
    ap = argparse.ArgumentParser(
        description=(
            "compare embeddings via LOO 1-NN BA with modes:\n"
            "  - multiclass: Category vs all other categories\n"
            "  - random: feature vs random (per-category, pair-based)\n"
            "  - upstream: feature vs upstream (per-category, pair-based)"
        )
    )
    ap.add_argument(
        "--task",
        choices=["multiclass", "random", "upstream"],
        required=True,
        help="comparison type",
    )
    ap.add_argument(
        "-o",
        "--outdir",
        default="/home/mica/gamba/data_processing/data/240-mammalian/global_balacc_combined",
        help="output directory for TSVs",
    )
    ap.add_argument("--group_name", type=str, default="all")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    if args.task == "random":
        rows = collect_random_rows(group_name=args.group_name)
        #rows = collect_random_rows()
        if not rows:
            raise SystemExit("no rows collected for random; check paths")
        df = pd.DataFrame(rows)
        percat_path = os.path.join(args.outdir, "balacc_random_per_category.tsv")
        df.to_csv(percat_path, sep="\t", index=False)
        print(f"[info] wrote per-category RANDOM table: {percat_path}")

        df_global = aggregate_per_category(df)
        global_path = os.path.join(args.outdir, "balacc_random_global.tsv")
        df_global.to_csv(global_path, sep="\t", index=False)
        print(f"[info] wrote global RANDOM table: {global_path}")

    elif args.task == "upstream":
        rows = collect_upstream_rows(group_name=args.group_name)
        #rows = collect_upstream_rows()
        if not rows:
            raise SystemExit("no rows collected for upstream; check paths")
        df = pd.DataFrame(rows)
        percat_path = os.path.join(args.outdir, "balacc_upstream_per_category.tsv")
        df.to_csv(percat_path, sep="\t", index=False)
        print(f"[info] wrote per-category UPSTREAM table: {percat_path}")

        df_global = aggregate_per_category(df)
        global_path = os.path.join(args.outdir, "balacc_upstream_global.tsv")
        df_global.to_csv(global_path, sep="\t", index=False)
        print(f"[info] wrote global UPSTREAM table: {global_path}")

    else:  # multiclass
        rows = collect_multiclass_rows()
        if not rows:
            raise SystemExit("no rows collected for multiclass; check paths")
        df = pd.DataFrame(rows)
        per_model_path = os.path.join(args.outdir, "balacc_multiclass_per_model.tsv")
        df.to_csv(per_model_path, sep="\t", index=False)
        print(f"[info] wrote per-model MULTICLASS table: {per_model_path}")

        df_global = aggregate_multiclass(df)
        global_path = os.path.join(args.outdir, "balacc_multiclass_global.tsv")
        df_global.to_csv(global_path, sep="\t", index=False)
        print(f"[info] wrote aggregated MULTICLASS table: {global_path}")


if __name__ == "__main__":
    main()
