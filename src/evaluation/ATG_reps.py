#!/usr/bin/env python3
import argparse
import os
import json
import logging
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import pyBigWig
from pyfaidx import Fasta
from tqdm import tqdm

import umap
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    cohen_kappa_score,
    matthews_corrcoef,
)

import sys
sys.path.append("../gamba")
sys.path.append("/home/mica/gamba/")

from src.evaluation.utils.helpers import extract_context
from src.evaluation.utils.specific_helpers import load_model, predict_scores_batched


# ---------------- logging ----------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


# ---------------- KNN + metrics helpers ----------------

def loo_1nn_predictions(embeddings, labels):
    labels = np.asarray(labels)
    X = np.asarray(embeddings)
    nn = NearestNeighbors(n_neighbors=2, metric="euclidean").fit(X)
    _, indices = nn.kneighbors(X)
    y_true = labels
    y_pred = labels[indices[:, 1]]
    return y_true, y_pred


def eval_metrics(y_true, y_pred, label_order=None):
    if label_order is None:
        label_order = np.unique(y_true)

    cm = confusion_matrix(y_true, y_pred, labels=label_order)
    row_sums = cm.sum(axis=1, keepdims=True)
    per_class_recall = np.diag(cm) / np.where(row_sums == 0, 1, row_sums).squeeze()

    valid = ~np.isnan(per_class_recall)
    ba = float(np.mean(per_class_recall[valid]))
    sem = float(np.std(per_class_recall[valid], ddof=1) / np.sqrt(np.sum(valid)))
    ci95 = float(1.96 * sem)

    metrics = {
        "micro_accuracy": float((y_true == y_pred).mean()),
        "balanced_accuracy": ba,
        "balanced_accuracy_sem": sem,
        "balanced_accuracy_ci95": ci95,
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=label_order, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(y_true, y_pred, labels=label_order, average="weighted", zero_division=0)
        ),
        "cohens_kappa": float(cohen_kappa_score(y_true, y_pred, labels=label_order)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "per_class_recall": dict(zip(label_order, per_class_recall.astype(float))),
        "support": dict(zip(label_order, cm.sum(axis=1).astype(int))),
    }
    return cm, metrics, label_order


def plot_knn_heatmap(embeddings, labels, output_path, title="1-NN"):
    if len(embeddings) == 0:
        logging.warning("[plot_knn_heatmap] no embeddings to plot")
        return None, None, None

    labels = np.asarray(labels)
    present = sorted(set(labels))
    y_true, y_pred = loo_1nn_predictions(embeddings, labels)
    cm, metrics, label_order = eval_metrics(y_true, y_pred, label_order=present)

    with np.errstate(invalid="ignore", divide="ignore"):
        acc_matrix = cm.astype(float) / np.where(
            cm.sum(axis=1, keepdims=True) == 0,
            1,
            cm.sum(axis=1, keepdims=True),
        )

    plt.figure(figsize=(5, 4))
    sns.heatmap(
        acc_matrix,
        xticklabels=label_order,
        yticklabels=label_order,
        vmin=0,
        vmax=1.0,
        cmap="Blues",
        annot=True,
        fmt=".2f",
        cbar_kws={"label": "per-class recall"},
    )
    plt.title(
        f"{title}\n"
        f"micro={metrics['micro_accuracy']:.2%} | "
        f"balanced={metrics['balanced_accuracy']:.2%} | "
        f"macro-F1={metrics['macro_f1']:.2%}"
    )
    plt.xlabel("predicted")
    plt.ylabel("true")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    logging.info(
        f"[KNN] {title} | micro={metrics['micro_accuracy']:.3f}, "
        f"balanced={metrics['balanced_accuracy']:.3f}, "
        f"macroF1={metrics['macro_f1']:.3f}, "
        f"weightedF1={metrics['weighted_f1']:.3f}, "
        f"kappa={metrics['cohens_kappa']:.3f}, "
        f"mcc={metrics['mcc']:.3f}"
    )
    return metrics, label_order, acc_matrix


# ---------------- NEW: saving reps ----------------

def save_reps(base_dir, model_tag, name, X, labels, metas, extra=None):
    """
    save embeddings + labels to npz and metadata to parquet.

    base_dir / f"reps_{model_tag}_{name}.npz"
    base_dir / f"reps_{model_tag}_{name}_meta.parquet"
    """
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    X = np.asarray(X, dtype=np.float32)
    labels = np.asarray(labels)

    prefix = f"reps_{model_tag}_{name}"
    np.savez_compressed(
        base_dir / f"{prefix}.npz",
        embeddings=X,
        labels=labels,
    )

    mdf = pd.DataFrame(metas)
    if "label" in mdf.columns:
        mdf["label"] = labels
    else:
        mdf.insert(0, "label", labels)

    if extra:
        for k, v in extra.items():
            mdf[k] = v

    mdf.to_parquet(base_dir / f"{prefix}_meta.parquet", index=False)


# ---------------- ATG context loading ----------------

def load_atg_contexts(
    atg_tsv_dir,
    bigwig_file,
    genome,
    model_type,
    chromosomes,
    max_examples=None,
):
    """
    load atg codon contexts for gamba/caduceus.

    for each transcript row with all label2–5 present:
      - build regions (start, start+3) for labels 1..5
      - call extract_context for each
      - keep example only if all 5 contexts succeed

    returns:
      contexts: list[dict], each with sequence, feature_start_in_window, feature_end_in_window,
                plus example_id, label_id
    """
    import glob

    bw = pyBigWig.open(bigwig_file)

    contexts = []
    n_examples = 0

    label_cols = {
        1: "label1_pos",
        2: "label2_pos_noncoding_near",
        3: "label3_pos_noncoding_far",
        4: "label4_pos_coding_near",
        5: "label5_pos_coding_far",
    }

    for chrom in chromosomes:
        pattern = os.path.join(atg_tsv_dir, f"{chrom}_atg_labels.tsv")
        matches = glob.glob(pattern)
        if not matches:
            logging.warning(f"no atg tsv for {chrom} at {pattern}")
            continue
        tsv = matches[0]
        logging.info(f"loading {tsv}")
        df = pd.read_csv(tsv, sep="\t")

        # require all four label pairs (2–5) to be present
        mask_all = (
            (df["label2_pos_noncoding_near"] != ".")
            & (df["label3_pos_noncoding_far"] != ".")
            & (df["label4_pos_coding_near"] != ".")
            & (df["label5_pos_coding_far"] != ".")
        )
        df = df[mask_all].copy()
        logging.info(f"{chrom}: {len(df)} transcripts with all label pairs")

        for _, row in df.iterrows():
            try:
                pos_dict = {
                    lid: int(row[col])
                    for lid, col in label_cols.items()
                }
            except ValueError:
                # malformed numeric field
                continue

            example_id = f"{row['chrom']}|{row['tx_id']}|{row['strand']}|{row['label1_pos']}"
            example_contexts = []
            ok = True

            for lid, pos in pos_dict.items():
                region = {
                    "chrom": row["chrom"],
                    "start": pos,
                    "end": pos + 3,  # codon span
                    "feature_id": f"{row['tx_id']}_L{lid}",
                }

                ctx = extract_context(
                    bigwig_file,
                    region,
                    genome,
                    model_type,  # "gamba" or "caduceus"
                )
                if not ctx or "sequence" not in ctx:
                    ok = False
                    break

                ctx["example_id"] = example_id
                ctx["label_id"] = lid
                # keep raw deltas as optional meta
                if lid == 2:
                    ctx["delta_bp"] = int(row["label2_delta_bp"])
                elif lid == 3:
                    ctx["delta_bp"] = int(row["label3_delta_bp"])
                elif lid == 4:
                    ctx["delta_bp"] = int(row["label4_delta_bp"])
                elif lid == 5:
                    ctx["delta_bp"] = int(row["label5_delta_bp"])
                else:
                    ctx["delta_bp"] = 0

                example_contexts.append(ctx)

            if not ok:
                continue

            contexts.extend(example_contexts)
            n_examples += 1

            if max_examples is not None and n_examples >= max_examples:
                bw.close()
                logging.info(f"reached max_examples={max_examples}")
                return contexts

    bw.close()
    logging.info(f"total examples with all labels: {n_examples}")
    logging.info(f"total contexts (5 per example): {len(contexts)}")
    return contexts


# ---------------- embedding (gamba / caduceus) ----------------

def compute_atg_roi_embeddings(
    model,
    tokenizer,
    contexts,
    batch_size,
    device,
    model_type,
    training_task,
):
    """
    run predict_scores_batched on ATG contexts and pool ROI (codon) embeddings.

    returns:
      roi_embeds [N, H]
      label_ids [N] (1..5)
      metas: list[dict] with example_id, label_id, chrom, start, end, delta_bp
    """
    logging.info(
        f"computing atg roi embeddings for {len(contexts)} contexts, "
        f"model_type={model_type}, task={training_task}"
    )

    seq_reps, region_info = predict_scores_batched(
        model,
        tokenizer,
        contexts,
        batch_size=batch_size,
        device=device,
        model_type=model_type,
        training_task=training_task,
    )

    assert len(seq_reps) == len(region_info) == len(contexts)
    for ctx, info in zip(contexts, region_info):
        info["example_id"] = ctx["example_id"]
        info["label_id"] = ctx["label_id"]
        info["chrom"] = ctx.get("chrom", info.get("chrom", None))
        info["start"] = ctx.get("start", info.get("start", -1))
        info["end"] = ctx.get("end", info.get("end", -1))
        info["delta_bp"] = ctx.get("delta_bp", 0)

    roi_embeds = []
    label_ids = []
    metas = []

    for rep, info in zip(seq_reps, region_info):
        rep = np.asarray(rep, dtype=np.float32)
        if rep.ndim != 2:
            continue

        fs = int(info.get("feature_start_in_window", 0))
        fe = int(info.get("feature_end_in_window", rep.shape[0]))
        if fe <= fs or fs < 0 or fe > rep.shape[0]:
            continue

        rep_slice = rep[fs:fe]
        if rep_slice.shape[0] == 0:
            continue

        pooled = rep_slice.mean(axis=0)

        lid = int(info["label_id"])
        ex_id = info["example_id"]

        roi_embeds.append(pooled.astype(np.float32))
        label_ids.append(lid)
        metas.append(
            {
                "example_id": ex_id,
                "label_id": lid,
                "chrom": info.get("chrom"),
                "start": int(info.get("start", -1)),
                "end": int(info.get("end", -1)),
                "delta_bp": int(info.get("delta_bp", 0)),
                "feature_start_in_window": fs,
                "feature_end_in_window": fe,
            }
        )

    if len(roi_embeds) == 0:
        logging.error("no roi embeddings produced")
        return np.empty((0, 1), dtype=np.float32), np.array([]), []

    roi_embeds = np.stack(roi_embeds)
    label_ids = np.asarray(label_ids, dtype=int)
    logging.info(f"roi_embeds shape={roi_embeds.shape}, n={len(label_ids)}")
    return roi_embeds, label_ids, metas



def plot_binary_knn(embeddings, labels, output_path, title):
    if len(embeddings) == 0:
        logging.warning(f"[KNN] no embeddings for {title}")
        return None, None, None

    y_true, y_pred = loo_1nn_predictions(embeddings, labels)
    present = sorted(set(labels))
    cm, metrics, label_order = eval_metrics(y_true, y_pred, label_order=present)

    with np.errstate(invalid="ignore", divide="ignore"):
        acc_matrix = cm.astype(float) / np.where(
            cm.sum(axis=1, keepdims=True) == 0,
            1,
            cm.sum(axis=1, keepdims=True),
        )

    plt.figure(figsize=(5, 4))
    sns.heatmap(
        acc_matrix,
        xticklabels=label_order,
        yticklabels=label_order,
        vmin=0,
        vmax=1,
        cmap="Blues",
        annot=True,
        fmt=".2f",
        cbar_kws={"label": "per-class recall"},
    )
    plt.title(title)
    plt.xlabel("predicted")
    plt.ylabel("true")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    logging.info(
        f"[KNN] {title} | micro={metrics['micro_accuracy']:.3f}, "
        f"balanced={metrics['balanced_accuracy']:.3f}, "
        f"macroF1={metrics['macro_f1']:.3f}, "
        f"weightedF1={metrics['weighted_f1']:.3f}, "
        f"kappa={metrics['cohens_kappa']:.3f}, "
        f"mcc={metrics['mcc']:.3f}"
    )
    return metrics, label_order, acc_matrix



# ---------------- main ATG analysis (5 tasks) ----------------

def analyze_atg_pairs_gamba_caduceus(
    atg_tsv_dir,
    genome_fasta,
    bigwig_file,
    checkpoint_dir,
    config_fpath,
    output_dir,
    model_type,
    training_task,
    last_step,
    chromosomes,
    batch_size,
    max_examples=None,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"using device: {device}")

    # load model
    model, tokenizer = load_model(
        checkpoint_dir,
        config_fpath,
        last_step=last_step,
        device=device,
        training_task=training_task,
        model_type=model_type,
    )

    genome = Fasta(genome_fasta)

    # load atg contexts
    contexts = load_atg_contexts(
        atg_tsv_dir=atg_tsv_dir,
        bigwig_file=bigwig_file,
        genome=genome,
        model_type=model_type,
        chromosomes=chromosomes,
        max_examples=max_examples,
    )
    if not contexts:
        logging.error("no contexts loaded, aborting")
        return

    # embed
    roi_embeds, label_ids, metas = compute_atg_roi_embeddings(
        model,
        tokenizer,
        contexts,
        batch_size=batch_size,
        device=device,
        model_type=model_type,
        training_task=training_task,
    )
    if roi_embeds.shape[0] == 0:
        logging.error("empty embeddings, aborting")
        return

    # index embeddings by example_id + label_id
    index_by_example = defaultdict(dict)
    for i, meta in enumerate(metas):
        ex_id = meta["example_id"]
        lid = int(meta["label_id"])
        index_by_example[ex_id][lid] = i

    valid_examples = [
        ex_id
        for ex_id, lids in index_by_example.items()
        if all(l in lids for l in (1, 2, 3, 4, 5))
    ]
    logging.info(f"valid examples with all 5 labels after embedding: {len(valid_examples)}")

    if len(valid_examples) == 0:
        logging.error("no valid examples with all 5 labels, aborting")
        return

    # model tag for saving
    if last_step == 0:
        step_tag = "random_init"
    else:
        step_tag = str(last_step)
    model_tag = f"{model_type}_{training_task}_step{step_tag}"

    # save all roi embeddings (labels 1–5)
    extra_all = {
        "model_type": model_type,
        "training_task": training_task,
        "last_step": last_step,
        "scope": "roi_all",
    }
    save_reps(output_dir, model_tag, "ATG_all_labels", roi_embeds, label_ids, metas, extra=extra_all)

    # helper: build binary task + track indices for metas
    def build_binary_task(functional_label, other_label, other_name):
        X = []
        y = []
        idxs = []
        for ex in valid_examples:
            i_func = index_by_example[ex][functional_label]
            i_other = index_by_example[ex][other_label]
            X.append(roi_embeds[i_func])
            y.append("start_codon")  # label1
            idxs.append(i_func)
            X.append(roi_embeds[i_other])
            y.append(other_name)
            idxs.append(i_other)
        return np.stack(X), np.array(y), idxs

    # task 1: label1 vs label2 (noncoding close – functional)
    X1, y1, _ = build_binary_task(1, 2, "c_vs_nc_close")

    # task 2: label1 vs label3 (noncoding far – proximity)
    X2, y2, _ = build_binary_task(1, 3, "c_vs_nc_far")

    # task 3: label1 vs label4 (coding close – functional)
    X3, y3, _ = build_binary_task(1, 4, "c_vs_c_close")

    # task 4: label1 vs label5 (coding far – proximity)
    X4, y4, _= build_binary_task(1, 5, "c_vs_c_far")

    # task 5: coding vs non-coding (labels 1,4,5 vs 2,3), all labels
    X5 = []
    y5 = []
    for ex in valid_examples:
        for lid in (1, 2, 3, 4, 5):
            idx = index_by_example[ex][lid]
            X5.append(roi_embeds[idx])
            if lid in (1, 4, 5):
                y5.append("coding")
            else:
                y5.append("non-coding")
    X5 = np.stack(X5)
    y5 = np.array(y5)

    # run knn + plots for each task
    task_metrics = {}

    metrics1, _, _ = plot_binary_knn(
        X1,
        y1,
        output_dir / f"knn_{model_type}_task1_c_vs_nc_close.png",
        title="task1: start vs noncoding close",
    )
    task_metrics["task1"] = metrics1["balanced_accuracy"] if metrics1 else np.nan

    metrics2, _, _ = plot_binary_knn(
        X2,
        y2,
        output_dir / f"knn_{model_type}_task2_c_vs_nc_far.png",
        title="task2: start vs noncoding far",
    )
    task_metrics["task2"] = metrics2["balanced_accuracy"] if metrics2 else np.nan

    metrics3, _, _ = plot_binary_knn(
        X3,
        y3,
        output_dir / f"knn_{model_type}_task3_c_vs_c_close.png",
        title="task3: start vs coding close",
    )
    task_metrics["task3"] = metrics3["balanced_accuracy"] if metrics3 else np.nan

    metrics4, _, _ = plot_binary_knn(
        X4,
        y4,
        output_dir / f"knn_{model_type}_task4_c_vs_c_far.png",
        title="task4: start vs coding far",
    )
    task_metrics["task4"] = metrics4["balanced_accuracy"] if metrics4 else np.nan

    metrics5, _, _ = plot_binary_knn(
        X5,
        y5,
        output_dir / f"knn_{model_type}_task5_c_vs_nc.png",
        title="task5: functional (1,4,5) vs proximity (2,3)",
    )
    task_metrics["task5"] = metrics5["balanced_accuracy"] if metrics5 else np.nan

    # save metrics json
    with open(output_dir / f"balanced_accuracy_{model_type}_tasks.json", "w") as f:
        json.dump(task_metrics, f, indent=2)

    # bar plot across tasks
    task_order = ["task1", "task2", "task3", "task4", "task5"]
    task_labels = [
        "1: coding v noncoding (close)",
        "2: coding v noncoding (far)",
        "3: coding vs coding (close)",
        "4: coding vs coding (far)",
        "5: coding vs noncoding",
    ]
    bas = [task_metrics.get(t, np.nan) for t in task_order]

    plt.figure(figsize=(7, 4))
    sns.barplot(x=task_labels, y=bas)
    plt.ylabel("balanced accuracy")
    plt.ylim(0, 1)
    plt.xticks(rotation=25, ha="right")
    plt.title(f"balanced accuracy per ATG task ({model_type}, {training_task})")
    plt.tight_layout()
    plt.savefig(output_dir / f"balanced_accuracy_{model_tag}_tasks_bar.png", dpi=300)
    plt.close()

    logging.info(f"task balanced accuracies: {task_metrics}")


# ---------------- cli ----------------

def main():
    parser = argparse.ArgumentParser(
        description="ATG codon representation tasks for gamba / caduceus (5 tasks, 1-NN)"
    )
    parser.add_argument(
        "--atg_tsv_dir",
        type=str,
        default='/home/mica/NucleotideTransformer/ATGs/',
        help="dir with chr*_atg_labels.tsv files",
    )
    parser.add_argument(
        "--bigwig_file",
        type=str,
        default="/home/mica/gamba/data_processing/data/240-mammalian/241-mammalian-2020v2.bigWig",
    )
    parser.add_argument(
        "--genome_fasta",
        type=str,
        default="/home/mica/gamba/data_processing/data/240-mammalian/hg38.ml.fa",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="/home/mica/gamba/",
        help="base checkpoint dir (for gamba this is the root; we add clean_dcps/CCP/)",
    )
    parser.add_argument(
        "--config_fpath",
        type=str,
        default="/home/mica/gamba/configs/jamba-small-240mammalian.json",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/mica/gamba/data_processing/data/240-mammalian/ATG_reps",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["gamba", "caduceus"],
        required=True,
    )
    parser.add_argument(
        "--training_task",
        type=str,
        choices=["dual", "cons_only", "seq_only"],
        required=True,
    )
    parser.add_argument(
        "--last_step",
        type=int,
        default=44000,
        help="checkpoint step (0 = random init)",
    )
    parser.add_argument(
        "--chromosomes",
        type=str,
        nargs="+",
        default=[
            "chr1", "chr2", "chr3", "chr4", "chr5", "chr6",
            "chr7", "chr8", "chr9", "chr10", "chr11", "chr12",
            "chr13", "chr14", "chr15", "chr16", "chr17", "chr18",
            "chr19", "chr20", "chr21", "chr22", "chrX",
        ],
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=None,
        help="optional cap on number of label1 examples (each contributes 5 contexts)",
    )

    args = parser.parse_args()

    # checkpoint subdir for gamba, like upstream script
    if args.model_type == "gamba":
        checkpoint_dir = os.path.join(args.checkpoint_dir, "clean_dcps/CCP/")
    else:
        checkpoint_dir = args.checkpoint_dir

    if args.last_step == 0:
        last_tag = "random_init"
    else:
        last_tag = args.last_step

    outdir = os.path.join(
        args.output_dir,
        f"ATG_{args.model_type}_{args.training_task}_step_{last_tag}",
    )
    os.makedirs(outdir, exist_ok=True)

    logging.info(f"writing outputs to {outdir}")
    logging.info(f"using checkpoint_dir={checkpoint_dir}")

    analyze_atg_pairs_gamba_caduceus(
        atg_tsv_dir=args.atg_tsv_dir,
        genome_fasta=args.genome_fasta,
        bigwig_file=args.bigwig_file,
        checkpoint_dir=checkpoint_dir,
        config_fpath=args.config_fpath,
        output_dir=outdir,
        model_type=args.model_type,
        training_task=args.training_task,
        last_step=args.last_step,
        chromosomes=args.chromosomes,
        batch_size=args.batch_size,
        max_examples=args.max_examples,
    )


if __name__ == "__main__":
    main()
