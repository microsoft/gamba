#!/usr/bin/env python3
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import pyBigWig
from pyfaidx import Fasta
import json
from tqdm import tqdm
import pandas as pd
import logging
import sys
import random
import seaborn as sns
from pathlib import Path
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats
sys.path.append("../gamba")
sys.path.append("/home/mica/gamba/")
from torch.nn import MSELoss, CrossEntropyLoss
from sequence_models.constants import MSA_PAD, START, STOP
from evodiff.utils import Tokenizer
from gamba.constants import TaskType, DNA_ALPHABET_PLUS
from gamba.collators import gLMCollator, gLMMLMCollator
from gamba.model import create_model, JambagambaModel, JambaGambaNoConsModel, JambaGambaNOALMModel
from my_caduceus.configuration_caduceus import CaduceusConfig
from my_caduceus.modeling_caduceus import (
    CaduceusConservationForMaskedLM,
    CaduceusForMaskedLM,
    CaduceusConservation
)
from src.evaluation.utils.helpers import load_bed_file, extract_context 
from src.evaluation.utils.specific_helpers import load_model, predict_scores_batched


from Bio.Seq import Seq
#import Counter
from collections import Counter

import umap
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.neighbors import NearestNeighbors


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


def extract_region_embeddings(representations, region_info, mode="roi"):
    """
    Extract embeddings for the region of interest (ROI) or full sequence.

    Args:
        representations: list of np.arrays of shape (seq_len, hidden_dim)
        region_info: list of dicts with keys 'feature_start_in_window', 'feature_end_in_window', 'category'
        mode: "roi" for region-of-interest, "full" for full-sequence average

    Returns:
        List of pooled embeddings and corresponding category labels
    """
    embeddings = []
    labels = []

    for rep, info in zip(representations, region_info):
        if isinstance(rep, float) or np.isnan(rep).any():
            continue

        if mode == "roi":
            fs = info["feature_start_in_window"]
            fe = info["feature_end_in_window"]
            rep_slice = rep[fs:fe]
        elif mode == "full":
            rep_slice = rep
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        if rep_slice.shape[0] == 0:
            continue

        pooled = rep_slice.mean(axis=0)  # shape: (hidden_dim,)
        embeddings.append(pooled)
        labels.append(info.get("category", "unknown"))
    print(f"[extract_region_embeddings] mode={mode}, returning {len(embeddings)} embeds and {len(labels)} labels")

    return np.stack(embeddings), labels


def plot_umap(embeddings, labels, output_path, title="UMAP of Representations"):
    if len(embeddings) != len(labels):
        print(f"[ERROR] plot_umap: {len(embeddings)} embeddings vs {len(labels)} labels")
        return

    if len(embeddings) == 0:
        print("[WARNING] No embeddings to plot")
        return
    umap_model = umap.UMAP()
    embedding_2d = umap_model.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=embedding_2d[:, 0], y=embedding_2d[:, 1], hue=labels, palette="tab10", s=40, alpha=0.8)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def leave_one_out_1nn_accuracy(embeddings, labels):
    labels = np.array(labels)
    embeddings = np.array(embeddings)

    nn = NearestNeighbors(n_neighbors=2, metric='euclidean').fit(embeddings)
    distances, indices = nn.kneighbors(embeddings)

    # Use the second closest point (excluding the point itself)
    nearest_neighbor_indices = indices[:, 1]
    preds = labels[nearest_neighbor_indices]

    accuracy = np.mean(preds == labels)
    conf_mat = confusion_matrix(labels, preds, labels=np.unique(labels))
    return accuracy, conf_mat

def plot_knn_heatmap(embeddings, labels, output_path, title="1-NN Classification Accuracy"):
    accuracy, conf_mat = leave_one_out_1nn_accuracy(embeddings, labels)
    label_set = np.unique(labels)

    # Normalize confusion matrix rows
    acc_matrix = conf_mat.astype(float) / conf_mat.sum(axis=1, keepdims=True)

    plt.figure(figsize=(10, 8))
    sns.heatmap(acc_matrix, xticklabels=label_set, yticklabels=label_set, vmin= 0, vmax = 0.85,
                cmap="Blues", annot=True, fmt=".2f", cbar_kws={"label": "1-NN Accuracy"})
    plt.title(f"{title} (Acc: {accuracy:.2%})")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def analyze_agreement(
    genome_fasta,
    bigwig_file,
    checkpoint_dir,
    config_fpath,
    output_dir,
    num_regions=100,
    chromosomes=None,
    last_step=44000,
    batch_size=8,
    training_chromosomes=None,
    test_chromosomes=None,
    training_task='dual',
    model_type='gamba'
):
    """
    Analyze agreement between predicted and true phyloP scores across different
    genomic regions and chromosomes using pre-saved BED files.
    """
    import logging
    from pathlib import Path
    import torch
    import glob
    from pyfaidx import Fasta

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    model, tokenizer = load_model(checkpoint_dir, config_fpath, last_step=last_step, device=device, training_task=training_task, model_type=model_type)
    genome = Fasta(genome_fasta)

    bw = pyBigWig.open(bigwig_file)

    categories = ["vista_enhancer", "UCNE", "repeats", "exons", "introns", "noncoding_regions", "coding_regions", "upstream_TSS", "UTR5", "UTR3", "promoters"]

    chromosome_groups = {}
    if training_chromosomes and test_chromosomes:
        chromosome_groups = {
            "training": training_chromosomes,
            "test": test_chromosomes
        }
    else:
        chromosome_groups = {"all": chromosomes}

    all_group_embeddings = {
        group_name: {"roi": [], "full": [], "full_labels": [], "roi_labels": []}
        for group_name in chromosome_groups
    }

    for group_name, group_chroms in chromosome_groups.items():
        logging.info(f"Analyzing {group_name} chromosomes: {group_chroms}")
        for category in categories:
            bed_files = glob.glob(f"/home/mica/gamba/data_processing/data/regions/{category}/*.bed")
            group_regions = []
            for bed_file in bed_files:
                loaded = load_bed_file(bed_file, category, genome, bw)
                group_regions.extend([r for r in loaded if r["chrom"] in group_chroms])

            if not group_regions:
                logging.warning(f"No regions found for {category} in {group_name} chromosomes")
                continue

            group_regions = group_regions[:num_regions]

            valid_regions = []
            for i, region in enumerate(group_regions):
                context = extract_context(bigwig_file, region, genome, model_type)
                if not context or "sequence" not in context:
                    print(f"[WARN] Region {i} has invalid or truncated sequence")
                    continue
                valid_regions.append(context)

            if not valid_regions:
                logging.warning(f"[SKIP] All regions in group {group_name} were invalid.")
                continue

            sequence_representations, region_info = predict_scores_batched(
                model, tokenizer, valid_regions,
                batch_size=batch_size,
                device=device,
                model_type=model_type,
                training_task=training_task
            )

            for r in region_info:
                r["category"] = category
            print(f"[DEBUG] Received {len(sequence_representations)} representations")
            print(f"[DEBUG] Received {len(region_info)} region metadata entries")

            full_embeds, full_labels = extract_region_embeddings(sequence_representations, region_info, mode="full")
            roi_embeds, roi_labels = extract_region_embeddings(sequence_representations, region_info, mode="roi")

            all_group_embeddings[group_name]["roi"].extend(roi_embeds)
            all_group_embeddings[group_name]["full"].extend(full_embeds)
            assert len(full_labels) == len(full_embeds), "Full labels and embeddings mismatch!"
            assert len(roi_labels) == len(roi_embeds), "ROI labels and embeddings mismatch!"
            all_group_embeddings[group_name]["full_labels"].extend(full_labels)
            all_group_embeddings[group_name]["roi_labels"].extend(roi_labels)

    for group_name, group_data in all_group_embeddings.items():
        plot_umap(group_data["roi"], group_data["roi_labels"], output_dir / f"global_umap_roi_{group_name}.png", title=f"Global UMAP - ROI ({group_name})")
        plot_knn_heatmap(group_data["roi"], group_data["roi_labels"], output_dir / f"global_knn_roi_{group_name}.png", title=f"1-NN Accuracy - ROI ({group_name})")
        plot_umap(group_data["full"], group_data["full_labels"], output_dir / f"global_umap_full_{group_name}.png", title=f"Global UMAP - Full ({group_name})")
        plot_knn_heatmap(group_data["full"], group_data["full_labels"], output_dir / f"global_knn_full_{group_name}.png", title=f"1-NN Accuracy - Full ({group_name})")

    bw.close()
    

def main():
    parser = argparse.ArgumentParser(
        description="Analyze agreement between predicted and true phyloP scores"
    )
    parser.add_argument(
        "--bigwig_file",
        type=str,
        default="/home/mica/gamba/data_processing/data/240-mammalian/241-mammalian-2020v2.bigWig",
        help="Path to the bigwig file with phyloP scores",
    )
    parser.add_argument(
        "--genome_fasta",
        type=str,
        default="/home/mica/gamba/data_processing/data/240-mammalian/hg38.ml.fa",
        help="Path to the genome fasta file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/mica/gamba/data_processing/data/240-mammalian/global_representations",
        help="Directory to save analysis results",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default='/home/mica/gamba/',
        help="Directory containing model checkpoints",
    )
    parser.add_argument(
        "--config_fpath",
        type=str,
        default='/home/mica/gamba/configs/jamba-small-240mammalian.json',
        help="Path to model config JSON",
    )
    parser.add_argument(
        "--num_regions",
        type=int,
        default=1000,
        help="Number of regions to sample per category",
    )
    parser.add_argument(
        "--chromosomes",
        type=str,
        nargs="+",
        default=["chr2", "chr19", "chr22"],
        help="List of chromosomes to analyze",
    )
    parser.add_argument(
        "--training_chromosomes",
        type=str,
        nargs="+",
        default=["chr19"],
        help="List of chromosomes used in training",
    )
    parser.add_argument(
        "--test_chromosomes",
        type=str,
        nargs="+",
        default=["chr2", "chr22"],
        help="List of chromosomes held out for testing",
    )
    parser.add_argument(
        "--last_step",
        type=int,
        default=44000,
        help="Checkpoint step to use",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for model predictions",
    )
    parser.add_argument(
        "--model_type", type=str, choices=["gamba", "caduceus"], required=True,
        help="Which model type to use (gamba or caduceus)"
    )
    parser.add_argument(
        "--training_task", type=str, choices=["dual", "cons_only", "seq_only"], required=True,
        help="Which task the model was trained on"
    )

    args = parser.parse_args()
    
    # Configure logging to include timestamps
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    logging.info(f"Starting analysis script with chromosomes: {args.chromosomes}")
    logging.info(f"Training chromosomes: {args.training_chromosomes}")
    logging.info(f"Test chromosomes: {args.test_chromosomes}")
    logging.info(f"Using model type: {args.model_type} on task: {args.training_task}")

    if args.model_type == 'gamba':
        checkpoint_dir = args.checkpoint_dir + f"/clean_dcps/CCP/"
        if args.training_task == "seq_only":
            checkpoint_dir = args.checkpoint_dir + f"/clean_dcps/"
            args.last_step = 56000
    else:
        checkpoint_dir = args.checkpoint_dir + f"/clean_caduceus_dcps/"
        args.last_step = 56000
    
    #change outputdir to + dcp checkpoint 
    output_dir = args.output_dir + f"/{args.model_type}_{args.training_task}_step_{args.last_step}/"
    try:
        analyze_agreement(
            args.genome_fasta,
            args.bigwig_file,
            checkpoint_dir,
            args.config_fpath,
            output_dir,
            num_regions=args.num_regions,
            chromosomes=args.chromosomes,
            training_chromosomes=args.training_chromosomes,
            test_chromosomes=args.test_chromosomes,
            last_step=args.last_step,
            batch_size=args.batch_size,
            training_task= args.training_task,
            model_type=args.model_type
        )
        logging.info("Analysis completed successfully")
    except Exception as e:
        logging.error(f"Error in analysis: {e}")
        import traceback
        logging.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    main()