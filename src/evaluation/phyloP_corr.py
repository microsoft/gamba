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
from src.evaluation.utils.specific_helpers import load_model #predict_scores_batched



CATEGORY_ORDER = [
    "vista_enhancer", "UCNE", "repeats", "exons", "introns",
    "noncoding_regions", "coding_regions", "upstream_TSS",
    "UTR5", "UTR3", "promoters", #"phyloP_negative", "phyloP_neutral", "phyloP_positive",
]


def predict_scores_batched(model, tokenizer, regions, batch_size=8, device=None, model_type="gamba", training_task="dual"):
    """Run predictions on sampled regions with masking applied only over the feature region."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from torch.nn import functional as F
    from gamba.collators import gLMCollator

    all_predictions = []
    all_true_scores = []
    all_seq_predictions = []
    all_true_seqs = []
    region_info = []

    logging.info(f"Running predictions on {len(regions)} regions with batch size {batch_size}...")

    if model_type == "gamba":
        collator = gLMCollator(tokenizer=tokenizer, test=True)
    else:
        collator = gLMMLMCollator(tokenizer=tokenizer, test=True)

    for i in tqdm(range(0, len(regions), batch_size), desc="Batch predictions"):
        batch_regions = regions[i:i + batch_size]
        batch_inputs = []
        batch_region_info = []
        for region in batch_regions:
            sequence_tokens = tokenizer.tokenizeMSA(region['sequence'])
            scores = region['scores']
            fs = region.get('feature_start_in_window', 0)
            fe = region.get('feature_end_in_window', len(scores))

            batch_inputs.append((sequence_tokens,  scores))
            # Record metadata
            batch_region_info.append({
                'chrom': region['chrom'],
                'start': region['start'],
                'end': region['end'],
                'feature_id': region.get('feature_id', 'unknown'),
                'mean_score': region.get('mean_score', 0.0),
                'feature_start_in_window': fs,
                'feature_end_in_window': fe
            })
            region_info.append(batch_region_info[-1])
            all_true_scores.append(scores)
            all_true_seqs.append(sequence_tokens)

        # Skip empty batches
        if not batch_inputs:
            continue

        # === Gamba Forward ===
        if model_type == "gamba":
            collated = collator(batch_inputs)
            with torch.no_grad():
                outputs = model(collated[0].to(device), collated[1].to(device))

            if "scaling_logits" in outputs:
                for j in range(outputs["scaling_logits"].size(0)):
                    means = outputs["scaling_logits"][j, :, 0].cpu().numpy()
                    #print(f"Sample of means values: {means[:10]}...")  # Print first 10 means for debugging
                    all_predictions.append(means)
            else:
                for j in range(len(batch_inputs)):
                    all_predictions.append(np.zeros_like(batch_inputs[j][1]))

            # Append seq logits if present
            if "seq_logits" in outputs:
                for j in range(outputs["seq_logits"].size(0)):
                    #print(f"shape of seq_logits: {outputs['seq_logits'].shape}")
                    logits = outputs["seq_logits"][j].cpu().numpy()
                    #print(f"logits: {logits}")
                    all_seq_predictions.append(logits)

            else:
                all_seq_predictions.extend([np.nan] * len(batch_inputs))

        # === Caduceus Forward ===
        elif model_type == "caduceus":
            feature_spans = [(r["feature_start_in_window"], r["feature_end_in_window"]) for r in batch_region_info]
            batch = collator(batch_inputs, region=feature_spans)
            with torch.no_grad():
                sequence_input = batch[0][:, 0, :].long()       # (B, T)
                scaling = batch[0][:, 1, :].float()             # (B, T)
                sequence_labels = batch[1][:, 0, :].long()      # (B, T)
                scale_lbls = batch[1][:, 1, :].float()          # (B, T)
                model_kwargs = {
                    "input_ids": sequence_input.to(device),
                    "labels": sequence_labels.to(device),
                    }
                # If model supports conservation prediction, pass conservation labels too
                if hasattr(model, "conservation_head"):
                    model_kwargs["conservation_labels"] = scale_lbls.to(device)
            
                outputs = model(**model_kwargs)

            if "cross_entropy_loss" in outputs:
                for _ in batch_inputs:
                    all_seq_predictions.append(outputs["cross_entropy_loss"].item())
            else:
                all_seq_predictions.extend([np.nan] * len(batch_inputs))

            if "scaling_logits" in outputs:
                for j in range(outputs["scaling_logits"].size(0)):
                    means = outputs["scaling_logits"][j, :, 0].cpu().numpy()
                    #print(f"Sample of means values: {means[:10]}...")  # Print first 10 means for debugging
                    all_predictions.append(means)
            else:
                for j in range(len(batch_inputs)):
                    all_predictions.append(np.zeros_like(batch_inputs[j][1]))

    return all_predictions, all_true_scores, region_info, all_seq_predictions, all_true_seqs


def reindex_categories(df):
    return df.reindex([cat for cat in CATEGORY_ORDER if cat in df.index])

from Bio.Seq import Seq

def extract_sequence_from_genome(genome: Fasta, chrom: str, start: int, end: int, strand: str) -> str:
    """
    Extract a sequence from the genome, reverse complementing it if on the minus strand.

    Args:
        genome: pyfaidx.Fasta object with loaded genome.
        chrom: Chromosome name (must match keys in genome, e.g., 'chr1').
        start: 0-based start coordinate (inclusive).
        end: 0-based end coordinate (exclusive).
        strand: '+' or '-'.

    Returns:
        DNA sequence as a string.
    """
    try:
        if chrom not in genome:
            raise ValueError(f"Chromosome {chrom} not found in genome FASTA.")

        seq = genome[chrom][start:end].seq.upper()

        if strand == '-':
            seq = str(Seq(seq).reverse_complement())

        return seq
    except Exception as e:
        print(f"Error extracting sequence from {chrom}:{start}-{end} ({strand}): {e}")
        return "N" * (end - start)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


def get_latest_dcp_checkpoint_path(ckpt_dir, last_step=-1):
    """Find the latest checkpoint path."""
    ckpt_path = None
    if last_step == -1:
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir, exist_ok=True)
        for dir_name in os.listdir(ckpt_dir):
            if "dcp_" in dir_name:
                step = int(dir_name.split("dcp_")[-1])
                if step > last_step:
                    ckpt_path = os.path.join(ckpt_dir, dir_name)
                    last_step = step
    else:
        ckpt_path = os.path.join(ckpt_dir, f"dcp_{last_step}")
    return ckpt_path

def calculate_correlations(true_scores, predicted_scores, region_info, ce_losses, feature_length=1000):
    results = []

    for i in range(len(true_scores)):
        true = true_scores[i]
        pred = predicted_scores[i]

        feature_start = region_info[i]['feature_start_in_window'] 
        feature_end = region_info[i]['feature_end_in_window']  

        #print(f"Feature start: {feature_start}, Feature end: {feature_end}")

        true_feature = np.array(true[feature_start:feature_end])
        pred_feature = np.array(pred[feature_start:feature_end])
        mask = ~(np.isnan(true_feature) | np.isnan(pred_feature))
        true_filtered = true_feature[mask]
        pred_filtered = pred_feature[mask]

        #print(f"Running correlation over {len(true_filtered)} points ")

        if len(true_filtered) > 10:
            corr = np.corrcoef(true_filtered, pred_filtered)[0, 1]
            mean_true = np.mean(true_filtered)
            mean_pred = np.mean(pred_filtered)

            results.append({
                'chrom': region_info[i]['chrom'],
                'start': region_info[i]['start'],
                'end': region_info[i]['end'],
                'feature_id': region_info[i].get('feature_id', 'unknown'),
                'mean_true_score': mean_true,
                'mean_pred_score': mean_pred,
                'loss': ce_losses[i] if ce_losses else np.nan,
                'correlation': corr,
                'num_points': len(true_filtered),
                'feature_length': feature_end - feature_start
            })

    return pd.DataFrame(results)

def calculate_mse(true_scores, predicted_scores, region_info, ce_losses, feature_length=1000):
    """
    Compute per-region MSE between predicted mean phyloP and true phyloP over the feature window.
    Returns a DataFrame with MSE per region (plus metadata and CE loss if available).
    """
    results = []
    for i in range(len(true_scores)):
        true = true_scores[i]
        pred = predicted_scores[i]

        feature_start = region_info[i]['feature_start_in_window']
        feature_end   = region_info[i]['feature_end_in_window']

        true_feature = np.array(true[feature_start:feature_end])
        pred_feature = np.array(pred[feature_start:feature_end])

        # mask invalids
        mask = ~(np.isnan(true_feature) | np.isnan(pred_feature))
        true_filtered = true_feature[mask]
        pred_filtered = pred_feature[mask]

        if len(true_filtered) > 0:
            mse_val = float(np.mean((true_filtered - pred_filtered) ** 2))
            mean_true = float(np.mean(true_filtered))
            mean_pred = float(np.mean(pred_filtered))

            results.append({
                'chrom':        region_info[i]['chrom'],
                'start':        region_info[i]['start'],
                'end':          region_info[i]['end'],
                'feature_id':   region_info[i].get('feature_id', 'unknown'),
                'mean_true_score': mean_true,
                'mean_pred_score': mean_pred,
                'loss':         ce_losses[i] if ce_losses else np.nan,
                'mse':          mse_val,
                'num_points':   int(len(true_filtered)),
                'feature_length': int(feature_end - feature_start),
            })

    return pd.DataFrame(results)


import torch.nn.functional as F

def calculate_ce_losses(sequence_logits_list, region_info, tokenizer, true_sequences=None):
    ce_losses = []

    for i, logits in enumerate(sequence_logits_list):
        if isinstance(logits, float) or np.isnan(logits).any():
            ce_losses.append(np.nan)
            continue

        logits = torch.tensor(logits)  # shape: (seq_len, vocab_size)
        preds = logits.argmax(dim=-1)

        fs = region_info[i]['feature_start_in_window']
        fe = region_info[i]['feature_end_in_window']
        print(f"Length of feature region: {fe - fs}")
        print(f"Feature starts at position {fs} and ends at {fe}.")

        # Get true sequence
        if true_sequences:
            true_seq = true_sequences[i]
        else:
            logging.warning(f"True labels not provided for region {i}, skipping CE loss.")
            ce_losses.append(np.nan)
            continue

        true_labels = torch.tensor(true_seq)

        # Trim logits if necessary
        if len(true_labels) != logits.shape[0]:
            logging.warning(f"[CE Loss] Logits len={logits.shape[0]}, True len={len(true_labels)}. Trimming logits.")
            # Remove [START] and [STOP] and trim to match true_labels
            logits = logits[1:]
            preds = preds[1:]
            # if logits is longer than true_labels, trim logits to match
            if logits.shape[0] > len(true_labels):
                logits = logits[:len(true_labels)]
                preds = preds[:len(true_labels)]
            
        else:
            logits = logits[1:-1]
            preds = preds[1:-1]

        labels_region = true_labels[fs:fe]
        logits_region = logits[fs:fe]

        print(f"Calculating CE loss for region of length: {len(labels_region)}")

        if len(labels_region) == 0 or logits_region.shape[0] == 0:
            ce_losses.append(np.nan)
            continue

        loss = F.cross_entropy(
            logits_region,
            labels_region,
            ignore_index=-100,
            reduction='mean'
        )

        ce_losses.append(loss.item())

    return ce_losses


#need to get the CE loss & conservation  ONLY in the region of interest
def plot_feature_bars(data, value_col, ylabel, title, out_file, output_dir, ylim=None):
    logging.info(f"Creating bar plot: {title}")

    # Filter to the two splits we care about
    data = data[data['data_split'].isin(['Training', 'Held Out'])].copy()

    # Aggregate across ALL chromosomes within each split & category
    agg = (data
           .groupby(['category', 'data_split'])[value_col]
           .agg(['mean', 'std', 'count'])
           .reset_index())

    # Pivot and enforce consistent category order
    pivot = agg.pivot(index='category', columns='data_split', values=['mean', 'std'])
    pivot = reindex_categories(pivot)

    categories = pivot.index.tolist()
    if not categories:
        logging.warning("No categories to plot after aggregation.")
        return

    x = np.arange(len(categories))
    width = 0.42

    plt.figure(figsize=(12, 8))

    # Bars for the two splits, same x with offsets
    for split, offset in [('Held Out', -width/2), ('Training', width/2)]:
        mean_key = ('mean', split)
        if mean_key not in pivot.columns:
            continue
        means = pivot[mean_key].values
        stds = pivot.get(('std', split), None)
        stds = stds.values if stds is not None else None

        # consistent styling, no per-chrom labels
        plt.bar(x + offset, means, width, label=split,
                yerr=stds, capsize=5, linewidth=1.2, alpha=0.9)

        # Annotate values
        for j, v in enumerate(means):
            if np.isfinite(v):
                plt.text(x[j] + offset, v + (0.02 if ylim is None else (ylim[1]-ylim[0])*0.02),
                         f'{v:.3f}', ha='center', fontsize=9, fontweight='bold')

    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.6)
    plt.xlabel('Feature Category', fontsize=12, fontweight='bold')
    plt.ylabel(ylabel, fontsize=12, fontweight='bold')
    plt.title(title, fontsize=14, fontweight='bold')
    if ylim:
        plt.ylim(ylim)
    plt.xticks(x, categories, rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    plt.legend(fontsize=10, loc='upper right')
    plt.grid(axis='y', alpha=0.25)
    plt.tight_layout()

    out_path = os.path.join(output_dir, out_file)
    plt.savefig(out_path, dpi=300)
    plt.close()
    logging.info(f"Bar plot saved to {out_path}")

def plot_feature_heatmap_by_split(
    data, value_col, ylabel, title, out_file, output_dir,
    vmin, vmax, cmap
):
    """
    One column per split ('Held Out', 'Training'), rows = categories.
    Values are the mean across *all* chromosomes within each split.
    """
    logging.info(f"Creating split-averaged heatmap: {title}")

    # Keep only the two splits of interest
    df = data[data['data_split'].isin(['Training', 'Held Out'])].copy()

    # Average over all chromosomes (and regions) per category x split
    agg = (df.groupby(['category', 'data_split'])[value_col]
             .mean()
             .reset_index())

    # Pivot to (categories x splits)
    pivot = agg.pivot(index='category', columns='data_split', values=value_col)

    # Ensure both columns exist and order them
    for col in ['Held Out', 'Training']:
        if col not in pivot.columns:
            pivot[col] = np.nan
    pivot = pivot[['Held Out', 'Training']]

    # Enforce consistent row order and drop empty rows
    pivot = pivot.reindex([c for c in CATEGORY_ORDER if c in pivot.index]).dropna(how='all')

    if pivot.empty:
        logging.warning("No data to plot after aggregation.")
        return

    plt.figure(figsize=(6, max(6, 0.5*len(pivot))))
    ax = sns.heatmap(
        pivot,
        annot=True, fmt='.3f',
        cmap=cmap, vmin=vmin, vmax=vmax,
        linewidths=0.5,
        cbar_kws={'label': ylabel},
        annot_kws={"size": 10, "weight": "bold"}
    )
    ax.set_xlabel("")  # columns already labeled
    ax.set_ylabel("Feature Category")
    plt.title(title, fontsize=14, fontweight='bold')
    plt.yticks(rotation=0, fontsize=10)
    plt.xticks(rotation=0, fontsize=10)
    plt.tight_layout()

    out_path = os.path.join(output_dir, out_file)
    plt.savefig(out_path, dpi=300)
    plt.close()
    logging.info(f"Split-averaged heatmap saved to {out_path}")


from pathlib import Path
import torch
import glob
import logging
from pyfaidx import Fasta
import pyBigWig
from tqdm import tqdm



def analyze_agreement(
    genome_fasta,
    bigwig_file,
    checkpoint_dir,
    config_fpath,
    output_dir,
    num_regions=100,
    region_length=2048,
    chromosomes=None,
    last_step=None,
    batch_size=8,
    training_chromosomes=None,
    test_chromosomes=None,
    training_task='dual',
    model_type='gamba'
):
    """
    Analyze agreement between predicted and true phyloP scores using pre-defined BED regions.
    Outputs CE loss and phyloP correlation per region, with heatmaps per category/chromosome.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    bw = pyBigWig.open(bigwig_file)

    model, tokenizer = load_model(
        checkpoint_dir, config_fpath,
        last_step=last_step, device=device,
        training_task=training_task, model_type=model_type
    )
    genome = Fasta(genome_fasta)

    categories = [
        #"phyloP_negative", "phyloP_neutral", "phyloP_positive", 
        "UCNE", "repeats", "exons", "introns", "noncoding_regions", "coding_regions", "upstream_TSS", "UTR5", "UTR3", "promoters", "vista_enhancer"]

    chromosome_groups = {
        "training": training_chromosomes or [],
        "test": test_chromosomes or [],
    }

    all_results = []

    for group_name, group_chroms in chromosome_groups.items():
        logging.info(f"Analyzing {group_name} chromosomes: {group_chroms}")
        for category in categories:
            bed_files = glob.glob(f"/home/mica/gamba/data_processing/data/regions/{category}/*.bed")
            group_regions = []
            for bed_file in bed_files:
                loaded = load_bed_file(bed_file, category, genome, bw)
                group_regions.extend([r for r in loaded if r["chrom"] in group_chroms])

            if not group_regions:
                logging.warning(f"No regions found for {category} in {group_name}")
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


            predicted_scores, true_scores, region_info, all_seq_predictions, all_true_sequences = predict_scores_batched(
                model, tokenizer, valid_regions, batch_size=batch_size,
                device=device, model_type=model_type, training_task=training_task
            )

            if model_type == 'gamba':
                ce_losses = calculate_ce_losses(all_seq_predictions, region_info, tokenizer, all_true_sequences)
            else:
                ce_losses = all_seq_predictions  # already loss values for caduceus

            #correlation_df = calculate_correlations(true_scores, predicted_scores, region_info, ce_losses)
            #correlation_df['category'] = category
            #correlation_df['group'] = group_name
            mse_df = calculate_mse(true_scores, predicted_scores, region_info, ce_losses)
            mse_df['category'] = category
            mse_df['group'] = group_name

            train_set = set(training_chromosomes or [])
            test_set  = set(test_chromosomes or [])

            def split_of(chrom):
                if chrom in train_set:
                    return 'Training'
                if chrom in test_set:
                    return 'Held Out'
                return 'Unknown'  # should not happen if lists are complete

            #correlation_df['data_split'] = correlation_df['chrom'].apply(split_of)
            mse_df['data_split'] = mse_df['chrom'].apply(split_of)


            # Save region-wise results
            #corr_out_path = output_dir / f"{category}_{group_name}_results.csv"
            #correlation_df.to_csv(corr_out_path, index=False)
            mse_out_path = output_dir / f"{category}_{group_name}_mse_results.csv"
            mse_df.to_csv(mse_out_path, index=False)

            #all_results.append(correlation_df)
            all_results.append(mse_df)

    # Merge all region results into one DataFrame
    full_df = pd.concat(all_results, ignore_index=True)

    # Save merged results
    #full_df.to_csv(output_dir / "all_region_results.csv", index=False)
    full_df.to_csv(output_dir / "all_region_mse_results.csv", index=False)

    # Create summary plots
    data=full_df
    from matplotlib.colors import LinearSegmentedColormap
    # plot_feature_bars(data, 'correlation',
    #               ylabel='Mean Correlation (Predicted vs True PhyloP)',
    #               title='Corre lation: Held Out vs Training Chromosomes',
    #               out_file='feature_comparison_bar_plot.png',
    #               output_dir=output_dir,
    #               ylim=(-0.1, 0.5))

    plot_feature_bars(data, 'loss',
                    ylabel='CE Loss',
                    title='Cross-Entropy Loss: Held Out vs Training Chromosomes',
                    out_file='feature_comparison_ce_loss_plot.png',
                    output_dir=output_dir,
                    ylim=(1.0, 1.40))

    
    # plot_feature_heatmap_by_split(
    #     data, value_col='correlation',
    #     ylabel='Mean Correlation',
    #     title='Mean Correlation (Held-Out vs Training, averaged)',
    #     out_file='feature_split_correlation_heatmap.png',
    #     output_dir=output_dir,
    #     vmin=-0.1, vmax=0.5,
    #     cmap=LinearSegmentedColormap.from_list('white_to_blue', [(1,1,1),(0,0.4,0.8)], N=100)
    # )

    plot_feature_heatmap_by_split(
        data, value_col='loss',
        ylabel='Mean Cross-Entropy Loss',
        title='CE Loss (Held-Out vs Training, averaged)',
        out_file='feature_split_ce_loss_heatmap.png',
        output_dir=output_dir,
        vmin=1.0, vmax=1.38,
        cmap=LinearSegmentedColormap.from_list('white_to_red', [(1,1,1),(0.8,0.1,0.1)], N=100)
    )
    
    plot_feature_bars(
        data, 'mse',
        ylabel='Mean MSE (Predicted Mean vs True phyloP)',
        title='MSE: Held Out vs Training Chromosomes',
        out_file='feature_comparison_mse_bar_plot.png',
        output_dir=output_dir,
        # Optionally set ylim; uncomment if you want a fixed view:
        ylim=(0.0, 3.5)
    )

    plot_feature_heatmap_by_split(
        data, value_col='mse',
        ylabel='Mean MSE',
        title='Mean MSE (Held-Out vs Training, averaged)',
        out_file='feature_split_mse_heatmap.png',
        output_dir=output_dir,
        # If you know your expected range, set vmin/vmax; otherwise let it auto-scale:
        vmin=None, vmax=None,
        cmap=LinearSegmentedColormap.from_list('white_to_blue', [(1,1,1),(0,0.4,0.8)], N=100)
    )
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
        default="/home/mica/gamba/data_processing/data/240-mammalian/phylop_corr_analysis",
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
        "--region_length",
        type=int,
        default=2048,
        help="Length of each sampled region",
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
        default=["chr1", "chr4", "chr5", "chr6", "chr7", "chr8", "chr9", "chr10","chr11", "chr12", "chr13", "chr14", "chr15", "chr17", "chr18", "chr19", "chr20", "chr21", "chrX"],
        help="List of chromosomes used in training",
    )
    parser.add_argument(
        "--test_chromosomes",
        type=str,
        nargs="+",
        default=["chr2", "chr22", "chr16", "chr3"],
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
            region_length=args.region_length,
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