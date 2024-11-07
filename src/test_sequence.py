import argparse
import datetime
import functools
import json
import os
import random
import glob
import logomaker
from typing import Optional, Sequence, Tuple, Type


import numpy as np
import wandb
import pyBigWig
import pandas as pd
import os
import argparse
import numpy as np
from pyfaidx import Fasta
import json
import logging

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Subset

from sequence_models.samplers import SortishSampler, ApproxBatchSampler
from sequence_models.utils import transformer_lr, warmup

import torch.nn.functional as F 
from evodiff.utils import Tokenizer
# import gamba using sys.append
import sys

sys.path.append("../gamba")
from gamba.constants import TaskType, DNA_ALPHABET_PLUS
import logging
_logger = logging.getLogger(__name__)
from gamba.collators import gLMCollator
from gamba.model import create_model, JambagambaModel

import matplotlib.pyplot as plt
import seaborn as sns


#want to load chr16:23,683,829-23,683,929 as a test sequence from the human genome
#load the sequence and tokenize it
#then run the model on it
#then plot the sequence and conservation scores

def get_latest_dcp_checkpoint_path(ckpt_dir: str, last_step: int = -1) -> Optional[str]:
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

def sequence_output(
    bigwig_file: str,
    genome_fasta: str,
    chromosome: str,
    start: int,
    end: int, 
    save_path: str,
    checkpoint: str,
    config_fpath: str,
):
    # open the bigwig file
    bw = pyBigWig.open(bigwig_file)

    # open the genome fasta file
    genome = Fasta(genome_fasta)

    # get the latest checkpoint path
    ckpt_path = get_latest_dcp_checkpoint_path(checkpoint, last_step=54000)

    with open(config_fpath, "r") as f:
        config = json.load(f)
    config["task"] = config["task"].lower().strip()
    epochs = config["epochs"]
    lr = config["lr"]
    warmup_steps = config["warmup_steps"]
    tokenizer = Tokenizer(DNA_ALPHABET_PLUS)
    task = TaskType(config["task"].lower().strip())
    

    print(
        f"Task: {task}, Model: {config['model_type']}, Dataset: {config['dataset']}, Model Config: {config['model_config']}"
    )
    # create the model
    model, block = create_model(
        task, config["model_type"], config["model_config"], tokenizer.mask_id.item(), 
    )

    #get d_model, n_head, n_layers, dim_feedforward and padding_id from the config
    d_model = config.get("d_model", 576) #576/2
    nhead = config.get("n_head", 8)  
    n_layers = config.get("n_layers", 6)
    dim_feedforward = config.get("dim_feedforward", d_model)
    padding_id = config.get("padding_id", 0)


    #set up the model load from last checkpoint
    model = JambagambaModel(
            model, d_model=d_model, nhead=nhead, n_layers=n_layers, padding_id=0, dim_feedfoward=dim_feedforward
        )
    
    print("Using standard checkpoint loading...", flush=True)
    checkpoint = torch.load(os.path.join(ckpt_path, "model_optimizer.pt"))
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer = Adam(
        model.parameters(), lr=lr, weight_decay=config.get("weight_decay", 0.0)
    )
    lr_func = warmup(warmup_steps)
    scheduler = LambdaLR(optimizer, lr_func)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    sd = torch.load(
        os.path.join(ckpt_path, "scheduler.pt"), map_location=torch.device("cpu")
    )
    scheduler.load_state_dict(sd["scheduler_state_dict"])

    # get the sequence from the genome
    sequence = genome[chromosome][start:end].seq

    print("sequence:", sequence)

    print(f"start and end: {start} and {end}")

    # get the conservation scores from the bigwig file
    scores = bw.values(chromosome, start, end)

    # round scores to 2 decimal places
    scores = np.round(scores, 2)
    true_scores = scores.copy()

    # tokenize the sequence
    sequence_tokens = tokenizer.tokenizeMSA(sequence)

    print("the first 10 TOKENIZED chars of the sequence are:", sequence_tokens[:10])
    print("the first 10 conservation scores are:", scores[:10])

    #lets set scores to all be  -100 mask
    # scores = np.full_like(scores, -100)
    # print("MASKED VERSION RUNNING!!!")
    
    #put sequence through gLM collator
    collator = gLMCollator(
            tokenizer=tokenizer,
            pad_to_multiple_of=config.get("pad_to_multiple_of", None),
        )
    collated = collator([(sequence_tokens, scores)])




    #move device to cuda if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # set the model to eval mode
    model.eval()

    # run the model on the sequence
    with torch.no_grad():
        output = model(collated[0].to(device), collated[1].to(device))
    #get the logit outputs from the model
    seq_logits = output["seq_logits"]
    scaling_logits = output["scaling_logits"]


    # plot Position Frequency Matrix for `seq_logits`
    # Get the sequence logits from the model output
    seq_logits = output["seq_logits"]
    
    # Identify indices for "A", "T", "G", "C" in DNA_ALPHABET_PLUS
    nucleotide_indices = [DNA_ALPHABET_PLUS.index(nuc) for nuc in "ATGC"]

    # Convert `seq_logits` to probabilities using softmax
    pfm = F.softmax(seq_logits, dim=-1).cpu().numpy().squeeze()

    # Select only the columns corresponding to "A", "T", "G", "C"
    pfm = pfm[:100, nucleotide_indices]  # Shape should now be (sequence_length, 4)
    
    # Convert to DataFrame for logomaker
    nucleotide_probs = pd.DataFrame(pfm, columns=["A", "T", "G", "C"])

    # Set up figure with two axes: one for the sequence and one for the logo
    fig, (ax_seq, ax_logo) = plt.subplots(2, 1, figsize=(12, 6), gridspec_kw={'height_ratios': [1, 5]}, sharex=True)

    # Plot the actual sequence as text on the upper axis
    ax_seq.set_ylim(0, 1)  # Dummy limits to make the text fit
    ax_seq.axis("off")  # Turn off the axis lines and labels

    # Display the true sequence
    for i, nucleotide in enumerate(sequence[:100]):
        ax_seq.text(i, 0.5, nucleotide, ha="center", va="center", fontsize=8, color="black")

    # Plot the sequence logo on the lower axis with specified colors
    logomaker.Logo(nucleotide_probs, ax=ax_logo, color_scheme={"A": "red", "T": "green", "G": "blue", "C": "yellow"})
    ax_logo.set_title("Sequence Logo (Predicted Nucleotide Probabilities)")
    ax_logo.set_xlabel("Position in sequence")
    ax_logo.set_ylabel("Information content")

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(f"{save_path}_sequence_logo_with_actual_sequence.png")
    plt.show()

    # Verify the shape of scaling_logits and ensure it has entries for each position
    print("Shape of scaling_logits:", scaling_logits.shape)  # Debugging line to check shape

    # # Remove batch dimension and trim to target length by removing tokens evenly from each end to get to 2048
    # scaling_logits = scaling_logits.squeeze(0)
    # if scaling_logits.size(0) > 2048:
    #     scaling_logits = scaling_logits[:2048]  # Limits to 2048 positions if tensor is larger
    # elif scaling_logits.size(0) < 2048:
    #     raise ValueError("The sequence length is less than 2048 positions.")
    # print("Shape of scaling_logits after squeeze and trim:", scaling_logits.shape)  # Should be (2048, 2)

    # # Extract means and variances
    # means = scaling_logits[:, 0].cpu().numpy()
    # variances = scaling_logits[:, 1].exp().cpu().numpy()  # Convert log-variance to variance

    # # Sample from the predicted distribution for each position
    # samples = np.random.normal(loc=means, scale=np.sqrt(variances), size=len(means))

    # # Plot the true and sampled predicted scores as subfigures
    # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # # True conservation scores
    # ax1.plot(true_scores, label="True Conservation Scores", color="green")
    # ax1.set_ylabel("True Score")
    # ax1.set_title("True Conservation Scores by Position")
    # ax1.legend()

    # # Sampled predicted scores with variance shading
    # ax2.plot(samples, label="Sampled Predicted Scores", color="blue")
    # ax2.fill_between(
    #     range(len(samples)),
    #     means - np.sqrt(variances),
    #     means + np.sqrt(variances),
    #     color="gray",
    #     alpha=0.3,
    #     label="Predicted Variance",
    # )
    # ax2.set_xlabel("Position in sequence")
    # ax2.set_ylabel("Sampled Score")
    # ax2.set_title("Sampled Predicted Scores by Position")
    # ax2.legend()

    # #set x axis labels to positions in sequence from start and end
    # ax1.set_xticks(range(0, len(true_scores), 200))
    # ax1.set_xticklabels(range(start, end, 200))

    # ax2.set_xticks(range(0, len(samples), 200))
    # ax2.set_xticklabels(range(start, end, 200))
    # # Adjust layout and save the figure
    # plt.tight_layout()
    # plt.savefig(f"{save_path}_stacked_true_vs_sampled.png")
    # plt.show()


    # #calculate a correlation  between sampled and predicted scores
    # correlation = np.corrcoef(true_scores, samples)[0, 1]
    # print(f"Correlation between true and sampled scores: {correlation}")
    # return
    # Remove batch dimension and trim to target length by removing tokens evenly from each end to get to 2048
    scaling_logits = scaling_logits.squeeze(0)
    if scaling_logits.size(0) > 2048:
        scaling_logits = scaling_logits[:2048]  # Limits to 2048 positions if tensor is larger
    elif scaling_logits.size(0) < 2048:
        raise ValueError("The sequence length is less than 2048 positions.")
    print("Shape of scaling_logits after squeeze and trim:", scaling_logits.shape)  # Should be (2048, 2)

    # Extract means and variances
    means = scaling_logits[:, 0].cpu().numpy()
    variances = scaling_logits[:, 1].exp().cpu().numpy()  # Convert log-variance to variance

    # Sample from the predicted distribution for each position
    samples = np.random.normal(loc=means, scale=np.sqrt(variances), size=len(means))
    
    #instead of sampling, lets just use the means
    samples = means

    fig, ax = plt.subplots(figsize=(10, 8))

    # True conservation scores as green dots
    ax.plot(true_scores, 'go', label="True Conservation Scores", markersize=2)

    # Predicted scores with variance shading
    ax.plot(means, label="Predicted Mean Scores", color="blue")
    ax.fill_between(
        range(len(means)),
        means - np.sqrt(variances),
        means + np.sqrt(variances),
        color="gray",
        alpha=0.3,
        label="Predicted Std Dev",
    )

    ax.set_xlabel("Position in sequence")
    ax.set_ylabel("Score")
    ax.set_title("True Conservation Scores and Predicted Distribution by Position")
    ax.legend()

    # Set x-axis labels to positions in sequence from start to end
    ax.set_xticks(range(0, len(true_scores), 200))
    ax.set_xticklabels(range(start, end, 200))

    # Add vertical dotted lines for specified positions (23678611 to 23678951)
    highlight_start = 23678611
    highlight_end = 23678951
    highlight_indices = range(highlight_start - start, highlight_end - start)

    for pos in highlight_indices:
        ax.axvline(x=pos, color='purple', linestyle=':', linewidth=0.5)  # Vertical dotted line

    # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig(f"{save_path}_true_vs_predicted_with_highlights.png")
    plt.show()

    # Plot the true and sampled predicted scores as subfigures
    # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # # True conservation scores
    # ax1.plot(true_scores, label="True Conservation Scores", color="green")
    # ax1.set_ylabel("True Score")
    # ax1.set_title("True Conservation Scores by Position")
    # ax1.legend()

    # # Sampled predicted scores with variance shading
    # ax2.plot(samples, label="Sampled Predicted Scores", color="blue")
    # ax2.fill_between(
    #     range(len(samples)),
    #     means - np.sqrt(variances),
    #     means + np.sqrt(variances),
    #     color="gray",
    #     alpha=0.3,
    #     label="Predicted Variance",
    # )
    # ax2.set_xlabel("Position in sequence")
    # ax2.set_ylabel("Sampled Score")
    # ax2.set_title("Sampled Predicted Scores by Position")
    # ax2.legend()

    # # Set x-axis labels to positions in sequence from start to end
    # ax1.set_xticks(range(0, len(true_scores), 200))
    # ax1.set_xticklabels(range(start, end, 200))
    # ax2.set_xticks(range(0, len(samples), 200))
    # ax2.set_xticklabels(range(start, end, 200))

    # # Add vertical dotted lines for specified positions (23678611 to 23678951)
    # highlight_start = 23678611
    # highlight_end = 23678951
    # highlight_indices = range(highlight_start - start, highlight_end - start)

    # for pos in highlight_indices:
    #     ax1.axvline(x=pos, color='purple', linestyle=':', linewidth=0.5)  # Vertical dotted line on ax1
    #     ax2.axvline(x=pos, color='purple', linestyle=':', linewidth=0.5)  # Vertical dotted line on ax2

    # # Adjust layout and save the figure
    # plt.tight_layout()
    # plt.savefig(f"{save_path}_stacked_true_vs_sampled_with_highlights.png")
    # plt.show()

    # Calculate and print correlation between true and sampled scores
    correlation = np.corrcoef(true_scores, samples)[0, 1]
    print(f"Correlation between true and sampled scores: {correlation}")
    return samples

def save_predicted_scores_as_bigwig(
    chrom: str,
    start: int,
    predicted_scores,
    output_path: str,
    chrom_sizes_path: str,
):
    # load chromosome sizes from a file (e.g., hg38.chrom.sizes)
    chrom_sizes = {}
    with open(chrom_sizes_path, "r") as f:
        for line in f:
            chrom_name, size = line.strip().split()
            chrom_sizes[chrom_name] = int(size)

    # bigWig file
    bw = pyBigWig.open(output_path, "w")
    bw.addHeader([(chrom, chrom_sizes[chrom])])
    print("predicted_scores:", predicted_scores)

    positions = np.arange(start, start + len(predicted_scores))
    chrom_ends = positions + 1  # half-open intervals [start, end)
    bw.addEntries([chrom] * len(predicted_scores), positions, ends=chrom_ends, values=predicted_scores)

    # close bigwig
    bw.close()
    print(f"Saved bigWig file to {output_path}")


    
def main():
    # process command line arguments
    parser = argparse.ArgumentParser(
        description="Generate data files for training, testing, and validation sets"
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
        "--save_path",
        type=str,
        default="/home/mica/gamba/data_processing/data/240-mammalian/results.pdf",
        help="Path to save the results figure",
    )
    parser.add_argument(
        "--chromosome",
        type=str,
        default="chr16",
        help="Chromosome to analyze",
    )
    parser.add_argument(
        "--start_pos",
        type=int,
        default=23678000, 
        help="Chromosome start pos to analyze",
    )
    parser.add_argument(
        "--end_pos",
        type=int,
        default=23680048, 
        help="Chromosome end pos to analyze",
    )
    parser.add_argument(
        "--chrom_sizes",
        type=str,
        default="/home/mica/gamba/data_processing/data/240-mammalian/hg38.chrom.sizes",
        help="Path to the chromosome sizes file",
    )

    args = parser.parse_args()

    ckpt_dir = os.getenv("AMLT_OUTPUT_DIR", "/tmp/") 

    predicted_scores = sequence_output(
        args.bigwig_file,
        args.genome_fasta,
        args.chromosome,
        args.start_pos,
        args.end_pos,
        args.save_path,
        ckpt_dir,
        '/home/mica/gamba/configs/jamba-small-240mammalian.json'
    )

    save_predicted_scores_as_bigwig(
        args.chromosome,
        args.start_pos,
        predicted_scores,
        "/home/mica/gamba/data_processing/data/240-mammalian/predicted_scores.bw",
        args.chrom_sizes,
    )


if __name__ == "__main__":
    main()





