import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyfaidx import Fasta
import torch
import sys
import os
sys.path.append("../gamba")
from typing import Optional, Sequence, Tuple, Type
import umap
import seaborn as sns
import torch.distributed as dist
from torch.cuda.amp import GradScaler
import re

import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset

from sequence_models.samplers import SortishSampler, ApproxBatchSampler
from sequence_models.utils import transformer_lr, warmup

import torch.nn.functional as F 
from evodiff.utils import Tokenizer
from gamba.collators import gLMCollator
from gamba.model import create_model, JambaGambaModelWithDegeneracies
from gamba.constants import TaskType, DNA_ALPHABET_PLUS
import pyBigWig
import json
import argparse
import datetime
import functools
import json
import os
import random
import glob
from typing import Optional, Sequence, Tuple, Type
from sklearn.metrics import accuracy_score

import numpy as np
import wandb
from Bio.Seq import Seq
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
from torch.distributed.fsdp import (
    BackwardPrefetch,
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.device_mesh import init_device_mesh
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Subset

from sequence_models.samplers import SortishSampler, ApproxBatchSampler
from sequence_models.utils import transformer_lr, warmup

from evodiff.utils import Tokenizer

#import GradScaler
from torch.cuda.amp import GradScaler

# import gamba using sys.append
import sys

sys.path.append(os.environ["PWD"])  # allow import from project directory.

from gamba.activation_checkpointing import apply_activation_checkpointing
from gamba.collators import gLMCollatorWithDegeneracies, LMCollator, OAMaskCollator
from gamba.constants import TaskType, DNA_ALPHABET_PLUS
from gamba.datasets import ConservationDataset
from gamba.model import (
    ARDiffusionModel,
    OrderAgnosticDiffusionModel,
    JambagambaModel,
    JambaGambaModelWithDegeneracies,
    OTHER_METRICS_KEY,
)
from gamba.model import create_model


import os
import torch
import time
import mamba_ssm
import causal_conv1d

print(f"causal_conv1d version: {causal_conv1d.__version__}")
print(f"mamba_ssm version: {mamba_ssm.__version__}")

# default values for RANK, LOCAL_RANK, and WORLD_SIZE if not set
ckpt_dir = os.getenv("AMLT_OUTPUT_DIR", "/tmp") + "/"
RANK = int(os.environ.get("RANK", "0"))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", "0"))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "1"))
DEVICE = torch.device(f"cuda:{LOCAL_RANK}" if torch.cuda.is_available() else "cpu")

import pickle

def save_gene_positions(gene_positions, output_file):
    with open(output_file, 'wb') as f:
        pickle.dump(gene_positions, f)

def load_gene_positions(input_file):
    with open(input_file, 'rb') as f:
        return pickle.load(f)


class SequenceDataset(Dataset):
    def __init__(self, sequences, scores, deg):
        self.sequences = sequences
        self.scores = scores
        self.degeneracies = deg

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.scores[idx], self.degeneracies[idx]

def evaluate_model_and_get_predictions(model, dataloader, device):
    model.eval()
    total_ce_loss = 0
    total_gaussian_loss = 0
    num_batches = 0
    total_tokens = 0
    total_seqs = 0
    total_accuracy = 0
    #intialize conservation logits as empty tuple of tensors
    conservation_logits = []
    true_phyloP = []
    true_degeneracies = []
    with torch.no_grad():
        for batch in dataloader:
            output = step(model, batch, None, None, training=False)
            num_batches += 1
            # with torch.no_grad():
            #     reduce_tensor = torch.stack(
            #         (
            #             output["n_processed"],
            #             output["n_seqs"],
            #             output["cross_entropy_loss"],
            #             output["gaussian_loss"],
            #             output["accuracy"],
            #             output["scaling_logits"],
            #             output["conservation_tgt"]
            #         )
            #     )
            #     if WORLD_SIZE > 1:
            #         dist.reduce(reduce_tensor, 0, op=dist.ReduceOp.SUM)
            #         total_tokens += int(reduce_tensor[0].item())
            #         total_seqs += int(reduce_tensor[1].item())
            #         total_ce_loss += reduce_tensor[2].item()
            #         total_gaussian_loss += reduce_tensor[3].item()
            #         total_accuracy += reduce_tensor[4].item()
            # else:
            total_tokens += output["n_processed"]
            total_seqs += output["n_seqs"]
            total_ce_loss += output["cross_entropy_loss"]
            total_gaussian_loss += output["gaussian_loss"]
            total_accuracy += output["accuracy"]
            
            seq_logits = output["seq_logits"]
            scaling_logits = output["scaling_logits"]
            conservation_tgt = output["conservation_tgt"]
            accuracy = output["accuracy"]
            ce_loss = output["cross_entropy_loss"]
            gaussian_loss = output["gaussian_loss"]
            degeneracies_tgt = output["degeneracies_tgt"]
           
            conservation_logits.append(scaling_logits)
            true_phyloP.append(conservation_tgt)
            true_degeneracies.append(degeneracies_tgt)

    print("NUM BATCHES: ", num_batches)
    conservation_logits = torch.cat(conservation_logits, dim=0)
    true_phyloP = torch.cat(true_phyloP, dim=0)
    true_degeneracies = torch.cat(true_degeneracies, dim=0)
 
    ce_loss = total_ce_loss / num_batches
    gaussian_loss = total_gaussian_loss / num_batches
    accuracy = total_accuracy / num_batches
    return accuracy, ce_loss, gaussian_loss, conservation_logits, true_phyloP, true_degeneracies

def save_continuous_stretches_to_bed(chromosome, gene_positions, output_file):
    bed_data = []
    for gene, data in gene_positions.items():
        for stretch in data['continuous_stretches']:
            bed_data.append({
                'chrom': chromosome,  
                'start': stretch['start'],
                'end': stretch['end'],
                'degeneracy': ' '.join(map(str, stretch['degeneracy'])),
                'reverse_complemented': data['reverse_complemented'],
                'gene': gene
            })
    
    bed_df = pd.DataFrame(bed_data)
    bed_df.to_csv(output_file, sep='\t', header=False, index=False)


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

#our bed file now is per chromosome and looks like:
#chromosome position_start position_end gene.id:position_in_gene degeneracy codon amino_acid mutation_summary
# chr22	15528191	15528192	ENST00000643195.1:0	0	A	M	T:L;C:L;G:V
#where mutation_summary means for non-degenerate sites (not 0-fold)
# the last column of the bed file contains information about how each mutation 
# to non-degenerate nucleotides changes the amino acid. For example, if the final 4 columns of the bed file are:
#so we need to get for each unique gene 'ENST00000643195.1' the first and last entry position, and then extract 
#each of the degeneracy sites as a label and save it, so we can check if the phyloP scores are conserved comparing 4 fold
#degeneracy (a value of 4) compare to a value of 0, and then see if our predicted phyloP matches this conservation trend
def load_bed_file(bed_file):
    bed_df = pd.read_csv(bed_file, sep='\t', header=None, names=['chrom', 'start', 'end', 'gene_pos', 'degeneracy', 'codon', 'amino_acid', 'mutation_summary'])
    #split gene_pos column into gene and ordering
    bed_df['gene'] = bed_df['gene_pos'].apply(lambda x: x.split(':')[0])
    bed_df['ordering'] = bed_df['gene_pos'].apply(lambda x: x.split(':')[1])
    #subset the df to only have ENST00000327669.5 gene
    #bed_df = bed_df[bed_df['gene'] == 'ENST00000327669.5']
    return bed_df


def load_exon_bed_file(bed_file):
    bed_df = pd.read_csv(bed_file, sep='\t', header=None, names=['chrom', 'start', 'end', 'degeneracy', 'reverse_complement', 'gene'])
    return bed_df

def extract_gene_positions(bed_df):
    gene_positions = {}
    # group by unique gene column
    for gene, group in bed_df.groupby('gene'):
        #print("gene:", gene)
        gene_start = group['start'].min()
        gene_end = group['end'].max()
        #print(f"start: {gene_start}, end: {gene_end}")

        #subset the df to just this gene
        gene_df = group[['start', 'end', 'ordering', 'degeneracy']]
        gene_df['ordering'] = gene_df['ordering'].astype(int)
        reverse_complement = False

        if gene_df['ordering'].is_monotonic_decreasing:
            reverse_complement = True

        #i want to get one entry per continuous segment in a gene, 
        # i.e. if the first row in the gene has start = 41610 and the next row has start 41611 they're continuous so just 
        # one entry, append the degeneracy values to a list for this genome segment
        #otherwise, start a new segment
        
        continuous_stretches = []
        current_stretch = [gene_df.iloc[0]['start'], gene_df.iloc[0]['end']]
        current_degeneracy = [gene_df.iloc[0]['degeneracy']]
       
        for i in range(1, len(gene_df)):
            start = gene_df.iloc[i]['start']
            end = gene_df.iloc[i]['end']
            print(f"gene start {start}, gene end {end}")
            #check if degeneracy value is an integer, if not, we're going to end the gene here (this . is either at the start or end of a gene)
            if gene_df.iloc[i]['degeneracy'] == '.':
                degeneracy = int(-500)
            else:
                degeneracy = int(gene_df.iloc[i]['degeneracy'])
            
            
            if start == current_stretch[1]:
                # extend  current stretch
                current_stretch[1] = end
                current_degeneracy.append(degeneracy)
            else:
                # save current stretch and start a new one
                continuous_stretches.append({
                    'start': current_stretch[0],
                    'end': current_stretch[1],
                    'degeneracy': np.array(current_degeneracy),
                })
                current_stretch = [start, end]
                current_degeneracy = [degeneracy]
               
        # add last stretch
        continuous_stretches.append({
            'start': current_stretch[0],
            'end': current_stretch[1],
            'degeneracy': np.array(current_degeneracy)
        })
        # confirm  length of degeneracy matches the distance between start and end
        for stretch in continuous_stretches:
            assert len(stretch['degeneracy']) == (stretch['end'] - stretch['start']), \
                f"Degeneracy length {len(stretch['degeneracy'])} does not match distance {stretch['end'] - stretch['start']}"

        
        gene_positions[gene] = {
            'start': gene_start,
            'end': gene_end,
            'continuous_stretches': continuous_stretches,
            'reverse_complemented': reverse_complement
        }
        print(f"continuous_stretches: {continuous_stretches}")

    return gene_positions



def extract_degeneracy_sites(bed_df):
    degeneracy_sites = []
    for index, row in bed_df.iterrows():
        degeneracy_sites.append({
            'chrom': row['chrom'],
            'start': row['start'],
            'end': row['end'],
            'degeneracy': row['degeneracy']
        })
    return degeneracy_sites

def step(
    model: nn.Module,
    batch: Sequence[torch.Tensor],
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    training: bool = True,
) -> dict:
    if any(el.numel() for el in batch) == 0:
        raise ValueError("Empty tensor in batch")

    batch = [el.to(DEVICE) for el in batch]
    scaler = GradScaler()
    if training:
        # step through model
        optimizer.zero_grad()
        outputs = model(*batch)
        scaler.scale(outputs["loss"]).backward()

        # Unscales the gradients of optimizer's assigned params in-place
        scaler.unscale_(optimizer)

        # Define max_norm
        max_norm = 1.0

        # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        # optimizer's gradients are already unscaled, so scaler.step does not unscale them,
        # although it still skips optimizer.step() if the gradients contain infs or NaNs.
        scaler.step(optimizer)
        scheduler.step()
        # Updates the scale for next iteration.
        scaler.update()
        print(f"entering model with batch {batch[0].shape}")
    else:
        # validation
        with torch.no_grad():
            outputs = model(*batch)
    return outputs


def process_bed_file(bed_df, genome, bw, tokenizer):
    sequences = []
    scores_list = []
    degeneracies = []
    #valid_chromosomes = "chr1 chr2 chr3 chr4 chr5 chr6 chr7 chr8 chr9 chr10 chr11 chr12 chr13 chr14 chr15 chr16 chr17 chr18 chr19 chr20 chr21 chr22 chrX".split()
    valid_chromosomes = "chr2 chr22".split()

    for index, row in bed_df.iterrows():
        #get chromosomes start and end by tab
        chromosome = row['chrom']
        start = row['start']
        end = row['end']
        degeneracy = row['degeneracy']
        reverse_complement = row['reverse_complement']

        #print(f"Processing {label} sequence {index} on chromosome {chromosome} from {start} to {end}")

        #cut sequence to 3000bp max:
        if end - start > 2048:
            end=start+2048
            #cut degeneracy array
            degeneracy = degeneracy[:2048]

        if chromosome not in valid_chromosomes:
            continue
    
        # Get the reference sequence
        ref_sequence = Seq(genome[chromosome][start:end].seq)


        if reverse_complement:
            ref_sequence = ref_sequence.reverse_complement()

        # Tokenize the reference sequence
        ref_sequence_tokens = tokenizer.tokenizeMSA(ref_sequence)

        # Initialize vals with zeros
        vals = np.zeros(end - start, dtype=np.float64)

        # Get the conservation scores from the bigwig file
        intervals = bw.intervals(chromosome, start, end)

        # Check if intervals is None
        if intervals is None:
            # print(f"Chromosome: {chromosome}, Start: {start}, End: {end}")
            # print("Error: intervals is None")
            continue
        else:
            for interval_start, interval_end, value in intervals:
                vals[interval_start - start : interval_end - start] = value

        # Round scores to 2 decimal places
        scores = np.round(vals, 2)

        #if reverse complement, reverse the scores and degeneracies
        if reverse_complement:
            scores = scores[::-1].copy()
            degeneracy = degeneracy[::-1]

        #ensure scores, sequences and degeneracies are the same length, if not, cut the longer one
        if len(scores) == len(ref_sequence_tokens) and len(degeneracy) == len(ref_sequence_tokens):
            sequences.append(ref_sequence_tokens)
            scores_list.append(scores)
            degeneracies.append(degeneracy)
        else:
            continue
    return sequences, scores_list, degeneracies

def check_actual_degeneracies(exon_scores, exon_degeneracies):
    print("exon scores shape:", ((exon_scores.shape)))
    # Check to see if the actual phyloP scores coincide with degeneracy of sites
    avg_scores_by_degeneracy = {0: [], 1: [], 2: [], 3: [], 4: []}
    
    for scores, degeneracies in zip(exon_scores, exon_degeneracies):
        scores = np.array(scores.cpu().numpy(), dtype=float)
        
        mask = scores != -100
        scores = scores[mask]
        degeneracies = degeneracies[mask]

        #turn degeneracies to cpu
        degeneracies = np.array(degeneracies.cpu(), dtype=int)
        for deg in [0, 1, 2, 3, 4]:
            deg_scores = scores[degeneracies == deg]
            if len(deg_scores) > 0:
                avg_scores_by_degeneracy[deg].append(np.mean(deg_scores))
    
    for deg in [0, 1, 2, 3, 4]:
        if avg_scores_by_degeneracy[deg]:
            avg_score = np.mean(avg_scores_by_degeneracy[deg])
            print(f"Average conservation for {deg}-fold sites: {avg_score}")
        else:
            print(f"No scores for {deg}-fold sites")


def check_predicted_degeneracies(conservation_logits, exon_scores, exon_degeneracies):
    avg_scores_by_degeneracy = {0: [], 1: [], 2: [], 3: [], 4: []}
    for logits, tgt, degeneracies in zip(conservation_logits, exon_scores, exon_degeneracies):
        # conservation logits are mean and var
        mean = logits[:, 0]
        log_var = logits[:, 1]

        
        # apply  mask to ignore positions with -100
        mask = tgt != -100
        mean = mean[mask]
        log_var = log_var[mask]
        degeneracies = degeneracies[mask]

        
        #turn log_var and mean to numpy arrays
        log_var = np.array(log_var.cpu(), dtype=float)
        mean = np.array(mean.cpu(), dtype=float)

        degeneracies = np.array(degeneracies.cpu(), dtype=int)
        var = np.exp(log_var)
        mean = np.exp(mean)
        
        for deg in [0, 1, 2, 3, 4]:
            deg_scores = mean[degeneracies == deg]
            if len(deg_scores) > 0:
                avg_scores_by_degeneracy[deg].append(np.mean(deg_scores))
    
    for deg in [0, 1, 2, 3, 4]:
        if avg_scores_by_degeneracy[deg]:
            avg_score = np.mean(avg_scores_by_degeneracy[deg])
            print(f"Average predicted conservation for {deg}-fold sites: {avg_score}")
        else:
            print(f"No scores for {deg}-fold sites")


def main():
    parser = argparse.ArgumentParser(description="Process exon and intron sequences and get representations")
    parser.add_argument('--genome_fasta', type=str, default='/home/mica/gamba/data_processing/data/240-mammalian/hg38.ml.fa', help='Path to the genome FASTA file')
    parser.add_argument('--big_wig', type=str, default='/home/mica/gamba/data_processing/data/240-mammalian/241-mammalian-2020v2.bigWig', help='Path to the bigWig file')
    parser.add_argument('--output_file', type=str, default='/home/mica/gamba/data_processing/data/degeneracy/chr2', help='Path to the output file')
    parser.add_argument('--config_fpath', type=str, default='/home/mica/gamba/configs/jamba-small-240mammalian.json', help='Path to the config file')
    parser.add_argument('--chr_coding_sites', type=str, default='/home/mica/gamba/data_processing/data/240-mammalian/chr2_degenotate/degeneracy-all-sites.bed', help='Path to the BED file with the degenotate annotated chromosomes')
    args = parser.parse_args()

    # Load BED files
    gene_df = load_bed_file(args.chr_coding_sites)
    #intron_bed_df = load_bed_file(args.intron_bed_file)

    #get chromosome from file path: /home/mica/gamba/data_processing/data/240-mammalian/chr2_degenotate
    match = re.search(r'chr[0-9XY]+', args.chr_coding_sites)
    if match:
        chromosome = match.group(0)
    else:
        #send error message need filename to look like /home/mica/gamba/data_processing/data/240-mammalian/{chr_name}_degenotate
        print("Error: chromosome name not found in file path")

    #our bed file now is per chromosome and looks like:
    #chromosome position_start position_end gene.id:position_in_gene degeneracy codon amino_acid mutation_summary
    # chr22	15528191	15528192	ENST00000643195.1:0	0	A	M	T:L;C:L;G:V
    #where mutation_summary means for non-degenerate sites (not 0-fold)
    # the last column of the bed file contains information about how each mutation 
    # to non-degenerate nucleotides changes the amino acid. For example, if the final 4 columns of the bed file are:
    #so we need to get for each unique gene 'ENST00000643195.1' the first and last entry position, and then extract 
    #each of the degeneracy sites as a label and save it, so we can check if the phyloP scores are conserved comparing 4 fold
    #degeneracy (a value of 4) compare to a value of 0, and then see if our predicted phyloP matches this conservation trend

    

    #should be 962 for ENST00000327669.5

    #check if gene_positions file exists:
    gene_positions_file = os.path.join(args.output_file, 'gene_positions.pkl')
    if os.path.exists(gene_positions_file):  
        # load gene_positions from file
        gene_positions = load_gene_positions(gene_positions_file)
    else:
        # extract gene_positions
        gene_positions = extract_gene_positions(gene_df)
        print(f"Gene positions: {gene_positions}")
        # save gene_positions to a file
        save_gene_positions(gene_positions, gene_positions_file)

    if not os.path.exists(os.path.join(args.output_file, 'continuous_stretches.bed')):
        # unroll gene positions to just be the continuous stretch information
        bed_output_file = os.path.join(args.output_file, 'continuous_stretches.bed')
        save_continuous_stretches_to_bed(chromosome, gene_positions, bed_output_file)
    else:
        print("Continuous stretches file already exists")
        # load from file
        bed_output_file = os.path.join(args.output_file, 'continuous_stretches.bed')

    #load the bed_output_file
    exon_bed_df = load_exon_bed_file(bed_output_file)

    # Load genome
    genome = Fasta(args.genome_fasta)

    # Load bigwig file
    bw = pyBigWig.open(args.big_wig)

    # Get checkpoint path with step=5400
    ckpt_dir = os.getenv("AMLT_OUTPUT_DIR", "/tmp/") 
    ckpt_path = get_latest_dcp_checkpoint_path(ckpt_dir, 80000)

    # Load model configuration
    with open(args.config_fpath, "r") as f:
        config = json.load(f)
    config["task"] = config["task"].lower().strip()
    tokenizer = Tokenizer(DNA_ALPHABET_PLUS)
    task = TaskType(config["task"].lower().strip())

    print(
        f"Task: {task}, Model: {config['model_type']}, Dataset: {config['dataset']}, Model Config: {config['model_config']}"
    )
    # Create the model
    model, block = create_model(
        task, config["model_type"], config["model_config"], tokenizer.mask_id.item(), 
    )

    # Get d_model, n_head, n_layers, dim_feedforward and padding_id from the config
    d_model = config.get("d_model", 576) #576/2
    nhead = config.get("n_head", 8)  
    n_layers = config.get("n_layers", 6)
    dim_feedforward = config.get("dim_feedforward", d_model)
    padding_id = config.get("padding_id", 0)

    # Set up the model load from last checkpoint
    model = JambaGambaModelWithDegeneracies(
            model, d_model=d_model, nhead=nhead, n_layers=n_layers, padding_id=0, dim_feedfoward=dim_feedforward
        )

    # Load the model checkpoint
    checkpoint = torch.load(os.path.join(ckpt_path, "model_optimizer.pt"))
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer = Adam(
        model.parameters(), lr=config["lr"], weight_decay=config.get("weight_decay", 0.0)
    )
    lr_func = warmup(config["warmup_steps"])
    scheduler = LambdaLR(optimizer, lr_func)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    sd = torch.load(
        os.path.join(ckpt_path, "scheduler.pt"), map_location=torch.device("cpu")
    )
    scheduler.load_state_dict(sd["scheduler_state_dict"])

    # Move device to cuda if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    collator = gLMCollatorWithDegeneracies(
        tokenizer=tokenizer,
        pad_to_multiple_of=None,
        test=True,
    )

    #exon sequences
    exon_sequences, exon_scores, exon_degeneracies = process_bed_file(exon_bed_df, genome, bw, tokenizer)

    exon_dataset = SequenceDataset(exon_sequences, exon_scores, exon_degeneracies)
    print("number of examples in exon dataset:", len(exon_dataset))
    exon_dataloader = DataLoader(exon_dataset, batch_size=2, collate_fn=collator)
    
    
    #check the accuracy of the model in predicting on coding sequences as a whole
    accuracy, ce_loss, gaussian_loss, conservation_logits, true_phyloP, true_degeneracies = evaluate_model_and_get_predictions(model, exon_dataloader, device)
    print(f"Accuracy on coding sequences: {accuracy:.4f}")
    print(f"Cross-Entropy Loss on coding sequences: {ce_loss:.4f}")
    print(f"Gaussian Loss on coding sequences: {gaussian_loss:.4f}")
    #put everything .cpu()


    #check to see if the actual phyloP scores in the exons have a wobble (i.e. the third position is not conserved)

                                         
    check_actual_degeneracies(true_phyloP, true_degeneracies)
    # print(f"len(conseration_logits): {len(conservation_logits)}")
    # print(f"len(exon_degeneracies): {len(exon_degeneracies)}")
    #check to see if the predicted phyloP scores follow the exon degeneracies
    #flip the ordering of conservation logits
    check_predicted_degeneracies(conservation_logits, true_phyloP, true_degeneracies)
    #check if drops in sequence logit prediction accuracy are correlated with drops in phyloP scores TO-DO
    


if __name__ == "__main__":
    main()