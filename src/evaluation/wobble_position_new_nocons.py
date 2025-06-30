import argparse
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from pyfaidx import Fasta
import pyBigWig

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
from gamba.model import create_model, JambagambaModel, JambaGambaNoConsModel
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

#import GradScaler
from torch.cuda.amp import GradScaler

# import gamba using sys.append
import sys

sys.path.append(os.environ["PWD"])  # allow import from project directory.

from gamba.activation_checkpointing import apply_activation_checkpointing
from gamba.constants import TaskType, DNA_ALPHABET_PLUS
from gamba.datasets import ConservationDataset
from gamba.model import (
    ARDiffusionModel,
    OrderAgnosticDiffusionModel,
    JambagambaModel,
    JambaGambaNoConsModel,
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


def check_continuous_stretches_bed_file(bed_file):
    # Read the BED file into a DataFrame with specified data types
    bed_df = pd.read_csv(bed_file, sep='\t', header=None, names=['chrom', 'start', 'end', 'degeneracy', 'reverse_complement', 'gene'])

    # Check that the length of each continuous stretch matches the length of degeneracies
    for index, row in bed_df.iterrows():
        start = row['start']
        end = row['end']
        degeneracy = row['degeneracy']
        degeneracy = degeneracy.split(' ')
        degeneracy = [int(x) if x != '.' else -500 for x in degeneracy]
        length_of_stretch = end - start 
        length_of_degeneracies = len(degeneracy)

        if length_of_stretch != length_of_degeneracies:
            print(f"Error: Length of stretch ({length_of_stretch}) does not match length of degeneracies ({length_of_degeneracies}) for row {index}")
        else:
            continue
            #print(f"Row {index} is valid: Length of stretch ({length_of_stretch}) matches length of degeneracies ({length_of_degeneracies})")


class SequenceDataset(Dataset):
    def __init__(self, sequences, scores):
        self.sequences = sequences
        self.scores = scores

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.scores[idx]

def process_bed_file(bed_df, genome, bw, tokenizer, context_size=1000):
    """Process bed file with context, using -100 padding for non-degenerate regions."""
    sequences = []
    scores_list = []
    degeneracies = []
    valid_chromosomes = "chr22" #"chr2"  #"chr19"

    for index, row in bed_df.iterrows():
        chromosome = row['chrom']
        if chromosome not in valid_chromosomes:
            continue

        # Original region
        start = row['start']
        end = row['end']
        degeneracy = row['degeneracy'].split(' ')
        degeneracy = [int(x) if x != '.' else -500 for x in degeneracy]
        reverse_complement = row['reverse_complement']
        
        # Add context
        context_start = max(0, start - context_size)
        context_end = min(end + context_size, len(genome[chromosome]))
        
        # Get sequence with context
        ref_sequence = Seq(genome[chromosome][context_start:context_end].seq)
        
        # Create padded degeneracy array
        full_degeneracy = [-100] * context_size + degeneracy + [-100] * context_size
        # Trim if necessary due to chromosome boundaries
        if context_start > 0:
            full_degeneracy = full_degeneracy[context_size-context_start:]
        if context_end < len(genome[chromosome]):
            full_degeneracy = full_degeneracy[:-(context_end-end)]

        # Get conservation scores
        vals = np.zeros(context_end - context_start, dtype=np.float64)
        intervals = bw.intervals(chromosome, context_start, context_end)
        
        if intervals is None:
            continue
            
        for interval_start, interval_end, value in intervals:
            relative_start = interval_start - context_start
            relative_end = interval_end - context_start
            vals[relative_start:relative_end] = value
            
        scores = np.round(vals, 2)

        if reverse_complement:
            ref_sequence = ref_sequence.reverse_complement()
            scores = scores[::-1].copy()
            full_degeneracy = full_degeneracy[::-1]

        # Tokenize sequence
        ref_sequence_tokens = tokenizer.tokenizeMSA(ref_sequence)

        # Ensure everything has the same length
        min_len = min(len(ref_sequence_tokens), len(scores), len(full_degeneracy))
        ref_sequence_tokens = ref_sequence_tokens[:min_len]
        scores = scores[:min_len]
        full_degeneracy = full_degeneracy[:min_len]

        if len(ref_sequence_tokens) > 2048:  # Keep your existing length limit
            ref_sequence_tokens = ref_sequence_tokens[:2048]
            scores = scores[:2048]
            full_degeneracy = full_degeneracy[:2048]

        sequences.append(ref_sequence_tokens)
        scores_list.append(scores)
        degeneracies.append(full_degeneracy)

    return sequences, scores_list, degeneracies

def evaluate_model_and_get_predictions(model, dataloader, device):
    """Modified evaluation function to handle context."""
    model.eval()
    total_ce_loss = 0
    total_gaussian_loss = 0
    num_batches = 0
    total_tokens = 0
    total_seqs = 0
    total_accuracy = 0
    conservation_logits = []
    true_phyloP = []
    true_degeneracies = []
    
    with torch.no_grad():
        for batch in dataloader:
            output = step(model, batch, None, None, training=False)
            num_batches += 1
            
            total_tokens += output["n_processed"]
            total_seqs += output["n_seqs"]
            total_ce_loss += output["cross_entropy_loss"]
            total_gaussian_loss += output["gaussian_loss"]
            total_accuracy += output["accuracy"]
            
            # Get next-position predictions by shifting
            scaling_logits = output["scaling_logits"][:, :-1]  # Remove last prediction
            conservation_tgt = output["conservation_tgt"][:, 1:]  # Shift targets right
            degeneracies_tgt = output["degeneracies_tgt"][:, 1:]  # Shift targets right
            
            conservation_logits.append(scaling_logits)
            true_phyloP.append(conservation_tgt)
            true_degeneracies.append(degeneracies_tgt)

    # Process tensors
    max_len = max([tensor.size(1) for tensor in conservation_logits])
    conservation_logits = [torch.nn.functional.pad(tensor, (0, 0, 0, max_len - tensor.size(1)), value=-100) 
                          for tensor in conservation_logits]
    true_phyloP = [torch.nn.functional.pad(tensor, (0, max_len - tensor.size(1)), value=-100) 
                   for tensor in true_phyloP]
    true_degeneracies = [torch.nn.functional.pad(tensor, (0, max_len - tensor.size(1)), value=-100) 
                         for tensor in true_degeneracies]

    conservation_logits = torch.cat(conservation_logits, dim=0)
    true_phyloP = torch.cat(true_phyloP, dim=0)
    true_degeneracies = torch.cat(true_degeneracies, dim=0)

    avg_accuracy = total_accuracy / num_batches
    avg_ce_loss = total_ce_loss / num_batches
    avg_gaussian_loss = total_gaussian_loss / num_batches

    return avg_accuracy, avg_ce_loss, avg_gaussian_loss, conservation_logits, true_phyloP, true_degeneracies

def check_predicted_degeneracies(conservation_logits, true_phyloP, true_degeneracies):
    """Evaluate PhyloP predictions against next-position degeneracies."""
    avg_scores_by_degeneracy = {0: [], 1: [], 2: [], 3: [], 4: []}
    std_scores_by_degeneracy = {0: [], 1: [], 2: [], 3: [], 4: []}
    
    for logits, tgt, degeneracies in zip(conservation_logits, true_phyloP, true_degeneracies):
        mean = logits[:, 0]
        log_var = logits[:, 1]
        
        # Only look at valid positions (not padding and actual degeneracy sites)
        mask = (tgt != -100) & (degeneracies >= 0) & (degeneracies <= 4)
        mean = mean[mask]
        log_var = log_var[mask]
        degeneracies = degeneracies[mask]
        
        mean = mean.cpu().numpy()
        log_var = log_var.cpu().numpy()
        degeneracies = degeneracies.cpu().numpy()
        
        for deg in [0, 1, 2, 3, 4]:
            deg_mask = degeneracies == deg
            if np.any(deg_mask):
                avg_scores_by_degeneracy[deg].append(np.mean(mean[deg_mask]))
                std_scores_by_degeneracy[deg].append(np.std(mean[deg_mask]))
    
    print("\nPREDICTED NEXT-TOKEN PHYLOP SCORES:")
    for deg in [0, 1, 2, 3, 4]:
        if avg_scores_by_degeneracy[deg]:
            avg_score = np.mean(avg_scores_by_degeneracy[deg])
            sem = np.std(avg_scores_by_degeneracy[deg], ddof=1) / np.sqrt(len(avg_scores_by_degeneracy[deg]))

            num_sites = len(avg_scores_by_degeneracy[deg])
            print(f"Degeneracy {deg}-fold sites:")
            print(f"  Average predicted conservation: {avg_score:.3f} ± {sem:.3f} (SEM)")
            print(f"  Number of sites: {num_sites}")
        else:
            print(f"No scores for {deg}-fold sites")

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
            #print(f"gene start {start}, gene end {end}")
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
        #print(f"continuous_stretches: {continuous_stretches}")

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

import os
import pickle

def compute_per_token_perplexities(model, dataloader, degeneracies_list, device="cuda", output_file=None):
    model.eval()
    model.to(device)

    all_perplexities = []
    all_degeneracies = []
    from collections import defaultdict
    per_deg_perplexities = defaultdict(list) 

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Per-token perplexity")):
            out, lbls = batch
            #print(f"type of out: {type(out)}, should be torch tensor")
            #print(f"shape of out: {(out.shape)}, should be batch, 2, seq_len")
            #print(f"type of lbls: {type(lbls)}, should be torch tensor")
            #print(f"shape of lbls: {(lbls.shape)}, should be batch, 2, seq_len")
            #out is batch, 2, seq_len where the first item of second dimension is sequence and second is scaling
            cons_scores = out[:, 1, :]
            #print(f"shape of cons_scores: {cons_scores.shape}, should be (batch, seq_len)")
            sequence = out[:, 0, :]
            #print(f"shape of sequence: {sequence.shape}, should be (batch, seq_len)")
            #lbls is the same format as out, so seq_lbls is the first item of dim 2, second is scaling_lbls
            seq_lbls = lbls[:, 0, :]
            #print(f"shape of seq_lbls: {seq_lbls.shape}, should be (batch, seq_len)")
            scaling_lbls = lbls[:, 1, :]
            #print(f"shape of scaling_lbls: {scaling_lbls.shape}, should be (batch, seq_len)")
            sequence = sequence[0].to(device)        # (seq_len,)
            #print(f"shape of sequence: {sequence.shape}, should be: (seq_len,)")
            cons_scores = cons_scores[0].to(device)  # (seq_len,)
            #print(f"shape of cons_scores: {cons_scores.shape}, should be: (seq_len,)")
            degeneracies = degeneracies_list[batch_idx]
            #print(f"shape of degeneracies: {len(degeneracies)}, should be: (seq_len,)")

            seq_len = sequence.size(0)
            print(f"Sequence length: {seq_len}, Batch index: {batch_idx}")
            print(f"Degeneracies: {degeneracies}")
            for pos in range(1, seq_len -1 ):
                print("pos:", pos)
                input_tokens = sequence[:pos].unsqueeze(0)      # (1, pos)
                print(f"shape of input_tokens: {input_tokens.shape}, should be (1, pos)")
                input_scores = cons_scores[:pos].unsqueeze(0)   # (1, pos)
                print(f"shape of input_scores: {input_scores.shape}, should be (1, pos)")
                next_token = sequence[:pos].unsqueeze(0)            # (1, 1)
                print(f"shape of next_token: {next_token.shape}, should be (1, pos)")
                next_score = cons_scores[:pos].unsqueeze(0)         # (1, 1)
                print(f"shape of next_score: {next_score.shape}, should be (1, pos)")

                # Stack as (batch, 2, seq_len) → model expects this
                src = torch.stack([input_tokens, input_scores], dim=1)  # (1, 2, pos)
                print(f"shape of src: {src.shape}, should be (1, 2, pos)")
                tgt = torch.stack([next_token, next_score], dim=1)      # (1, 2, pos)
                print(f"shape of tgt: {tgt.shape}, should be (1, 2, pos)")
                #print degeneracies unless at the end of the sequence
                # if pos < seq_len - 2:
                #     print(f"Degeneracy at pos {pos}: {degeneracies[pos]}")
                try:
                    outputs = model(src, tgt)
                    logits = outputs["seq_logits"]  # (1, pos, vocab_size)
                    print(f"seq_logits shape: {outputs['seq_logits'].shape}")
                    log_probs = F.log_softmax(logits[0, -1], dim=-1)
                    #need to get the next token from tgt, which is the last pos of the first dimension, tgt shape is (1, 2, pos) i need to ignore batch, then get the first item of dim 2 at pos
                    next_token = tgt[0, 0, -1]
                    print("next_token dtype:", tgt[0, 0, -1].dtype, "value:", tgt[0, 0, -1])
                    next_token = int(next_token.item())  # Convert to int for indexing
                    print(f"next_token: {next_token}, log_probs shape: {log_probs.shape}")
                    nll = -log_probs[next_token]
                    perplexity = torch.exp(nll).item()
                    print(f"✅ Success at batch {batch_idx}, pos {pos} — perplexity: {perplexity:.3f}")


                    all_perplexities.append(perplexity)
                    all_degeneracies.append(degeneracies[pos])
                    per_deg_perplexities[degeneracies[pos]].append(perplexity) 
                except Exception as e:
                    print(f"Error at batch {batch_idx}, pos {pos}: {e}")
                    continue

            #break
    # Convert to numpy arrays for easier handling
    # === Mean ± SEM per degeneracy level ===
    if not per_deg_perplexities:
        print("⚠️ No valid perplexities were collected. Check for model output issues.")

    print("\n📊 Mean ± SEM Perplexity by Degeneracy:")
    #don't include -100 as a key
    per_deg_perplexities = {k: v for k, v in per_deg_perplexities.items() if k != -100}
    for deg in sorted(per_deg_perplexities.keys()):
        values = per_deg_perplexities[deg]
        mean = np.mean(values)
        sem = np.std(values, ddof=1) / np.sqrt(len(values))
        print(f"  Degeneracy {deg}-fold: {mean:.3f} ± {sem:.3f}  (N = {len(values)})")

    return all_perplexities, all_degeneracies



def main():
    parser = argparse.ArgumentParser(description="Token-wise perplexity vs degeneracy")
    parser.add_argument('--genome_fasta', type=str,
                        default='/home/mica/gamba/data_processing/data/240-mammalian/hg38.ml.fa')
    parser.add_argument('--big_wig', type=str,
                        default='/home/mica/gamba/data_processing/data/240-mammalian/241-mammalian-2020v2.bigWig')
    parser.add_argument('--output_file', type=str,
                        default='/home/mica/gamba/data_processing/data/degeneracy/chr22/')
    parser.add_argument('--config_fpath', type=str,
                        default='/home/mica/gamba/configs/jamba-small-240mammalian.json')
    parser.add_argument('--chr_coding_sites', type=str,
                        default='/home/mica/gamba/data_processing/data/240-mammalian/chr22_degenotate/degeneracy-all-sites.bed')
    parser.add_argument('--target_chrom', type=str, default='chr22')
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
        #print(f"Gene positions: {gene_positions}")
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #check_continuous_stretches_bed_file(bed_output_file)
    check_continuous_stretches_bed_file(bed_output_file)
    #load the bed_output_file
    exon_bed_df = load_exon_bed_file(bed_output_file)

    # Load genome
    genome = Fasta(args.genome_fasta)

    # Load bigwig file
    bw = pyBigWig.open(args.big_wig)

    # Get checkpoint path with step=5400
    # ckpt_dir = os.getenv("AMLT_OUTPUT_DIR", "/tmp/") 
    # ckpt_path = get_latest_dcp_checkpoint_path(ckpt_dir, 18000)
    ckpt_path = "/home/mica/gamba/clean_dcps/dcp_nocons_56000"

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
    d_model = config.get("d_model", 512) #576/2
    nhead = config.get("n_head", 8)  
    n_layers = config.get("n_layers", 6)
    dim_feedforward = config.get("dim_feedforward", d_model)
    padding_id = config.get("padding_id", 0)

    # Set up the model load from last checkpoint
    model = JambaGambaNoConsModel(
            model, d_model=d_model, nhead=nhead, n_layers=n_layers, padding_id=0, dim_feedfoward=dim_feedforward
        )

    # Load the model checkpoint
    checkpoint = torch.load(os.path.join(ckpt_path, "model_optimizer.pt"), map_location=device)
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
    model.to(device)
    model.eval()


    collator = gLMCollator(
        tokenizer=tokenizer,
        pad_to_multiple_of=None,
        test=True,
    )
    #exon sequences


    # Prepare sequences and degeneracies
    exon_sequences, exon_scores, exon_degeneracies = process_bed_file(exon_bed_df, genome, bw, tokenizer)

    exon_dataset = SequenceDataset(exon_sequences, exon_scores)  # Note: degeneracies not included
    exon_dataloader = DataLoader(exon_dataset, batch_size=1, collate_fn=collator)
    print("number of examples in exon dataset:", len(exon_dataset))

    # Zip degeneracies outside the dataloader
    perps, degs = compute_per_token_perplexities(model, exon_dataloader, exon_degeneracies, device="cuda")

    # Print per-degeneracy perplexity stats
    for deg in sorted(set(degs)):
        mask = [d == deg for d in degs]
        scores = [p for p, m in zip(perps, mask) if m]
        avg_score = sum(scores) / len(scores)
        print(f"Average perplexity for {deg}-fold sites: {avg_score:.3f} (Number of sites: {len(scores)})")


if __name__ == "__main__":
    main()
