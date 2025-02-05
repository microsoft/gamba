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

import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset


from sequence_models.samplers import SortishSampler, ApproxBatchSampler
from sequence_models.utils import transformer_lr, warmup

import torch.nn.functional as F 
from evodiff.utils import Tokenizer
from gamba.collators import gLMCollator
from gamba.model import create_model, JambagambaModel
from gamba.constants import TaskType, DNA_ALPHABET_PLUS
import pyBigWig
import json

class SequenceDataset(Dataset):
    def __init__(self, sequences, scores):
        self.sequences = sequences
        self.scores = scores

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.scores[idx]
    
def safe_mean(arr, axis=None):
    """Calculate mean while handling empty arrays and NaN values."""
    if len(arr) == 0 or (isinstance(arr, np.ndarray) and arr.size == 0):
        return np.nan
    return np.nanmean(arr, axis=axis)

def get_representations(model, dataloader, device, original_spans):
    model.eval()
    representations = []
    conservations = []
    variances = []
    
    # Special tokens to mask
    special_tokens = [-100, 8, 9]
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            sequences, scores = batch
            sequences = sequences.to(device)
            scores = scores.to(device)
            output = model(sequences, scores)
            
            # Get raw tensors
            batch_representations = output["representation"].cpu().numpy()  # [batch, seq_len+2, hidden]
            scaling_logits = output["scaling_logits"].cpu().numpy()  # [batch, seq_len, 2]
            sequence_data = sequences.cpu().numpy()  # [batch, 2, seq_len+2]
            
            print(f"\nBatch {batch_idx} shapes:")
            print(f"Sequence data: {sequence_data.shape}")
            print(f"Batch representations: {batch_representations.shape}")
            print(f"Scaling logits: {scaling_logits.shape}")
            
            # Get sequence length for scaling logits
            seq_len = scaling_logits.shape[1]
            
            # Split scaling logits into mean and variance
            batch_conservations = scaling_logits[..., 0]  # [batch, seq_len]
            batch_log_var = scaling_logits[..., 1]  # [batch, seq_len]
            batch_variances = np.exp(batch_log_var)
            
            for idx in range(len(batch_representations)):
                span_idx = batch_idx * dataloader.batch_size + idx
                if span_idx < len(original_spans):
                    start, end = original_spans[span_idx]
                    
                    if start < end and end <= seq_len:
                        # Get sequence tokens for this span
                        seq_tokens = sequence_data[idx, 0, start:end]  # Take first channel
                        valid_mask = ~np.isin(seq_tokens, special_tokens)
                        
                        # Create expanded mask for representations
                        repr_mask = np.expand_dims(valid_mask, -1)
                        repr_mask = np.tile(repr_mask, (1, batch_representations.shape[-1]))
                        
                        # Get slices
                        repr_slice = batch_representations[idx, start:end]
                        cons_slice = batch_conservations[idx, start:end]
                        var_slice = batch_variances[idx, start:end]
                        
                        # Apply masks
                        repr_slice_masked = repr_slice[valid_mask]
                        cons_slice_masked = cons_slice[valid_mask]
                        var_slice_masked = var_slice[valid_mask]
                        
                        if len(repr_slice_masked) > 0:
                            repr_mean = np.mean(repr_slice_masked, axis=0)
                            cons_mean = np.mean(cons_slice_masked)
                            var_mean = np.mean(var_slice_masked)
                            
                            representations.append(repr_mean)
                            conservations.append(cons_mean)
                            variances.append(var_mean)
                            
                            print(f"\nSample {idx}:")
                            print(f"Valid tokens: {np.sum(valid_mask)}/{len(valid_mask)}")
                            print(f"Conservation: {cons_mean:.3f}")
                            print(f"Variance: {var_mean:.3f}")
            
            del sequences, scores, output
            torch.cuda.empty_cache()
    
    if not representations:
        print("Warning: No valid representations generated")
        return np.array([]), np.array([]), np.array([])
    
    repr_array = np.array(representations)
    cons_array = np.array(conservations)
    var_array = np.array(variances)
    
    print("\nFinal statistics:")
    print(f"Total samples: {len(repr_array)}")
    print(f"Conservation mean: {np.mean(cons_array):.3f} ± {np.std(cons_array):.3f}")
    print(f"Variance mean: {np.mean(var_array):.3f} ± {np.std(var_array):.3f}")
    
    return repr_array, cons_array, var_array

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

def load_bed_file(bed_file):
    bed_df = pd.read_csv(bed_file, sep='\t', header=None, names=['chrom', 'start', 'end', 'label', 'info'])
    return bed_df

def process_bed_file(bed_df, genome, chrom_sizes, bw, tokenizer):
    sequences = []
    scores_list = []
    original_spans = []
    target_length = 2048
    
    valid_chromosomes = "chr1 chr2 chr3 chr4 chr5 chr6 chr7 chr8 chr9 chr10 chr11 chr12 chr13 chr14 chr15 chr16 chr17 chr18 chr19 chr20 chr21 chr22 chrX".split()
    chrom_sizes_df = pd.read_csv(chrom_sizes, sep='\t', header=None, names=['chrom', 'size'])
    
    skipped = 0
    for index, row in bed_df.iterrows():
        chromosome = row['chrom']
        if chromosome not in valid_chromosomes:
            skipped += 1
            continue
            
        chrom_size = chrom_sizes_df[chrom_sizes_df['chrom'] == chromosome]['size'].iloc[0]
        start = row['start']
        end = row['end']
        
        if start > end:
            start, end = end, start
            
        original_length = end - start
        pad_length = target_length - original_length
        
        # Calculate padded_start while ensuring we don't go beyond chromosome boundaries
        padded_start = max(0, start - pad_length)
        actual_start_offset = start - padded_start
        
        # Check if we'll exceed chromosome bounds
        if padded_start + target_length > chrom_size:
            skipped += 1
            continue
            
        # Track where original sequence starts in padded input
        original_spans.append((actual_start_offset, actual_start_offset + original_length))
        
        try:
            # Get sequence with padding
            sequence = genome[chromosome][padded_start:padded_start + target_length].seq
            sequence_tokens = tokenizer.tokenizeMSA(sequence)
            
            # Get conservation scores
            vals = np.zeros(target_length, dtype=np.float64)
            intervals = bw.intervals(chromosome, padded_start, padded_start + target_length)
            
            if intervals is not None:
                for interval_start, interval_end, value in intervals:
                    relative_start = interval_start - padded_start
                    relative_end = interval_end - padded_start
                    if 0 <= relative_start < target_length and 0 <= relative_end <= target_length:
                        vals[relative_start:relative_end] = value
                        
            scores = np.round(vals, 2)
            sequences.append(sequence_tokens)
            scores_list.append(scores)
            
        except Exception as e:
            print(f"Error processing {chromosome}:{start}-{end}: {str(e)}")
            skipped += 1
            continue
    
    print(f"Processed {len(sequences)} sequences, skipped {skipped} sequences")
    return sequences, scores_list, original_spans


def generate_random_regions(n_regions, genome_sizes_file, region_size=500):
    """Generate random genomic regions that are properly spaced."""
    chrom_sizes = pd.read_csv(genome_sizes_file, sep='\t', header=None, names=['chrom', 'size'])
    valid_chroms = [f'chr{i}' for i in range(1,23)] + ['chrX']
    chrom_sizes = chrom_sizes[chrom_sizes['chrom'].isin(valid_chroms)]
    
    random_regions = []
    target_length = 2048  # Match the target length used in process_bed_file
    
    while len(random_regions) < n_regions:
        chrom = np.random.choice(valid_chroms)
        chrom_size = chrom_sizes[chrom_sizes['chrom'] == chrom]['size'].iloc[0]
        
        # Ensure we have enough space for padding
        max_pos = chrom_size - target_length
        if max_pos <= 0:
            continue
            
        start = np.random.randint(0, max_pos)
        end = start + region_size
        
        # Check if we have enough space for padding
        if start >= 0 and end <= chrom_size:
            random_regions.append([chrom, start, end])
    
    df = pd.DataFrame(random_regions, columns=['chrom', 'start', 'end'])
    df = df.sort_values(['chrom', 'start']).reset_index(drop=True)
    return df

def process_dataset(bed_path_or_df, genome, chrom_sizes, bw, tokenizer, 
                   model, device, collator, output_path, dataset_name, is_random=False):
    """Process a single dataset and return representations and scores."""
    if os.path.exists(output_path):
        print(f"Loading existing representations from {output_path}")
        data = np.load(output_path)
        return data['repr'], data['cons'], data['var']
    
    print(f"Processing sequences from {dataset_name}...")
    
    # Handle both DataFrame and file path inputs
    if isinstance(bed_path_or_df, pd.DataFrame):
        bed_df = bed_path_or_df
    else:
        bed_df = pd.read_csv(bed_path_or_df, sep='\t', header=None, 
                            names=['chrom', 'start', 'end', 'label', 'info'])
    
    sequences, cons_scores, spans = process_bed_file(bed_df, genome, chrom_sizes, bw, tokenizer)
    
    if not sequences:
        raise ValueError(f"No valid sequences processed for {dataset_name}")
        
    dataset = SequenceDataset(sequences, cons_scores)
    loader = DataLoader(dataset, batch_size=20, collate_fn=collator)
    representations, conservations, variances = get_representations(model, loader, device, spans)
    
    if np.all(np.isnan(conservations)) or np.all(np.isnan(variances)):
        print(f"Warning: All values are NaN for {dataset_name}")
    
    np.savez_compressed(output_path,
                       repr=representations,
                       cons=conservations,
                       var=variances)
    
    print(f"Saved {len(representations)} representations to {output_path}")
    return representations, conservations, variances

def ensure_2d(array):
    """Ensure array is 2D."""
    if len(array.shape) == 1:
        return array.reshape(1, -1)
    return array

def visualize_embeddings(repr1, repr2, labels1, labels2, output_path):
    """Create and save UMAP visualization of embeddings."""
    print(f"Shapes - Group 1: {repr1.shape}, Group 2: {repr2.shape}")
    
    # Ensure 2D arrays
    repr1 = ensure_2d(repr1)
    repr2 = ensure_2d(repr2)
    
    # Combine data
    combined_repr = np.concatenate([repr1, repr2])
    labels = [labels1] * len(repr1) + [labels2] * len(repr2)
    
    # Generate UMAP embeddings
    embeddings = umap.UMAP().fit_transform(combined_repr)
    
    # Create plot
    plt.figure(figsize=(10, 6))
    colors = {'UCNE': 'blue', 'random': 'orange'}
    
    for label in set(labels):
        mask = np.array(labels) == label
        plt.scatter(embeddings[mask, 0], embeddings[mask, 1], 
                   color=colors[label], label=label, s=5)
    
    plt.legend()
    plt.title('UMAP of Sequence Embeddings')
    plt.savefig(output_path)
    plt.close()


def print_stats(cons1, var1, cons2, var2, group1_name="Group 1", group2_name="Group 2"):
    """Print conservation and variance statistics."""
    print("\nStatistics:")
    print(f"{group1_name}:")
    print(f"  Conservation: {np.mean(cons1):.3f} ± {np.std(cons1):.3f}")
    print(f"  Variance: {np.mean(var1):.3f} ± {np.std(var1):.3f}")
    print(f"{group2_name}:")
    print(f"  Conservation: {np.mean(cons2):.3f} ± {np.std(cons2):.3f}")
    print(f"  Variance: {np.mean(var2):.3f} ± {np.std(var2):.3f}")


def main():
    parser = argparse.ArgumentParser(description="Get representations of sequences")
    parser.add_argument('--genome_fasta', type=str, default='/home/mica/gamba/data_processing/data/240-mammalian/hg38.ml.fa', help='Path to the genome FASTA file')
    parser.add_argument('--chrom_sizes', type=str, default='/home/mica/gamba/data_processing/data/240-mammalian/hg38.chrom.sizes', help='Path to the chromosome sizes file')
    parser.add_argument('--big_wig', type=str, default='/home/mica/gamba/data_processing/data/240-mammalian/241-mammalian-2020v2.bigWig', help='Path to the bigWig file')
    parser.add_argument('--output_dir', type=str, default='/home/mica/gamba/data_processing/data/conserved_elements/', help='Path to the output file')
    parser.add_argument('--config_fpath', type=str, default='/home/mica/gamba/configs/jamba-small-240mammalian.json', help='Path to the config file')
    parser.add_argument('--bed_file1', type=str, default ='/home/mica/gamba/data_processing/data/conserved_elements/hg19_UCNE_coordinates.bed', help='First BED file')
    parser.add_argument('--bed_file2', help='Second BED file (optional)')
    args = parser.parse_args()
    
    # Load data
    bed1 = load_bed_file(args.bed_file1)
    if args.bed_file2:
        bed2 = load_bed_file(args.bed_file2)
    else:
        bed2 = generate_random_regions(len(bed1), args.chrom_sizes)
    
    genome = Fasta(args.genome_fasta)
    bw = pyBigWig.open(args.big_wig)
    ckpt_dir = os.getenv("AMLT_OUTPUT_DIR", "/tmp/") 
    ckpt_path = get_latest_dcp_checkpoint_path(ckpt_dir, 18000)


    # Load model configuration
    with open(args.config_fpath, "r") as f:
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
    

    # Load the model checkpoint
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

    # Move device to cuda if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    collator = gLMCollator(
        tokenizer=tokenizer,
        pad_to_multiple_of=None,
        test=True,
    )
    # Generate keywords from bed files
    bed1_name = os.path.splitext(os.path.basename(args.bed_file1))[0]
    #bed1_filename is: hg19_UCNE_coordinates
    bed1_filename = f"representations_{bed1_name}.npz"
    if args.bed_file2:
        bed2_name = os.path.splitext(os.path.basename(args.bed_file2))[0]
        bed2_filename = f"representations_{bed2_name}.npz"
    else:
        bed2_filename = f"representations_{bed1_name}_random.npz"
    
    bed1_repr_path = os.path.join(args.output_dir, bed1_filename)
    bed2_repr_path = os.path.join(args.output_dir, bed2_filename)

    print(f"bed1_repr_path: {bed1_repr_path}, bed2_repr_path: {bed2_repr_path}")
    print(f"bed1_filename: {bed1_filename}, bed2_filename: {bed2_filename}")
    
     # Process both datasets
    repr1, cons1, var1 = process_dataset(
        bed1, genome, args.chrom_sizes, bw, tokenizer,
        model, device, collator, bed1_repr_path, "UCNE dataset"
    )
    
    repr2, cons2, var2 = process_dataset(
        bed2, genome, args.chrom_sizes, bw, tokenizer,
        model, device, collator, bed2_repr_path, "random dataset"
    )
    
    # Visualize results
    plot_path = f'{args.output_dir}/umap_plot_{bed1_filename}_{bed2_filename}.png'
    visualize_embeddings(repr1, repr2, 'UCNE', 'random', plot_path)
    
    # Print statistics
    print_stats(cons1, var1, cons2, var2, "UCNE regions", "Random regions")


if __name__ == "__main__":
    main()


