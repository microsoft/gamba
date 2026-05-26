import argparse
import numpy as np
import torch
import pickle
import logging
import pyBigWig
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def compute_histogram_streaming(bw, chroms, num_bins=1000, sample_ratio=0.01, chunk_size=1000000):
    """
    Compute histogram of phyloP scores using streaming to avoid memory issues.
    """
    # [Same implementation as before]
    logging.info("Finding min/max values to establish histogram bins...")
    
    # First pass: find min and max values by sampling
    min_val = float('inf')
    max_val = float('-inf')
    total_positions = 0
    
    for chrom in tqdm(chroms, desc="Finding min/max values"):
        chrom_length = bw.chroms()[chrom]
        total_positions += chrom_length
        
        # Process chromosome in chunks to avoid memory issues
        for start in range(0, chrom_length, chunk_size):
            end = min(start + chunk_size, chrom_length)
            
            # Only sample a fraction of positions
            if np.random.random() <= sample_ratio:
                try:
                    values = np.array(bw.values(chrom, start, end))
                    # Filter out NaN values
                    values = values[~np.isnan(values)]
                    
                    if len(values) > 0:
                        min_val = min(min_val, np.min(values))
                        max_val = max(max_val, np.max(values))
                except Exception as e:
                    logging.warning(f"Error processing {chrom}:{start}-{end}: {e}")
    
    logging.info(f"Min value: {min_val}, Max value: {max_val}")
    
    # Set up bins for histogram
    bin_edges = np.linspace(min_val, max_val, num_bins + 1)
    hist_counts = np.zeros(num_bins, dtype=np.int64)
    
    # Second pass: compute histogram
    logging.info("Computing histogram...")
    for chrom in tqdm(chroms, desc="Computing histogram"):
        chrom_length = bw.chroms()[chrom]
        
        # Process chromosome in chunks
        for start in range(0, chrom_length, chunk_size):
            end = min(start + chunk_size, chrom_length)
            
            # Only sample a fraction of positions
            if np.random.random() <= sample_ratio:
                try:
                    values = np.array(bw.values(chrom, start, end))
                    # Filter out NaN values
                    values = values[~np.isnan(values)]
                    
                    if len(values) > 0:
                        # Update histogram
                        chunk_hist, _ = np.histogram(values, bins=bin_edges)
                        hist_counts += chunk_hist
                except Exception as e:
                    logging.warning(f"Error processing {chrom}:{start}-{end}: {e}")
    
    # Adjust counts to account for sampling
    hist_counts = (hist_counts / sample_ratio).astype(np.int64)
    
    return bin_edges, hist_counts, total_positions

def compute_uniform_distribution_weights(hist_counts, bin_edges, max_weight=500.0, eps=1e-10):
    """
    Generate weights that transform the original distribution into a uniform distribution.
    
    Args:
        hist_counts: Original histogram counts
        bin_edges: Bin edges for the histogram
        max_weight: Maximum weight cap to avoid numerical issues
        eps: Small value to avoid division by zero
    """
    # Target a completely flat distribution - all bins should have equal contribution
    # The ideal target value is the average count across all bins
    target_value = np.mean(hist_counts[hist_counts > 0])  # Consider only non-empty bins
    
    # For perfect uniformity, each bin should contribute equally
    # So weights should be inversely proportional to counts
    raw_weights = np.ones_like(hist_counts, dtype=float)
    non_zero_mask = hist_counts > 0
    raw_weights[non_zero_mask] = target_value / (hist_counts[non_zero_mask] + eps)


    #re-emphasize the most conserved weights >3.0
    extreme_conservation_mask = bin_edges[:-1] > 3.0
    raw_weights[extreme_conservation_mask] *= 1.5  # Extra multiplier
    
    # Normalize weights to average 1.0
    normalized_weights = raw_weights / np.mean(raw_weights)
    
    # Cap extreme weights
    weights = np.minimum(normalized_weights, max_weight)
    
    return weights

def compute_adjusted_weights(hist_counts, bin_edges, min_weight=0.2, max_weight=50.0, conservation_boost=5.0, acceleration_boost=5.0, eps=1e-6):
    """
    Compute weights with conservation boost and minimum weight.
    
    Args:
        hist_counts: Histogram counts
        bin_edges: Bin edges for the histogram
        min_weight: Minimum weight to assign (even to most common scores)
        max_weight: Maximum weight to apply
        conservation_boost: Factor to boost weights for conserved regions
        acceleration_boost: Factor to boost weights for accelerated regions
        eps: Small value to add for numerical stability
        
    Returns:
        bin_weights: Weight for each bin
    """
    # Get bin centers
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Convert counts to frequencies
    frequencies = hist_counts / hist_counts.sum()
    
    # Compute initial weights as inverse of frequencies to the power
    raw_weights = 1.0 / (frequencies + eps)**2
    
    # Apply minimum weight
    raw_weights = np.maximum(raw_weights, min_weight * raw_weights.mean())
    
    # Apply boosting for positive scores (conserved regions)
    conservation_mask = bin_centers > 0.5  # Start boosting a bit away from zero
    raw_weights[conservation_mask] *= conservation_boost
    
    # Apply boosting for negative scores (accelerated regions)
    acceleration_mask = bin_centers < -2.0  # Boost more negative scores
    raw_weights[acceleration_mask] *= acceleration_boost
    
    # Additional boosting for extreme conservation (e.g., > 5)
    extreme_conservation_mask = bin_centers > 5.0
    raw_weights[extreme_conservation_mask] *= 2.0  # Extra multiplier
    
    # Additional boosting for extreme acceleration (e.g., < -10)
    extreme_acceleration_mask = bin_centers < -10.0
    raw_weights[extreme_acceleration_mask] *= 2.0  # Extra multiplier
    
    # Normalize weights to have mean = 1.0
    bin_weights = raw_weights / raw_weights.mean()
    
    # Apply max weight cap (much higher than before)
    bin_weights = np.minimum(bin_weights, max_weight)
    
    return bin_weights

def visualize_reweighted_distribution(weights_path, output_dir=None):
    """
    Visualize the original and re-weighted distributions of phyloP scores.
    
    Args:
        weights_path: Path to the precomputed weights pickle file
        output_dir: Directory to save the visualization (if None, will use same dir as weights)
    """
    # Load the precomputed weights
    with open(weights_path, 'rb') as f:
        weights_data = pickle.load(f)
    
    bin_edges = weights_data['bin_edges'].numpy()
    bin_weights = weights_data['bin_weights'].numpy()
    
    # Calculate bin centers
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Calculate bin widths for histogram plotting
    bin_widths = bin_edges[1:] - bin_edges[:-1]
    
    # To estimate the original distribution, we need to estimate frequency counts
    # Since we don't have the raw counts in the pickle file, we'll simulate them
    # by assuming the weights are roughly proportional to 1/frequency
    
    # We'll set a baseline frequency as if weights=1.0 corresponds to the average frequency
    baseline_frequency = 1000000  # Just an arbitrary large number
    
    # For bins with weight > 0, estimate original frequency
    estimated_frequencies = np.ones_like(bin_weights)
    non_zero_mask = bin_weights > 0
    estimated_frequencies[non_zero_mask] = baseline_frequency / bin_weights[non_zero_mask]
    
    # Calculate re-weighted frequencies (effectively flat distribution)
    reweighted_frequencies = estimated_frequencies * bin_weights
    
    # Create plot
    fig, axes = plt.subplots(3, 1, figsize=(12, 15))
    
    # Plot 1: Original distribution (estimated)
    axes[0].bar(bin_centers, estimated_frequencies, width=bin_widths, alpha=0.7)
    axes[0].set_yscale('log')
    axes[0].set_title('Estimated Original Distribution of phyloP Scores')
    axes[0].set_xlabel('phyloP Score')
    axes[0].set_ylabel('Frequency (log scale)')
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Weights
    axes[1].plot(bin_centers, bin_weights, 'r-', linewidth=2)
    axes[1].set_title('Weight per phyloP Score Bin')
    axes[1].set_xlabel('phyloP Score')
    axes[1].set_ylabel('Weight')
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Re-weighted distribution (what the model "sees")
    axes[2].bar(bin_centers, reweighted_frequencies, width=bin_widths, alpha=0.7, color='green')
    axes[2].set_title('Re-weighted Distribution (Effective Impact on Training)')
    axes[2].set_xlabel('phyloP Score')
    axes[2].set_ylabel('Effective Contribution')
    axes[2].grid(True, alpha=0.3)
    
    # Add zero line for reference on all plots
    for ax in axes:
        ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    
    # Adjust layout and save
    plt.tight_layout()
    
    if output_dir is None:
        output_dir = Path(weights_path).parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
    output_path = output_dir / "phylop_reweighted_distribution.png"
    plt.savefig(output_path)
    print(f"Saved visualization to {output_path}")
    
    return fig, axes

# Extended version that uses raw histogram counts if available
def visualize_reweighted_distribution_with_counts(weights_path, hist_counts=None, output_dir=None):
    """
    Visualize the original and re-weighted distributions of phyloP scores.
    
    Args:
        weights_path: Path to the precomputed weights pickle file
        hist_counts: Raw histogram counts if available (optional)
        output_dir: Directory to save the visualization (if None, will use same dir as weights)
    """
    # Load the precomputed weights
    with open(weights_path, 'rb') as f:
        weights_data = pickle.load(f)
    
    bin_edges = weights_data['bin_edges'].numpy()
    bin_weights = weights_data['bin_weights'].numpy()
    
    # Calculate bin centers
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Calculate bin widths for histogram plotting
    bin_widths = bin_edges[1:] - bin_edges[:-1]
    
    # Use actual histogram counts if provided, otherwise estimate
    if hist_counts is not None:
        # Use actual counts
        original_frequencies = hist_counts
    else:
        # Estimate original frequencies
        baseline_frequency = 1000000
        estimated_frequencies = np.ones_like(bin_weights)
        non_zero_mask = bin_weights > 0
        estimated_frequencies[non_zero_mask] = baseline_frequency / bin_weights[non_zero_mask]
        original_frequencies = estimated_frequencies
    
    # Calculate re-weighted frequencies
    reweighted_frequencies = original_frequencies * bin_weights
    
    # Create plot
    fig, axes = plt.subplots(3, 1, figsize=(12, 15))
    
    # Plot 1: Original distribution
    axes[0].bar(bin_centers, original_frequencies, width=bin_widths, alpha=0.7)
    axes[0].set_yscale('log')
    axes[0].set_title('Original Distribution of phyloP Scores')
    axes[0].set_xlabel('phyloP Score')
    axes[0].set_ylabel('Frequency (log scale)')
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Weights
    axes[1].plot(bin_centers, bin_weights, 'r-', linewidth=2)
    axes[1].set_title('Weight per phyloP Score Bin')
    axes[1].set_xlabel('phyloP Score')
    axes[1].set_ylabel('Weight')
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Re-weighted distribution
    axes[2].bar(bin_centers, reweighted_frequencies, width=bin_widths, alpha=0.7, color='green')
    axes[2].set_title('Re-weighted Distribution (Effective Impact on Training)')
    axes[2].set_xlabel('phyloP Score')
    axes[2].set_ylabel('Effective Contribution')
    axes[2].grid(True, alpha=0.3)
    
    # Add zero line for reference on all plots
    for ax in axes:
        ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    
    # Adjust layout and save
    plt.tight_layout()
    
    if output_dir is None:
        output_dir = Path(weights_path).parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
    output_path = output_dir / "phylop_reweighted_distribution.png"
    plt.savefig(output_path)
    print(f"Saved visualization to {output_path}")
    
    return fig, axes

def main():
    # Process command line arguments
    parser = argparse.ArgumentParser(
        description="Compute and save phyloP score weights based on frequency with conservation boost"
    )
    parser.add_argument(
        "--bigwig_file",
        type=str,
        default="/home/mica/gamba/data_processing/data/240-mammalian/241-mammalian-2020v2.bigWig",
        help="Path to the phyloP bigwig file",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="/home/mica/gamba/data_processing/data/240-mammalian/phyloP_weights.pkl",
        help="File name to save the weights",
    )
    parser.add_argument(
        "--num_bins",
        type=int,
        default=1000,  # Increased bins for better resolution
        help="Number of bins for the histogram",
    )
    parser.add_argument(
        "--max_weight",
        type=float,
        default=10.0,
        help="Maximum weight to apply",
    )
    parser.add_argument(
        "--min_weight",
        type=float,
        default=0.05,
        help="Minimum weight as a fraction of the mean weight",
    )
    parser.add_argument(
        "--conservation_boost",
        type=float,
        default=2.0,
        help="Factor to boost weights for conserved regions (positive scores)",
    )
    parser.add_argument(
        "--sample_ratio",
        type=float,
        default=0.01,
        help="Fraction of positions to sample (to speed up processing)",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Create visualization of the distribution and weights",
    )
    parser.add_argument(
        "--chroms",
        type=str,
        nargs="+",
        help="List of chromosomes to process (default: all)",
    )
    args = parser.parse_args()

    # Load the bigwig file
    logging.info(f"Opening BigWig file: {args.bigwig_file}")
    bw = pyBigWig.open(args.bigwig_file)
    

    #check if output file does not exist:
    if not Path(args.output_file).exists():
        # Get chromosomes
        all_chroms = list(bw.chroms().keys())
        if args.chroms:
            chroms = [c for c in args.chroms if c in all_chroms]
            logging.info(f"Processing specified chromosomes: {chroms}")
        else:
            chroms = all_chroms
            logging.info(f"Processing all {len(chroms)} chromosomes")
        
        # Compute histogram using streaming approach
        bin_edges, hist_counts, total_positions = compute_histogram_streaming(
            bw, 
            chroms, 
            num_bins=args.num_bins,
            sample_ratio=args.sample_ratio
        )
        
        # Compute adjusted weights with conservation boost
        # bin_weights = compute_adjusted_weights(
        #     hist_counts, 
        #     bin_edges, 
        #     min_weight=args.min_weight,
        #     max_weight=args.max_weight,
        #     conservation_boost=args.conservation_boost
        # )

        #compute target distribution weights
        bin_weights = compute_uniform_distribution_weights(
            hist_counts,
            bin_edges,
            max_weight=args.max_weight
        )
        
        # Convert to torch tensors
        bin_edges_tensor = torch.tensor(bin_edges, dtype=torch.float32)
        bin_weights_tensor = torch.tensor(bin_weights, dtype=torch.float32)
        
        # Create result dictionary
        result = {
            'bin_edges': bin_edges_tensor,
            'bin_weights': bin_weights_tensor,
            'num_bins': args.num_bins,
            'max_weight': args.max_weight,
            'min_weight': args.min_weight,
            'hist_counts': torch.tensor(hist_counts, dtype=torch.float32), 
            'conservation_boost': args.conservation_boost,
            'total_positions': total_positions,
            'chromosome_ids': chroms,
        }

        
        # Save to file
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'wb') as f:
            pickle.dump(result, f)
        
        logging.info(f"Saved weights to {output_path}")
    else:
        #read in the results data
        with open(args.output_file, 'rb') as f:
            result = pickle.load(f)
        bin_edges = result['bin_edges']
        bin_weights = result['bin_weights']
        hist_counts = result.get('hist_counts', None)


    # Visualize re-weighted distribution
    visualize_reweighted_distribution_with_counts(
        args.output_file,
        hist_counts=hist_counts,
        output_dir=Path(args.output_file).parent
    )
    
    # Visualize the distribution and weights if requested
    if args.visualize:
        viz_path = output_path.with_suffix('.png')
        
        plt.figure(figsize=(12, 10))
        
        # Plot 1: Distribution of phyloP scores
        plt.subplot(2, 1, 1)
        plt.hist(
            np.repeat(
                (bin_edges[:-1] + bin_edges[1:]) / 2, 
                hist_counts
            ), 
            bins=args.num_bins, 
            alpha=0.7
        )
        plt.yscale('log')
        plt.title('Distribution of phyloP Scores')
        plt.xlabel('phyloP Score')
        plt.ylabel('Frequency (log scale)')
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Weights per bin
        plt.subplot(2, 1, 2)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        plt.plot(bin_centers, bin_weights, 'r-', linewidth=2)
        plt.title('Weight per phyloP Score Bin')
        plt.xlabel('phyloP Score')
        plt.ylabel('Weight')
        plt.grid(True)
        
        # Add zero line for reference
        plt.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        
        # Save the figure
        plt.tight_layout()
        plt.savefig(viz_path)
        logging.info(f"Saved visualization to {viz_path}")
    
    bw.close()
    logging.info("Done!")

if __name__ == "__main__":
    main()