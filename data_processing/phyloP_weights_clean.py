import argparse
import numpy as np
import torch
import pickle
import logging
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import glob

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def compute_histogram_from_npy(data_dir, subdirs=["train", "test", "valid"], pattern="*_conservation_small.npy", num_bins=1000, sample_ratio=0.01):
    """
    Compute histogram of phyloP scores by reading .npy files from multiple subdirectories.
    
    Args:
        data_dir: Base directory containing subdirectories with .npy files
        subdirs: List of subdirectories to search in (default: ["train", "test", "valid"])
        pattern: Glob pattern to match relevant files
        num_bins: Number of bins for the histogram
        sample_ratio: Fraction of data to sample to speed up processing
        
    Returns:
        bin_edges: Edges of the histogram bins
        hist_counts: Counts for each bin
        total_positions: Total number of positions processed
    """
    logging.info(f"Finding .npy files in {data_dir}/{{{','.join(subdirs)}}} matching pattern {pattern}")
    
    # Get list of all matching files from all subdirectories
    data_dir = Path(data_dir)
    npy_files = []
    
    for subdir in subdirs:
        subdir_path = data_dir / subdir
        if subdir_path.exists():
            subdir_files = list(subdir_path.glob(pattern))
            npy_files.extend(subdir_files)
            logging.info(f"Found {len(subdir_files)} files in {subdir}")
        else:
            logging.warning(f"Subdirectory {subdir_path} does not exist")
    
    if not npy_files:
        raise FileNotFoundError(f"No files matching {pattern} found in {data_dir}")
    
    logging.info(f"Found {len(npy_files)} .npy files")
    
    # First pass: find min and max values by sampling
    min_val = float('inf')
    max_val = float('-inf')
    total_positions = 0
    
    logging.info("Finding min/max values to establish histogram bins...")
    for npy_file in tqdm(npy_files, desc="Finding min/max values"):
        try:
            # Load the conservation scores from the .npy file
            conservation_scores = np.load(npy_file)
            
            # Count total positions
            total_positions += conservation_scores.size
            
            # Sample data for efficiency
            if sample_ratio < 1.0:
                mask = np.random.random(conservation_scores.shape) <= sample_ratio
                sampled_scores = conservation_scores[mask]
            else:
                sampled_scores = conservation_scores
            
            # Filter out NaN values
            sampled_scores = sampled_scores[~np.isnan(sampled_scores)]
            
            if len(sampled_scores) > 0:
                min_val = min(min_val, np.min(sampled_scores))
                max_val = max(max_val, np.max(sampled_scores))
                
        except Exception as e:
            logging.warning(f"Error processing {npy_file}: {e}")
    
    logging.info(f"Min value: {min_val}, Max value: {max_val}")
    
    # Set up bins for histogram
    bin_edges = np.linspace(min_val, max_val, num_bins + 1)
    hist_counts = np.zeros(num_bins, dtype=np.int64)
    
    # Second pass: compute histogram
    logging.info("Computing histogram...")
    for npy_file in tqdm(npy_files, desc="Computing histogram"):
        try:
            # Load the conservation scores from the .npy file
            conservation_scores = np.load(npy_file)
            
            # Sample data for efficiency
            if sample_ratio < 1.0:
                mask = np.random.random(conservation_scores.shape) <= sample_ratio
                sampled_scores = conservation_scores[mask]
            else:
                sampled_scores = conservation_scores
            
            # Filter out NaN values
            sampled_scores = sampled_scores[~np.isnan(sampled_scores)]
            
            if len(sampled_scores) > 0:
                # Update histogram
                chunk_hist, _ = np.histogram(sampled_scores, bins=bin_edges)
                hist_counts += chunk_hist
                
        except Exception as e:
            logging.warning(f"Error processing {npy_file}: {e}")
    
    # Adjust counts to account for sampling
    if sample_ratio < 1.0:
        hist_counts = (hist_counts / sample_ratio).astype(np.int64)
    
    return bin_edges, hist_counts, total_positions

def compute_adjusted_weights(hist_counts, bin_edges, eps=1e-6):
    """
    Compute weights for each bin based on the inverse of the frequency of the bin.
    
    Args:
        hist_counts: Histogram counts
        bin_edges: Bin edges for the histogram
        eps: Small value to add for numerical stability
        
    Returns:
        bin_weights: Weight for each bin
    """
    # Get bin centers
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Convert counts to frequencies
    frequencies = hist_counts / hist_counts.sum()
    
    # Compute initial weights as inverse of frequencies to the power
    raw_weights = 1.0 / (frequencies + eps)


    #if bin_center is < 0, divide weight by 10
    bin_weights = raw_weights
    bin_weights[bin_centers < 0] = raw_weights[bin_centers < 0] / 10
    
    bin_weights = raw_weights + 100
    
    return bin_weights

def compute_uniform_distribution_weights(hist_counts, bin_edges, eps=1e-10):
    """
    Generate weights that transform the original distribution into a more uniform distribution
    with emphasis on conservation.
    
    Args:
        hist_counts: Original histogram counts
        bin_edges: Bin edges for the histogram
        eps: Small value to avoid division by zero
    """
        # Target a completely flat distribution - all bins should have equal contribution
    
    #take negative log of frequency as weight
    # weights = np.minimum(-np.log(hist_counts + eps)*1000 + 20000, np.ones_like(hist_counts)*10000)
    weights = (1 / (hist_counts + eps)) + 1

    
    return weights


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
        
    output_path = output_dir / "phylop_reweighted_distribution_clean.png"
    plt.savefig(output_path)
    print(f"Saved visualization to {output_path}")
    
    return fig, axes

def main():
    # Process command line arguments
    parser = argparse.ArgumentParser(
        description="Compute and save phyloP score weights from .npy files"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/home/mica/gamba/data_processing/data/240-mammalian",
        help="Base directory containing train/test/valid subdirectories with .npy files",
    )
    parser.add_argument(
        "--subdirs",
        type=str,
        nargs="+",
        default=["train", "test", "valid"],
        help="Subdirectories to search for .npy files",
    )
    parser.add_argument(
        "--file_pattern",
        type=str,
        default="*_conservation_small.npy",
        help="Pattern to match relevant .npy files",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="/home/mica/gamba/data_processing/data/240-mammalian/phyloP_weights_clean.pkl",
        help="File name to save the weights",
    )
    parser.add_argument(
        "--num_bins",
        type=int,
        default=1000,
        help="Number of bins for the histogram",
    )
    parser.add_argument(
        "--sample_ratio",
        type=float,
        default=0.05,
        help="Fraction of positions to sample (to speed up processing)",
    )
    parser.add_argument(
        "--weight_method",
        type=str,
        choices=["inverse_frequency", "uniform", "balanced"],
        default="balanced",
        help="Method to compute weights: inverse_frequency, uniform, or balanced (similar to bigwig result)",
    )
    args = parser.parse_args()

    # Check if output file already exists
    output_path = Path(args.output_file)
    if not output_path.exists():
        # Compute histogram from .npy files
        bin_edges, hist_counts, total_positions = compute_histogram_from_npy(
            args.data_dir,
            subdirs=args.subdirs,
            pattern=args.file_pattern,
            num_bins=args.num_bins,
            sample_ratio=args.sample_ratio
        )
        
        # Compute weights based on selected method
        if args.weight_method == "uniform":
            bin_weights = compute_uniform_distribution_weights(
                hist_counts,
                bin_edges,
            )
        else:
            bin_weights = compute_adjusted_weights(
                hist_counts,
                bin_edges,
            )
        # Convert to torch tensors
        bin_edges_tensor = torch.tensor(bin_edges, dtype=torch.float32)
        bin_weights_tensor = torch.tensor(bin_weights, dtype=torch.float32)
        hist_counts_tensor = torch.tensor(hist_counts, dtype=torch.float32)
        
        # Create result dictionary
        result = {
            'bin_edges': bin_edges_tensor,
            'bin_weights': bin_weights_tensor,
            'hist_counts': hist_counts_tensor,
            'num_bins': args.num_bins,
            'total_positions': total_positions,
            'weight_method': args.weight_method,
        }
        
        # Save to file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'wb') as f:
            pickle.dump(result, f)
        
        logging.info(f"Saved weights to {output_path}")
    else:
        # Load existing data for visualization
        logging.info(f"Output file {output_path} already exists, loading for visualization")
        with open(output_path, 'rb') as f:
            result = pickle.load(f)
        
        if 'hist_counts' in result:
            hist_counts = result['hist_counts']
        else:
            hist_counts = None

    # Visualize re-weighted distribution
    visualize_reweighted_distribution_with_counts(
        args.output_file,
        hist_counts=hist_counts,
        output_dir=output_path.parent
    )
    
    logging.info("Done!")

if __name__ == "__main__":
    main()