import argparse
import argparse
import numpy as np

import matplotlib.pyplot as plt
import pandas as pd
import os
import logging
import json

_logger = logging.getLogger(__name__)



def extract_scale_values(scaling_features, chrom_num, chrom_size):
    # open the right scaling features file scaling_features/[chrom_num]_features.out
    scaling_features_file = f"{scaling_features}chr{chrom_num}_features.out"

    df = pd.read_csv(scaling_features_file, sep="\t")

    #print column names
    print(df.columns)

    # extract the "scale" column
    scale_values = df['scale'].tolist()

    # check if length of scale_values matches chromosome size
    if len(scale_values) != chrom_size:
        raise ValueError(f"Length of scale values ({len(scale_values)}) does not match chromosome size ({chrom_size})")

    return scale_values

def extract_gaps( gaps_file, chrom_size):
    print("in extract gaps!")
    # load gap counts from the .npz file
    with np.load(gaps_file) as data:
        gaps = data['counts']

    # check if length of gaps matches chromosome size
    if len(gaps) != chrom_size:
        raise ValueError(f"Length of gaps ({len(gaps)}) does not match chromosome size ({chrom_size})")

    return gaps

def plot_dist(
    bed: pd.DataFrame,
    file_path: str,
    splits_file: str,
    scaling_features: str,
    gaps_file_path: str,
    verbose: bool = True,
):

    # use the splits json to save the numpy array as a compressed numpy file by chrom_num
    # read in the splits file
    with open(splits_file, "r") as f:
        splits = json.load(f)

    # create a dictionary to map chromosomes to splits
    chromosome_splits = {}
    for split, chroms in splits.items():
        for chrom in chroms:
            chromosome_splits[chrom] = split

    for index, row in bed.iterrows():
        chrom = row["chrom"]
        chrom_num = chrom.split("chr")[1]
        if chrom_num not in ["18", "19", "20", "21", "22"]:
            continue
        size = row["end"]
        chrom_num = chrom.split("chr")[1]
        gaps_file = f"{gaps_file_path}chr{chrom_num}_gap_counts.npz"
        gaps = extract_gaps(gaps_file, size)
        print("max of gaps value:", max(gaps))
        print("min of gaps value:", min(gaps))
        print("extracted gaps successfully for chrom:", chrom_num)
        scale_values = extract_scale_values(scaling_features, chrom_num, size)
        scale_values = np.array(scale_values, dtype=np.float64)
        print("min of the scaling features:", min(scale_values))
        print("max of the scaling features:", max(scale_values))
        bin_size = 1000
        num_bins = size // bin_size
        binned_scale_values = np.mean(scale_values[:num_bins * bin_size].reshape(-1, bin_size), axis=1)
        binned_gaps = np.mean(gaps[:num_bins * bin_size].reshape(-1, bin_size), axis=1)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
        
        ax1.scatter(range(num_bins), binned_gaps, label='Gaps', color='tab:blue', s=10)
        ax1.set_ylabel('Gaps', color='tab:blue')
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        ax1.set_xlim(0, num_bins)
        ax1.set_ylim(0, 240)
        ax1.legend(loc='upper right')

        ax2.scatter(range(num_bins), binned_scale_values, label='Scale Values', color='tab:orange', s=10)
        ax2.set_xlabel('Bins (1,000 bp each)')
        ax2.set_ylabel('Scale Values', color='tab:orange')
        ax2.tick_params(axis='y', labelcolor='tab:orange')
        ax2.set_ylim(0, 1)
        ax2.legend(loc='upper right')

        fig.suptitle(f'Chromosome {chrom_num} - Scale Values and Gaps')
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join(file_path, f'data_vis/chr{chrom_num}_dist.png'))
        plt.close()

    # for index, row in bed.iterrows():
    #     chrom = row["chrom"]
    #     chrom_num = chrom.split("chr")[1]
    #     if chrom_num not in ["18", "19", "20", "21", "22"]:
    #         continue
    #     size = row["end"]
    #     chrom_num = chrom.split("chr")[1]
    #     gaps_file = f"{gaps_file_path}chr{chrom_num}_gap_counts.npz"
    #     gaps = extract_gaps(gaps_file, size)
    #     print("extracted gaps successfully for chrom:", chrom_num)
    #     scale_values = extract_scale_values(scaling_features, chrom_num, size)
    #     scale_values = np.array(scale_values, dtype=np.float64)
    #     bin_size = 1000
    #     num_bins = size // bin_size
    #     binned_scale_values = np.mean(scale_values[:num_bins * bin_size].reshape(-1, bin_size), axis=1)
    #     binned_gaps = np.mean(gaps[:num_bins * bin_size].reshape(-1, bin_size), axis=1)

    #     fig, ax1 = plt.subplots(figsize=(10, 5))
    #     ax1.plot(binned_gaps, label='Gaps', color='tab:blue')
    #     ax1.set_xlabel('Bins (1,000 bp each)')
    #     ax1.set_ylabel('Gaps', color='tab:blue')
    #     ax1.tick_params(axis='y', labelcolor='tab:blue')
    #     ax1.set_xlim(0, num_bins)
    #     ax1.set_ylim(0, 240)

    #     ax2 = ax1.twinx()
    #     ax2.plot(binned_scale_values, label='Scale Values', color='tab:orange')
    #     ax2.set_ylabel('Scale Values', color='tab:orange')
    #     ax2.tick_params(axis='y', labelcolor='tab:orange')
    #     ax2.set_ylim(-3, 3)

    #     fig.suptitle(f'Chromosome {chrom_num} - Scale Values and Gaps')
    #     fig.tight_layout()
    #     plt.savefig(os.path.join(file_path, f'data_vis/chr{chrom_num}_lines_dist.png'))
    #     plt.close()


   

def main():
    # process command line arguments
    parser = argparse.ArgumentParser(
        description="Generate data files for training, testing, and validation sets"
    )
    parser.add_argument(
        "--bed_file",
        type=str,
        default="/home/t-mconsens/gamba/data_processing/data/240-mammalian/hg38.bed",
        help="File name of the bed file",
    )
    parser.add_argument(
        "--file_path",
        type=str,
        default="/home/t-mconsens/gamba/data_processing/data/240-mammalian/",
        help="Directory to save the new sequence and conservation scores fasta",
    )
    parser.add_argument(
        "--genome_fasta",
        type=str,
        default="/home/t-mconsens/gamba/data_processing/data/240-mammalian/hg38.ml.fa",
        help="Path to the genome fasta file",
    )
    parser.add_argument(
        "--scaling_features",
        type=str,
        default="/home/t-mconsens/gamba/data_processing/data/240-mammalian/scaling_features/",
        help="Path to the folder with the scaling features",
    )
    parser.add_argument(
        "--splits_file",
        type=str,
        default="/home/t-mconsens/gamba/data_processing/data/240-mammalian/splits.json",
        help="Path to the splits JSON file",
    )
    parser.add_argument(
        "--gaps_file_path",
        type=str,
        default="/home/t-mconsens/gamba/data_processing/data/240-mammalian/gaps/",
        help="Path to the folder with the gap counts",
    )

    args = parser.parse_args()

    # load the BED file to pandas df
    bed = pd.read_csv(
        args.bed_file, sep="\t", header=None, names=["chrom", "start", "end"]
    )

    plot_dist(
        bed, args.file_path, args.splits_file, args.scaling_features, args.gaps_file_path,
    )


if __name__ == "__main__":
    main()
