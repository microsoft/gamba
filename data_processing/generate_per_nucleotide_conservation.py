import pandas as pd
import os
import argparse
import numpy as np
from pyfaidx import Fasta
import json
import logging
from evodiff.utils import Tokenizer
from gamba.constants import DNA_ALPHABET_PLUS

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

def make_datasets(
    bed: pd.DataFrame,
    file_path: str,
    genome_fasta: str,
    splits_file: str,
    scaling_features: str,
    gaps_file_path: str,
    verbose: bool = True,
):
    # open the genome fasta file
    genome = Fasta(genome_fasta)

    # create directories for train, test, and valid if they don't exist
    os.makedirs(f"{file_path}train", exist_ok=True)
    os.makedirs(f"{file_path}test", exist_ok=True)
    os.makedirs(f"{file_path}valid", exist_ok=True)

    # use the splits json to save the numpy array as a compressed numpy file by chrom_num
    # read in the splits file
    with open(splits_file, "r") as f:
        splits = json.load(f)

    # create a dictionary to map chromosomes to splits
    chromosome_splits = {}
    for split, chroms in splits.items():
        for chrom in chroms:
            chromosome_splits[chrom] = split

    # iterate over the BED file
    for index, row in bed.iterrows():

        # get the chromosome and size from the BED file
        chrom = row["chrom"]
        chrom_num = chrom.split("chr")[1]
        if chrom_num not in ["18", "19", "20", "21", "22"]:
            continue
        size = row["end"]
        chrom_num = chrom.split("chr")[1]

        #get the right gaps file
        gaps_file = f"{gaps_file_path}chr{chrom_num}_gap_counts.npz"

        #read in the gaps
        gaps = extract_gaps(gaps_file, size)
        print("extracted gaps successfully for chrom:", chrom_num)

        # extract scaling param per nucleotide
        scale_values = extract_scale_values(scaling_features, chrom_num, size)

        #turn scale values into numpy array
        scale_values = np.array(scale_values, dtype=np.float64)

        # get the sequence from the genome
        sequence = genome[chrom][:size].seq

        # tokenize the sequence already
        tokenizer = Tokenizer(DNA_ALPHABET_PLUS)
        sequence = tokenizer.tokenize(sequence)

        # get the split for the current chromosome
        split_name = chromosome_splits[chrom_num]
        if verbose:
            _logger.info(f"Saving {split_name} data for chromosome: {chrom_num}")
        split_dir = f"{file_path}{split_name}/"
        seq_cons_file = f"{split_dir}{chrom_num}_withgapsnscaling.npz"
        os.makedirs(split_dir, exist_ok=True)
        np.savez_compressed(seq_cons_file, sequence=sequence, conservation=scale_values, gaps=gaps)

    if verbose:
        _logger.info(f"Processing chromosome: {chrom}")


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

    make_datasets(
        bed, args.file_path, args.genome_fasta, args.splits_file, args.scaling_features, args.gaps_file_path,
    )


if __name__ == "__main__":
    main()
