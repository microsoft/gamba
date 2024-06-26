import pyBigWig
import pandas as pd
import os
import argparse
import numpy as np
from pyfaidx import Fasta
import json
import logging

_logger = logging.getLogger(__name__)


def make_datasets(
    bigwig_file: str,
    bed: pd.DataFrame,
    file_path: str,
    genome_fasta: str,
    splits_file: str,
    verbose: bool = True,
):
    # open the bigwig file
    bw = pyBigWig.open(bigwig_file)

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
        size = row["end"]
        chrom_num = chrom.split("chr")[1]

        # if verbose print using logger:
        if verbose:
            _logger.info(
                f"Processing chromosome: {chrom}, chromosome number: {chrom_num}"
            )

        # get the sequence from the genome
        sequence = genome[chrom][:size].seq

        # convert the characters to int8
        # .encode() method comes from built-in python method, results in ASCII
        sequence = np.frombuffer(sequence.encode(), dtype=np.int8)

        # get the conservation scores from the bigwig file
        intervals = bw.intervals(chrom, 0, size)

        # if intervals is not None, get the scores as a numpy array, numpy float64
        if intervals is not None:
            vals = np.array([interval[2] for interval in intervals])

        # get the split for the current chromosome
        split_name = chromosome_splits[chrom_num]
        if verbose:
            _logger.info(f"Saving {split_dir} data for chromosome: {chrom_num}")
        split_dir = f"{file_path}{split_name}/"
        seq_cons_file = f"{split_dir}{chrom_num}.npz"
        os.makedirs(split_dir, exist_ok=True)
        np.savez_compressed(seq_cons_file, sequence=sequence, conservation=vals)

    # close the bigwig file
    bw.close()
    if verbose:
        _logger.info(f"Processing chromosome: {chrom}, chromosome number: {chrom_num}")


def main():
    # process command line arguments
    parser = argparse.ArgumentParser(
        description="Generate data files for training, testing, and validation sets"
    )
    parser.add_argument(
        "--bigwig_file",
        type=str,
        default="/home/t-mconsens/gamba/data_processing/data/240-mammalian/241-mammalian-2020v2.bigWig",
        help="Path to the bigwig file with phyloP scores",
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
        "--splits_file",
        type=str,
        default="/home/t-mconsens/gamba/data_processing/data/240-mammalian/splits.json",
        help="Path to the splits JSON file",
    )
    args = parser.parse_args()

    # load the BED file to pandas df
    bed = pd.read_csv(
        args.bed_file, sep="\t", header=None, names=["chrom", "start", "end"]
    )

    make_datasets(
        args.bigwig_file, bed, args.file_path, args.genome_fasta, args.splits_file
    )


if __name__ == "__main__":
    main()
