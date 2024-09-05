import pyBigWig
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
# import gamba using sys.append
import sys

sys.path.append("../gamba")


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

    # dictionary to store concatenated sequences and scores
    chrom_data = {}

    # dictionary to store chromosome sizes
    chrom_sizes = {}

    # iterate over the BED file
    for index, row in bed.iterrows():
        # get the chromosome and size from the BED file
        chrom = row["chrom"]
        start = row["start"]
        end = row["end"]
        chrom_num = chrom.split("chr")[1]
        print("the chromosome is:", chrom)
        print("the start is:", start)
        print("the end is:", end)

        # get the sequence from the genome
        sequence = genome[chrom][start:end].seq
        # print first 10 characters of the sequence
        print("the first 10 characters of the sequence are:", sequence[:10])

        # tokenize the sequence
        tokenizer = Tokenizer(DNA_ALPHABET_PLUS)
        sequence = tokenizer.tokenizeMSA(sequence)
        print("the first 10 TOKENIZED chars of the sequence are:", sequence[:10])

        # initialize vals with zeros
        vals = np.zeros(end - start, dtype=np.float64)

        # get the conservation scores from the bigwig file
        intervals = bw.intervals(chrom, start, end)

        # Check if intervals is None
        if intervals is None:
            print("Error: intervals is None")
        else:
            for interval_start, interval_end, value in intervals:
                vals[interval_start - start : interval_end - start] = value

        # print first 10 conservation scores
        print("the first 10 conservation scores are:", vals[:10])
        #round vals to 2 decimal places
        vals = np.round(vals, 2)

        # concatenate sequences and scores
        if chrom not in chrom_data:
            chrom_data[chrom] = {"sequence": sequence, "conservation": vals}
            chrom_sizes[chrom] = len(sequence)
        else:
            chrom_data[chrom]["sequence"] = np.concatenate(
                (chrom_data[chrom]["sequence"], sequence)
            )
            chrom_data[chrom]["conservation"] = np.concatenate(
                (chrom_data[chrom]["conservation"], vals)
            )
            chrom_sizes[chrom] += len(sequence)

    # save concatenated sequences and scores per chromosome
    for chrom, data in chrom_data.items():
        chrom_num = chrom.split("chr")[1]
        split_name = chromosome_splits[chrom_num]
        if verbose:
            _logger.info(f"Saving {split_name} data for chromosome: {chrom_num}")
        split_dir = f"{file_path}{split_name}/"
        seq_cons_file = f"{split_dir}{chrom_num}.npz"
        os.makedirs(split_dir, exist_ok=True)
        np.savez_compressed(
            seq_cons_file, sequence=data["sequence"], conservation=data["conservation"]
        )

    # save chromosome sizes to a new file
    chrom_sizes_file = os.path.join(file_path, "chrom_sizes.txt")
    with open(chrom_sizes_file, "w") as f:
        for chrom, size in chrom_sizes.items():
            f.write(f"{chrom}\t{size}\n")

    # close the bigwig file
    bw.close()
    if verbose:
        _logger.info(f"Processing completed.")


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
        default="/home/t-mconsens/gamba/data_processing/data/240-mammalian/regions.bed",
        help="File name of the bed file excluding low quality regions",
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