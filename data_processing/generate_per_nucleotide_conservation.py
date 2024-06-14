import pyBigWig
import pandas as pd
import os
import argparse
import numpy as np
from pyfaidx import Fasta
import json


def make_datasets(bigwig_file, bed, file_path, genome_fasta, splits_file):
    # open the bigwig file
    bw = pyBigWig.open(bigwig_file)

    # open the genome fasta file
    genome = Fasta(genome_fasta)

    # create directories for train, test, and valid if they don't exist
    os.makedirs(f"{file_path}train", exist_ok=True)
    os.makedirs(f"{file_path}test", exist_ok=True)
    os.makedirs(f"{file_path}valid", exist_ok=True)

    # iterate over the BED file
    for index, row in bed.iterrows():
        # get the chromosome and size from the BED file
        chrom = row["chrom"]
        size = row["end"]
        chrom_num = chrom.split("chr")[1]

        print(f"Processing chromosome: {chrom}, chromosome number: {chrom_num}")

        # get the sequence from the genome
        sequence = genome[chrom][:size].seq

        # convert the characters to int8
        sequence = np.frombuffer(sequence.encode(), dtype=np.int8)

        # get the conservation scores from the bigwig file
        intervals = bw.intervals(chrom, 0, size)

        # if intervals is not None, get the scores as a numpy array, numpy float64
        if intervals is not None:
            vals = np.array([interval[2] for interval in intervals])
        # use the splits json to save the numpy array as a compressed numpy file by chrom_num
        # read in the splits file
        with open(splits_file, "r") as f:
            splits = json.load(f)
        # iterate over the splits
        for split, chroms in splits.items():
            if chrom_num in chroms:
                print(f"Saving {split} data for chromosome: {chrom_num}")
                split_dir = f"{file_path}{split}/"
                seq_npz_file = f"{split_dir}seq_{chrom_num}.npz"
                score_npz_file = f"{split_dir}score_{chrom_num}.npz"
                os.makedirs(split_dir, exist_ok=True)
                np.savez_compressed(seq_npz_file, data=sequence)
                np.savez_compressed(score_npz_file, data=vals)

    # close the bigwig file
    bw.close()


def main():
    # process command line arguments
    parser = argparse.ArgumentParser(
        description="Generate data files for training, testing, and validation sets"
    )
    parser.add_argument(
        "--bigwig_file",
        type=str,
        default="/home/t-mconsens/gamba/data_processing/data/241-mammalian-2020v2.bigWig",
        help="Path to the bigwig file with phyloP scores",
    )
    parser.add_argument(
        "--bed_file",
        type=str,
        default="/home/t-mconsens/gamba/data_processing/data/hg38.bed",
        help="File name of the bed file",
    )
    parser.add_argument(
        "--file_path",
        type=str,
        default="/home/t-mconsens/gamba/data_processing/data/",
        help="Directory to save the new sequence and conservation scores fasta",
    )
    parser.add_argument(
        "--genome_fasta",
        type=str,
        default="/home/t-mconsens/gamba/data_processing/data/hg38.ml.fa",
        help="Path to the genome fasta file",
    )
    parser.add_argument(
        "--splits_file",
        type=str,
        default="/home/t-mconsens/gamba/data_processing/data/splits.json",
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
    print(f"Sequences and conservation scores fasta files created in: {args.file_path}")


if __name__ == "__main__":
    main()
