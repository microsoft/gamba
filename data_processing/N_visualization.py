import pyBigWig
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import logging
from pyfaidx import Fasta

_logger = logging.getLogger(__name__)


def plot_Ns(fasta_file: str, bed: pd.DataFrame, file_path: str, verbose: bool = True):
    # open the fasta file
    genome = Fasta(fasta_file)

    # iterate over the BED file
    for index, row in bed.iterrows():
        # get the chromosome and size from the BED file
        chrom = row["chrom"]
        size = row["end"]
        chrom_num = chrom.split("chr")[1]

        # if verbose print using logger:
        if verbose:
            _logger.info(f"Processing chromosome: {chrom}")

        # get the number of Ns in in 1000bp bins from 0 to size per chromosome
        bins = int(np.ceil(size / 1000))
        n_nucleotides = []
        for i in range(bins):
            start = i * 1000
            end = min((i + 1) * 1000, size)
            # get the sequence from the genome
            sequence = genome[chrom][start:end].seq
            n_count = sequence.count("N")
            n_nucleotides.append(n_count)

        # plot the distribution of Ns across the whole chromosome for every bin
        plt.scatter(range(bins), n_nucleotides, s=1)
        plt.title(f"Ns for Chromosome {chrom_num}")
        plt.xlim(0, bins)
        plt.xlabel("1,000bp Bin")
        plt.ylabel("Ns")
        # set a y lim
        plt.ylim(0, 4000)
        plt.savefig(f"{file_path}Ns_chr{chrom_num}.png")
        plt.close()

    # if verbose print where saved
    if verbose:
        _logger.info(f"Saved conservation scores to {file_path}")


def main():
    # process command line arguments
    parser = argparse.ArgumentParser(
        description="Plot the number of Ns in sequence every 1,000bp"
    )
    parser.add_argument(
        "--fasta_file",
        type=str,
        default="/home/t-mconsens/gamba/data_processing/data/240-mammalian/hg38.ml.fa",
        help="Path to the human genome fasta file",
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
        default="/home/t-mconsens/gamba/data_processing/data/240-mammalian/data_vis/",
        help="Directory to save the plotted scores",
    )
    args = parser.parse_args()

    # load the BED file to pandas df
    bed = pd.read_csv(
        args.bed_file, sep="\t", header=None, names=["chrom", "start", "end"]
    )

    plot_Ns(args.fasta_file, bed, args.file_path)


if __name__ == "__main__":
    main()
