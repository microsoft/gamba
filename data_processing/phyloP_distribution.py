import pyBigWig
import argparse
import pyBigWig
import argparse
import numpy as np

import matplotlib.pyplot as plt
import pandas as pd
import os


def plot_dist(bigwig_file: str, bed: pd.DataFrame, file_path: str):
    # open the bigwig file
    bw = pyBigWig.open(bigwig_file)

    # set the x and y axis limits
    x_min = -20
    x_max = 9
    y_min = 0.0001
    y_max = 10**7.5

    # iterate over the BED file
    for index, row in bed.iterrows():
        # get the chromosome and size from the BED file
        chrom = row["chrom"]
        size = row["end"]
        chrom_num = chrom.split("chr")[1]

        print(f"Processing chromosome: {chrom}, chromosome number: {chrom_num}")

        # get the conservation scores from the bigwig file
        intervals = bw.intervals(chrom, 0, size)

        # if intervals is not None, get the scores as a numpy array, numpy float64
        if intervals is not None:
            vals = np.array([interval[2] for interval in intervals])
            print(f"Vals:, {min(vals)}, {max(vals)}")

            # plot the distribution of scores
            plt.hist(vals, bins=1000)
            plt.title(f"Distribution of Conservation Scores for Chromosome {chrom_num}")
            plt.xlabel("Conservation Score")
            plt.ylabel("Frequency")
            # log scale y
            plt.yscale("log")
            plt.xlim(x_min, x_max)
            plt.ylim(y_min, y_max)
            plt.savefig(f"{file_path}distribution_chr{chrom_num}.png")
            plt.close()

    # close the bigwig file
    bw.close()


def main():
    # process command line arguments
    parser = argparse.ArgumentParser(
        description="Plot the distribution of conservation scores in a bigwig file"
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
        default="/home/t-mconsens/gamba/data_processing/data/data_vis/",
        help="Directory to save the plotted distributions",
    )
    parser.add_argument(
        "--genome_fasta",
        type=str,
        default="/home/t-mconsens/gamba/data_processing/data/hg38.ml.fa",
        help="Path to the genome fasta file",
    )
    args = parser.parse_args()

    # load the BED file to pandas df
    bed = pd.read_csv(
        args.bed_file, sep="\t", header=None, names=["chrom", "start", "end"]
    )

    plot_dist(args.bigwig_file, bed, args.file_path)
    print(f"Plots created in: {args.file_path}")


if __name__ == "__main__":
    main()
