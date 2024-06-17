import pyBigWig
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def plot_scores(bigwig_file: str, bed: pd.DataFrame, file_path: str):
    # open the bigwig file
    bw = pyBigWig.open(bigwig_file)

    # iterate over the BED file
    for index, row in bed.iterrows():
        # get the chromosome and size from the BED file
        chrom = row["chrom"]
        size = row["end"]
        chrom_num = chrom.split("chr")[1]

        print(f"Processing chromosome: {chrom}, chromosome number: {chrom_num}")
        # get the conservation scores from the bigwig file in 1000bp bins from 0 to size
        bins = int(np.ceil(size / 1000))
        scores = []
        for i in range(bins):
            start = i * 1000
            end = min((i + 1) * 1000, size)
            intervals = bw.intervals(chrom, start, end)
            if intervals is not None:
                vals = np.array([interval[2] for interval in intervals])
                scores.append(np.mean(vals))
            else:
                scores.append(0)
        # use these average scores to plot the conservation across the whole chromosome for every bin
        plt.scatter(range(bins), scores, s=1)
        plt.title(f"Conservation Scores for Chromosome {chrom_num}")
        plt.xlabel("1000bp Bin")
        plt.ylabel("Conservation Score")
        plt.savefig(f"{file_path}conservation_chr{chrom_num}.png")
        plt.close()

    # close the bigwig file
    bw.close()


def main():
    # process command line arguments
    parser = argparse.ArgumentParser(
        description="Plot the conservation scores in a bigwig file"
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
        help="Directory to save the plotted scores",
    )
    args = parser.parse_args()

    # load the BED file to pandas df
    bed = pd.read_csv(
        args.bed_file, sep="\t", header=None, names=["chrom", "start", "end"]
    )

    plot_scores(args.bigwig_file, bed, args.file_path)
    print(f"Plots created in: {args.file_path}")


if __name__ == "__main__":
    main()
