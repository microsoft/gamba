import argparse
import numpy as np
import pandas as pd
import logging
from pyfaidx import Fasta

_logger = logging.getLogger(__name__)


def read_exclusion_file(exclusion_file):
    exclusion_regions = {}
    with open(exclusion_file, "r") as file:
        for line in file:
            parts = line.strip().split("\t")
            if len(parts) == 5:
                chrom = parts[1]
                chrom_start = int(parts[2])
                chrom_end = int(parts[3])
                if chrom not in exclusion_regions:
                    exclusion_regions[chrom] = []
                exclusion_regions[chrom].append((chrom_start, chrom_end))
    return exclusion_regions


def merge_intervals(intervals):
    print("merging intervals")
    if not intervals:
        return []
    # sort intervals by start
    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]]
    for current in intervals:
        last = merged[-1]
        if current[0] <= last[1]:  # overlapping intervals, merge
            merged[-1] = (last[0], max(last[1], current[1]))
        else:
            merged.append(current)
    return merged


def save_non_excluded_regions(
    fasta_file: str,
    bed: pd.DataFrame,
    exclusion_regions: dict,
    output_file: str,
    verbose: bool = True,
):
    # open the fasta file
    genome = Fasta(fasta_file)
    N_density_regions = {}

    # iterate over the BED file
    for index, row in bed.iterrows():
        # get the chromosome and size from the BED file
        chrom = row["chrom"]
        size = row["end"]
        chrom_num = chrom.split("chr")[1]

        # if verbose print using logger:
        if verbose:
            _logger.info(f"Processing chromosome: {chrom}")
        print(f"Processing chromosome: {chrom}")

        # get the number of Ns in in 1000bp bins from 0 to size per chromosome
        bins = int(np.ceil(size / 1000))
        for i in range(bins):
            start = i * 1000
            end = min((i + 1) * 1000, size)
            # get the sequence from the genome
            sequence = genome[chrom][start:end].seq
            n_count = sequence.count("N")
            if n_count > 0.1 * (end - start):
                if chrom not in N_density_regions:
                    N_density_regions[chrom] = []
                N_density_regions[chrom].append((start, end))

    # merge N_density regions to exclusion regions
    for chrom in N_density_regions:
        if chrom not in exclusion_regions:
            exclusion_regions[chrom] = []
        exclusion_regions[chrom].extend(N_density_regions[chrom])
        exclusion_regions[chrom] = merge_intervals(exclusion_regions[chrom])

    # save excluded regions to file:
    with open(f"{output_file}_excluded.bed", "w") as file:
        for chrom in exclusion_regions:
            for start, end in exclusion_regions[chrom]:
                file.write(f"{chrom}\t{start}\t{end}\n")

    # non-excluded regions
    non_excluded_regions = {}
    for chrom in bed["chrom"].unique():
        size = bed[bed["chrom"] == chrom]["end"].max()
        excluded_intervals = sorted(exclusion_regions.get(chrom, []))
        non_excluded_intervals = []
        prev_end = 0
        for start, end in excluded_intervals:
            if prev_end < start:
                non_excluded_intervals.append((prev_end, start))
            prev_end = end
        if prev_end < size:
            non_excluded_intervals.append((prev_end, size))
        non_excluded_regions[chrom] = non_excluded_intervals

    # save non-excluded regions to a file
    with open(f"{output_file}.bed", "w") as file:
        for chrom in non_excluded_regions:
            for start, end in non_excluded_regions[chrom]:
                file.write(f"{chrom}\t{start}\t{end}\n")

    # if verbose print where saved
    if verbose:
        _logger.info(f"Saved non-excluded regions to {output_file}")


def main():
    # process command line arguments
    parser = argparse.ArgumentParser(
        description="Save the non-excluded regions to a file"
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
        "--exclusion_file",
        type=str,
        default="/home/t-mconsens/gamba/data_processing/data/240-mammalian/centromeres.txt",
        help="File name of the exclusion regions file",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="/home/t-mconsens/gamba/data_processing/data/240-mammalian/regions",
        help="File name to save the non-excluded regions",
    )
    args = parser.parse_args()

    # load the BED file to pandas df
    bed = pd.read_csv(
        args.bed_file, sep="\t", header=None, names=["chrom", "start", "end"]
    )

    # read the exclusion regions (centromeres)
    exclusion_regions = read_exclusion_file(args.exclusion_file)

    save_non_excluded_regions(args.fasta_file, bed, exclusion_regions, args.output_file)


if __name__ == "__main__":
    main()
