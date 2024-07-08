import numpy as np
import os
import argparse
import pandas as pd
import os.path as osp

import os
import numpy as np
import pandas as pd


def count_gaps_in_maf(human_bed, maf_file, output_dir):
    """Count number of gaps in non-human species at every position in the human genome in a MAF alignment file and save as compressed .npz files by chromosome

    :param human_bed: str, path to BED file for human genome
    :param maf_file: str, path to MAF file
    :param output_dir: str, path to directory to save output .npz files
    """
    # read human bed and map chrom sizes
    bed = pd.read_csv(human_bed, sep="\t", header=None, names=["chrom", "start", "end"])
    chromosomes = [str(i) for i in range(1, 23)] + ["X"]
    chrom_sizes = {
        chrom: bed[bed["chrom"] == ("chr" + chrom)]["end"].values
        for chrom in chromosomes
    }

    # dictionary to store counts
    gap_counts = {}

    # open MAF file
    with open(maf_file, "r") as maf:
        for line in maf:
            if line.startswith("a"):  # new alignment block
                human_seq = None
                non_human_seqs = []

            elif line.startswith("s"):  # sequence
                parts = line.split()
                species, chrom, start, length = (
                    parts[1],
                    parts[1].split(".")[0],
                    int(parts[2]),
                    int(parts[3]),
                )
                seq = parts[6]

                if "Homo_sapiens" in species:
                    human_seq = (chrom, start, length, seq)
                else:
                    non_human_seqs.append(seq)

            elif line.startswith("\n") and human_seq:  # end of alignment
                chrom, start, _, _ = human_seq
                if chrom not in gap_counts:
                    max_chromosome_length = chrom_sizes[chrom][0]
                    gap_counts[chrom] = np.zeros(max_chromosome_length, dtype=int)

                for seq in non_human_seqs:
                    gaps = np.array([1 if base == "-" else 0 for base in seq])
                    required_length = start + len(seq)
                    if required_length > gap_counts[chrom].size:
                        gap_counts[chrom] = np.pad(
                            gap_counts[chrom],
                            (0, required_length - gap_counts[chrom].size),
                        )
                    gap_counts[chrom][start : start + len(seq)] += gaps

    # save
    for chrom, counts in gap_counts.items():
        np.savez_compressed(
            os.path.join(output_dir, f"{chrom}_gap_counts.npz"), counts=counts
        )

    return gap_counts


def main():
    # process command line arguments
    parser = argparse.ArgumentParser(
        description="Generate a BED file from the hg38.chrom.sizes file"
    )
    parser.add_argument(
        "--human_bed",
        type=str,
        default="/home/t-mconsens/gamba/data_processing/data/240-mammalian/hg38.chrom.sizes",
        help="Path to the human bed file with chrom sizes",
    )
    parser.add_argument(
        "--maf_file",
        type=str,
        default="",
        help="Directory to the MAF file",
    )
    parser.add_argument(
        "--file_path",
        type=str,
        default="/home/t-mconsens/gamba/data_processing/data/240-mammalian/",
        help="Directory to save the BED file",
    )
    args = parser.parse_args()

    count_gaps_in_maf(args.human_bed, args.maf_file, args.file_path)


if __name__ == "__main__":
    main()
