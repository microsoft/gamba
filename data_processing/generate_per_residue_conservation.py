import pyBigWig
import pandas as pd
import os
import argparse
from pyfaidx import Fasta


# create two fasta files in the specified directory: sequences.fasta and conservation_scores.fasta
# sequences.fasta file will contain the sequences from the genome corresponding to the regions specified in the BED file
# conservation_scores.fasta file will contain the conservation scores for these regions
# fasta files will have a header in the format >chrom:start-end, followed by the sequence or conservation scores
def make_scores_df(bigwig_file, bed, file_path, genome_fasta):
    # open the bigwig file
    bw = pyBigWig.open(bigwig_file)

    # open the genome fasta file
    genome = Fasta(genome_fasta)

    # empty list to store conservation scores
    conservation_scores = []

    # open the output fasta files
    with open(f"{file_path}sequences.fasta", "w") as seq_fasta, open(
        f"{file_path}conservation_scores.fasta", "w"
    ) as cons_fasta:
        # iterate over the BED file
        for index, row in bed.iterrows():
            # get the conservation scores
            chrom = row["chrom"]
            start = row["start"]
            end = row["end"]

            # get scores for each position in the range
            vals = bw.values(chrom, start, end, numpy=True)

            # check if the returned vals are valid
            if vals is not None:
                # replace nans with 0s
                vals = [0 if pd.isna(val) else val for val in vals]
            else:
                # if the returned vals are invalid, append 0s
                vals = [0] * (end - start)

            conservation_scores.append(vals)

            # get the sequence from the genome
            sequence = genome[chrom][start:end].seq

            # write the sequence to the fasta file
            seq_fasta.write(f">{chrom}:{start}-{end}\n{sequence}\n")

            # write the conservation scores to the fasta file
            cons_fasta.write(f">{chrom}:{start}-{end}\n{''.join(map(str, vals))}\n")

    # close the bigwig file
    bw.close()


def main():
    # process command line arguments
    parser = argparse.ArgumentParser(
        description="Generate.fasta file for sequences and for conservation scores from a bigwig file and a BED file"
    )
    parser.add_argument(
        "--bigwig_file",
        type=str,
        default="/home/t-mconsens/gamba/data_processing/data/241-mammalian-2020v2.bigwig",
        help="Path to the bigwig file with phyloP scores",
    )
    parser.add_argument(
        "--bed_file",
        type=str,
        default="/home/t-mconsens/gamba/data_processing/data/sequences_human.bed",
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
        default="/home/t-mconsens/gamba/data_processing/data/hg38.fa",
        help="Path to the genome fasta file",
    )
    args = parser.parse_args()

    # load the BED file to pandas df
    bed = pd.read_csv(
        args.bed_file, sep="\t", header=None, names=["chrom", "start", "end"]
    )

    make_scores_df(args.bigwig_file, bed, args.file_path, args.genome_fasta)
    print(f"Sequences and conservation scores fasta files created in: {args.file_path}")


if __name__ == "__main__":
    main()
