import argparse
import pandas as pd
import numpy as np
import pyBigWig
from pyfaidx import Fasta
from evodiff.utils import Tokenizer
import sys
from pathlib import Path

sys.path.append("../gamba")
from gamba.constants import DNA_ALPHABET_PLUS


def get_excluded_regions(bed_df, chrom, chrom_length):
    """Return list of (start, end) regions NOT covered by the BED regions."""
    bed_chr = bed_df[bed_df["chrom"] == chrom].sort_values("start")
    excluded = []
    prev_end = 0

    for _, row in bed_chr.iterrows():
        start, end = int(row["start"]), int(row["end"])
        if start > prev_end:
            excluded.append((prev_end, start))
        prev_end = max(prev_end, end)

    if prev_end < chrom_length:
        excluded.append((prev_end, chrom_length))

    return excluded


def process_excluded_regions(chromosome, bed_df, genome_fasta, bigwig_file, output_prefix):
    genome = Fasta(genome_fasta)
    bw = pyBigWig.open(bigwig_file)
    tokenizer = Tokenizer(DNA_ALPHABET_PLUS)

    chrom_length = len(genome[chromosome])
    excluded_regions = get_excluded_regions(bed_df, chromosome, chrom_length)

    all_tokens = []
    all_scores = []

    for i, (start, end) in enumerate(excluded_regions):
        seq = genome[chromosome][start:end].seq.upper()
        if not seq:
            continue

        tokens = tokenizer.tokenizeMSA(seq)
        vals = np.zeros(end - start, dtype=np.float32)

        intervals = bw.intervals(chromosome, start, end)
        if intervals:
            for int_start, int_end, val in intervals:
                vals[int_start - start:int_end - start] = val
        vals = np.round(vals, 2)
        all_tokens.append(tokens)
        all_scores.append(vals)

    # Concatenate and save
    np.save(f"{output_prefix}_{chromosome}_sequence.npy", np.concatenate(all_tokens))
    np.save(f"{output_prefix}_{chromosome}_score.npy", np.concatenate(all_scores))
    print(f"Saved {len(all_tokens)} excluded regions.")


def main():
    parser = argparse.ArgumentParser(description="Tokenize and score excluded genome regions.")
    parser.add_argument(
        "--bigwig_file",
        type=str,
        default="/home/mica/gamba/data_processing/data/240-mammalian/241-mammalian-2020v2.bigWig",
        help="Path to the bigwig file with phyloP scores",
    )
    parser.add_argument(
        "--bed_file",
        type=str,
        default="/home/mica/gamba/data_processing/data/240-mammalian/regions.bed",
        help="File name of the bed file excluding low quality regions",
    )
    parser.add_argument(
        "--genome_fasta",
        type=str,
        default="/home/mica/gamba/data_processing/data/240-mammalian/hg38.ml.fa",
        help="Path to the genome fasta file",
    )
    parser.add_argument(
        "--output_prefix",
        type=str,
        default="/home/mica/gamba/data_processing/data/240-mammalian/opposite_data",
        help="name of the output files (without extension)",
    )
    parser.add_argument(
        "--chromosome",
        type=str,
        default="chr2",
        help="Chromosome to analyze",
    )
    args = parser.parse_args()

    args = parser.parse_args()

    bed_df = pd.read_csv(args.bed_file, sep="\t", header=None, names=["chrom", "start", "end"])
    process_excluded_regions(
        args.chromosome, bed_df, args.genome_fasta, args.bigwig_file, args.output_prefix
    )


if __name__ == "__main__":
    main()
