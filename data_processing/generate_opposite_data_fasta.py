import argparse
import pandas as pd
from pyfaidx import Fasta
from pathlib import Path

def extract_excluded_sequences(genome_fasta: str, bed_file: str, chromosome: str, output_fasta: str):
    fasta = Fasta(genome_fasta)
    bed = pd.read_csv(bed_file, sep="\t", header=None, names=["chrom", "start", "end"])
    bed_chr = bed[bed["chrom"] == chromosome].sort_values(by="start")

    chr_length = len(fasta[chromosome])
    excluded_regions = []

    # Find gaps between included regions
    prev_end = 0
    for _, row in bed_chr.iterrows():
        start = int(row["start"])
        if start > prev_end:
            excluded_regions.append((prev_end, start))
        prev_end = max(prev_end, int(row["end"]))

    # Add the tail end if needed
    if prev_end < chr_length:
        excluded_regions.append((prev_end, chr_length))

    # Write excluded regions to FASTA
    with open(output_fasta, "w") as out_f:
        for i, (start, end) in enumerate(excluded_regions):
            seq = fasta[chromosome][start:end].seq.upper()
            if len(seq) == 0:
                continue
            out_f.write(f">excluded_{i}_{chromosome}:{start}-{end}\n{seq}\n")

    print(f"Saved {len(excluded_regions)} excluded regions to {output_fasta}")


def main():
    # process command line arguments
    parser = argparse.ArgumentParser(
        description="Generate FASTA data files for excluded data"
    )
    parser.add_argument(
        "--bed_file",
        type=str,
        default="/home/mica/gamba/data_processing/data/240-mammalian/regions.bed",
        help="File name of the bed file excluding low quality regions",
    )
    parser.add_argument(
        "--file_path",
        type=str,
        default="/home/mica/gamba/data_processing/data/240-mammalian/",
        help="Directory to save the new sequence and conservation scores fasta",
    )
    parser.add_argument(
        "--genome_fasta",
        type=str,
        default="/home/mica/gamba/data_processing/data/240-mammalian/hg38.ml.fa",
        help="Path to the genome fasta file",
    )
    parser.add_argument(
        "--output_fasta",
        type=str,
        default="/home/mica/gamba/data_processing/data/240-mammalian/opposite_fasta",
        help="Path to the splits JSON file",
    )
    parser.add_argument(
        "--chromosome",
        type=str,
        default="chr2",
        help="Chromosome to analyze",
    )
    args = parser.parse_args()

    output_fasta = f'{args.output_fasta}_{args.chromosome}.fa'
    extract_excluded_sequences(args.genome_fasta, args.bed_file, args.chromosome, output_fasta)

if __name__ == "__main__":
    main()
