import argparse
import pandas as pd
from pyfaidx import Fasta
from pathlib import Path

def extract_sequences(genome_fasta: str, bed_file: str, chromosome: str, output_fasta: str):
    fasta = Fasta(genome_fasta)
    bed = pd.read_csv(bed_file, sep="\t", header=None, names=["chrom", "start", "end"])
    bed_chr = bed[bed["chrom"] == chromosome]

    with open(output_fasta, "w") as out_f:
        for i, row in bed_chr.iterrows():
            chrom, start, end = row["chrom"], int(row["start"]), int(row["end"])
            seq = fasta[chrom][start:end].seq.upper()
            if len(seq) == 0:
                continue  # skip empty 
            out_f.write(f">region_{i}_{chrom}:{start}-{end}\n{seq}\n")

    print(f"Saved {len(bed_chr)} sequences to {output_fasta}")


def main():
    # process command line arguments
    parser = argparse.ArgumentParser(
        description="Generate FASTA data files for training, testing, and validation sets"
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
        default="/home/mica/gamba/data_processing/data/240-mammalian/cleaned_fasta",
        help="Path to the splits JSON file",
    )
    parser.add_argument(
        "--chromosome",
        type=str,
        default="chr2",
        help="Chromosome to analyze",
    )
    args = parser.parse_args()

    # load the BED file to pandas df
    bed = pd.read_csv(
        args.bed_file, sep="\t", header=None, names=["chrom", "start", "end"]
    )

    print("chromosome is:", args.chromosome)
    #add chromosome name to output fasta
    output_fasta = f'{args.output_fasta}_{args.chromosome}.fa'


    extract_sequences(args.genome_fasta, args.bed_file, args.chromosome, output_fasta)


if __name__ == "__main__":
    main()
