import os
import argparse


# generates a BED file from the hg38.chrom.sizes file
def make_bed(chrom_sizes: str, file_path: str):

    # open the hg38.chrom.sizes file
    chrom_sizes = open(chrom_sizes, "r")

    # convert to BED format
    bed_lines = []
    for line in chrom_sizes:
        # if line follows chr(number) 238956422' format (ignore all chrnumber_/random/alt/scaffold lines):
        if (
            line.startswith("chr")
            and "_" not in line
            and "random" not in line
            and "chrM" not in line
            and "chrY" not in line
            and "alt" not in line
            and "scaffold" not in line
        ):
            chrom, size = line.strip().split("\t")
            bed_lines.append("\t".join([chrom, "0", size]))

    # write to a BED file
    with open((f"{file_path}hg38.bed"), "w") as bed_file:
        bed_file.write("\n".join(bed_lines))


def main():
    # process command line arguments
    parser = argparse.ArgumentParser(
        description="Generate a BED file from the hg38.chrom.sizes file"
    )
    parser.add_argument(
        "--chrom_sizes",
        type=str,
        default="/home/t-mconsens/gamba/data_processing/data/hg38.chrom.sizes",
        help="Path to the hg38.chrom.sizes file",
    )
    parser.add_argument(
        "--file_path",
        type=str,
        default="/home/t-mconsens/gamba/data_processing/data/",
        help="Directory to save the BED file",
    )
    args = parser.parse_args()

    make_bed(args.chrom_sizes, args.file_path)

    print(f"BED file created:{args.file_path}hg38.bed")


if __name__ == "__main__":
    main()
