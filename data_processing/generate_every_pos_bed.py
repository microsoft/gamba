import os
import argparse
import logging

_logger = logging.getLogger(__name__)


def make_bed_per_chrom(chrom_sizes: str, file_path: str, test: bool = False, verbose: bool = True):
    # open chrom sizes
    test=True
    with open(chrom_sizes, "r") as chrom_sizes_file:
        for line in chrom_sizes_file:
            # subset to chromosomes we want
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
                size = int(size)
                
                if test and chrom == "chr1":
                    start = (size // 4)*3 - 500
                    end = start + 100
                    bed_lines = [f"{chrom}\t{i}\t{i+1}" for i in range(start, end)]
                    bed_file_path = f"{file_path}{chrom}_everypos_hg38_test.bed"
                else:
                    bed_lines = [f"{chrom}\t{i}\t{i+1}" for i in range(size)]
                    bed_file_path = f"{file_path}{chrom}_everypos_hg38.bed"

                if verbose:
                    print("Processing Chromosome: ", chrom, "Size: ", size)
                
                with open(bed_file_path, "w") as bed_file:
                    bed_file.write("\n".join(bed_lines))

                if verbose:
                    _logger.info(f"Saved BED file for {chrom} to {bed_file_path}")
                if test:
                    break


def main():
    # process command line arguments
    parser = argparse.ArgumentParser(
        description="Generate a BED file from the hg38.chrom.sizes file"
    )
    parser.add_argument(
        "--chrom_sizes",
        type=str,
        default="/home/t-mconsens/gamba/data_processing/data/240-mammalian/hg38.chrom.sizes",
        help="Path to the hg38.chrom.sizes file",
    )
    parser.add_argument(
        "--file_path",
        type=str,
        default="/home/t-mconsens/gamba/data_processing/data/240-mammalian/all_chrom_beds/",
        help="Directory to save the BED file",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run in test mode, subsetting to 1000 positions in the middle of chromosome 1",
    )
    args = parser.parse_args()

    os.makedirs(args.file_path, exist_ok=True)

    make_bed_per_chrom(args.chrom_sizes, args.file_path, test=args.test)


if __name__ == "__main__":
    main()