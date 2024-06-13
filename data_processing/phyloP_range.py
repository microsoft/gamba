import pyBigWig
import argparse


def get_range(bigwig_file):
    with pyBigWig.open(bigwig_file) as bw:
        min_val = float("inf")
        max_val = float("-inf")
        for chrom in bw.chroms():
            for start, end, value in bw.intervals(chrom):
                min_val = min(min_val, value)
                max_val = max(max_val, value)
        return min_val, max_val


def main():
    # process command line arguments
    parser = argparse.ArgumentParser(
        description="Max and min ranges for bigwig phyloP scores"
    )
    parser.add_argument(
        "--bigwig_file",
        type=str,
        default="/home/t-mconsens/gamba/data_processing/data/241-mammalian-2020v2.bigWig",
        help="Path to the bigwig file with phyloP scores",
    )
    parser.add_argument(
        "--file_path",
        type=str,
        default="/home/t-mconsens/gamba/data_processing/data/data_vis/",
        help="Directory to save the txt file with min and max values",
    )
    args = parser.parse_args()
    min_val, max_val = get_range(args.bigwig_file)
    print(f"Min value: {min_val}, Max value: {max_val}")
    print(f"Range: {max_val - min_val}")
    # save the min and max value to a file
    with open(f"{args.file_path}min_max_range.txt", "w") as f:
        f.write(f"Min value: {min_val}, Max value: {max_val}\n")
        f.write(f"Range: {max_val - min_val}\n")


if __name__ == "__main__":
    main()
