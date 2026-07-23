import os
import numpy as np
import argparse
import json


def uncompress_and_save(chromosomes, splits_file, data_dir, task=None):
    with open(splits_file, "r") as f:
        splits = json.load(f)

    # create a dictionary to map chromosomes to splits
    chromosome_splits = {}
    for split, chroms in splits.items():
        for chrom in chroms:
            chromosome_splits[chrom] = split

    if task:
        filename_addition="_small"
    else:
        filename_addition=""

    for chromosome in chromosomes:
        split = chromosome_splits[chromosome]
        print(f"Uncompressing {chromosome} and saving as .npy")
        data = np.load(os.path.join(data_dir, f"{split}/{chromosome}.npz"))
        seq_data = data["sequence"]
        cons_data = data["conservation"]
        np.save(os.path.join(data_dir, f"{split}/{chromosome}_sequence{filename_addition}.npy"), seq_data)
        np.save(os.path.join(data_dir, f"{split}/{chromosome}_conservation{filename_addition}.npy"), cons_data)

    

def main():
    parser = argparse.ArgumentParser(description="Uncompress .npz files and save as .npy")
    parser.add_argument(
        "--file_path",
        type=str,
        required=True,
        default="data_processing/data/240-mammalian/",
        help="Directory to find files to uncompress",
    )
    parser.add_argument(
        "--splits_file",
        type=str,
        required=True,
        default="data_processing/data/240-mammalian/splits.json",
        help="Path to the splits JSON file",
    )
    parser.add_argument(
        "--type",
        type=str,
        default=None,
        help="Task type small dataset or full dataset",
    )
    args = parser.parse_args()

    #full list of chromosomes 1-22 + X
    chromosomes = [str(i) for i in range(1, 23)] + ["X"]
    uncompress_and_save(chromosomes, args.splits_file, args.file_path, task=args.type)

if __name__ == "__main__":
    main()
