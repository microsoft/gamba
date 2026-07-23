import numpy as np
import os
import argparse
import json

def calculate_chromosome_sizes(input_file, output_file, data_dir, splits_file, task, chrom_lengths):
    chrom_sizes = {}
    with open(splits_file, "r") as f:
        splits = json.load(f)

    #read chrom_lengths from bed file
    with open(chrom_lengths, 'r') as infile:
        chrom_lengths = {}
        for line in infile:
            parts = line.strip().split('\t')
            chrom = parts[0]
            length = int(parts[1])
            chrom_lengths[chrom] = length

    # create a dictionary to map chromosomes to splits
    chromosome_splits = {}
    for split, chroms in splits.items():
        for chrom in chroms:
            chromosome_splits[chrom] = split

    if task:
        filename_addition="_small"
    else:
        filename_addition=""


    chromosomes = ["chr"+str(i) for i in range(1, 23)] + ["chrX"]
    with open(input_file, 'r') as infile:
        excluded_regions = {}
        for line in infile:
            parts = line.strip().split('\t')
            chrom = parts[0]
            start = int(parts[1])
            end = int(parts[2])

            # only process specified chromosomes
            if chrom in chromosomes:
                if chrom not in excluded_regions:
                    excluded_regions[chrom] = []
                excluded_regions[chrom].append((start, end))

    
    # process each chromosome
    for chrom in chromosomes:
        included_size = 0
        prev_end = 0

        # Get the excluded regions for this chromosome
        regions = excluded_regions.get(chrom, [])
        if regions:
            # Sort regions by start position
            regions.sort()

            # Calculate included regions
            for start, end in regions:
                # Add the region between the previous excluded region and the current one
                if prev_end < start:
                    included_size += start - prev_end
                prev_end = end

        # Add the final included region after the last excluded region
        chrom_length = chrom_lengths.get(chrom, 0)
        if prev_end < chrom_length:
            included_size += chrom_length - prev_end

        chrom_sizes[chrom] = included_size

    print("made it past the first loop")
    print(chrom_sizes)

    # write the chromosome sizes to the output file
    with open(output_file, 'w') as outfile:
        for chrom, size in chrom_sizes.items():
            outfile.write(f"{chrom}\t{size}\n")
            print("wrote to outfile")

            # check if the .npy files for sequence and sizes exist
            #find split in chromosome_splits
            #remove chr from chrom
            chromosome = chrom.replace("chr", "")
            split = chromosome_splits[chromosome] 
            sequence_file = os.path.join(data_dir, f"{split}/{chromosome}_sequence{filename_addition}.npy")
            conservation_file = os.path.join(data_dir, f"{split}/{chromosome}_conservation{filename_addition}.npy")

            if os.path.exists(sequence_file) and os.path.exists(conservation_file):
                sequence_array = np.load(sequence_file)
                conservation_array = np.load(conservation_file)

                print("Size of sequence array:", len(sequence_array))
                print("Size of conservation array:", len(conservation_array))
                print("Size of chrom_sizes:", size)

                # assert that the size matches the calculated chrom size from the .bed file
                assert len(sequence_array) == size, f"Size mismatch in {sequence_file}!"
                assert len(conservation_array) == size, f"Size mismatch in {conservation_file}!"

                # If the assertion passes, print success message
                print(f"Success: {chrom}_sequence.npy and {chrom}_conservation.npy match the .bed file sizes.")
            else:
                print(f"Warning: {sequence_file} or {conservation_file} does not exist.")


def main():
    # process command line arguments
    parser = argparse.ArgumentParser(
        description="Check chromosome sizes in .npy files"
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default="/data_processing/data/240-mammalian/regions_excluded.bed",
        help="File name of the exclusion regions file",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="/data_processing/data/240-mammalian/cleaned_chrom_sizes.txt",
        help="File name to save chrom_sizes.txt file",
    )
    parser.add_argument(
        "--splits_json",
        type=str,
        default="/data_processing/data/240-mammalian/splits.json",
        help="The train/test/validation splits",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/data_processing/data/240-mammalian/",
        help="Directory to find files to uncompress",
    )
    parser.add_argument(
        "--type",
        type=str,
        default=None,
        help="Task type small dataset or full dataset",
    )
    parser.add_argument(
        "--chrom_size_original",
        type=str,
        default="/data_processing/data/240-mammalian/hg38.chrom.sizes",
        help="Original chromosome sizes file",
    )
    args = parser.parse_args()

    calculate_chromosome_sizes(args.input_file, args.output_file, args.data_dir, args.splits_json, args.type, args.chrom_size_original)


if __name__ == "__main__":
    main()
