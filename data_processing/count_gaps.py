import numpy as np
import os
import argparse
import pandas as pd
import multiprocessing
import os.path as osp

def count_seqs_in_maf(human_bed, maf_file, output_dir, chromosome):
    """Count number of gaps in non-human species at every position in the human genome in a MAF alignment file and save as compressed .npz files by chromosome

    :param human_bed: str, path to BED file for human genome
    :param maf_file: str, path to MAF file
    :param output_dir: str, path to directory to save output .npz files
    :param chromosome: str, chromosome to process
    """
    # read human bed and map chrom sizes
    print(f"Starting to process chromosome {chromosome} in MAF file: {maf_file}")

    bed = pd.read_csv(human_bed, sep="\t", header=None, names=["chrom", "start", "end"])
    chrom_sizes = {
        "chr" + chromosome: bed[bed["chrom"] == ("chr" + chromosome)]["end"].values
    }

    # dictionary to store counts
    gap_counts = {}

    #dictionary to store the number of sequences at every position
    species_positions_aligned_count = {}

    # open MAF file
    with open(maf_file, "r") as maf:
        for line in maf:
            if line.startswith("a"):  # new alignment block
                human_seq = None
                non_human_seqs = []

            elif line.startswith("s"):  # sequence
                parts = line.split()
                species, chrom, start, length = (
                    parts[1].split(".")[0],
                    parts[1].split(".")[1],
                    int(parts[2]),
                    int(parts[3]),
                )
                seq = parts[6]

                # check if the sequence is human
                if "Homo_sapiens" in species:
                    human_seq = (chrom, start, length, seq)
                else:
                    non_human_seqs.append(seq)

            elif line.startswith("\n"):  # end of alignment
                if human_seq is None:
                    continue
                chrom, start, length, seq = human_seq
                # check if chrom is the one we are processing
                if chrom == "chr" + chromosome:
                    if chrom not in gap_counts:
                        print(
                            f"Processed alignment block ending at position {start + len(seq)} in chromosome {chrom}"
                        )
                        max_chromosome_length = chrom_sizes[chrom][0]
                        gap_counts[chrom] = np.zeros(max_chromosome_length, dtype=int)
                        species_positions_aligned_count[chrom] = len(non_human_seqs) - gap_counts[chrom]

                    for seq in non_human_seqs:
                        gaps = np.array([1 if base == "-" or base =="N" else 0 for base in seq])
                        required_length = start + len(seq)
                        if required_length > gap_counts[chrom].size:
                            gap_counts[chrom] = np.pad(
                                gap_counts[chrom],
                                (0, required_length - gap_counts[chrom].size),
                            )
                        gap_counts[chrom][start : start + len(seq)] += gaps
                    
                    #now we will subtract the gap counts from the number of sequences at every position to get the species positions aligned count:
                    species_positions_aligned_count[chrom] = len(non_human_seqs) - gap_counts[chrom]


                    # save intermediate results periodically
                    if start % 1000000 == 0:  
                        np.savez_compressed(
                            os.path.join(output_dir, f"{chrom}_alignment_counts.npz"),
                            counts=species_positions_aligned_count[chrom],
                        )
                        print(
                            f"Saved intermediate {chrom}_alignment_counts.npz at position {start}"
                        )

    # save final results
    for chrom, counts in gap_counts.items():
        np.savez_compressed(
            os.path.join(output_dir, f"{chrom}_alignment_counts.npz"), counts=counts
        )
        print(f"Saved final {chrom}_alignment_counts.npz")

    return species_positions_aligned_count


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
        default="/data/retry/241-mammalian-2020v2b.maf",
        help="Directory to the MAF file",
    )
    parser.add_argument(
        "--file_path",
        type=str,
        default="/home/t-mconsens/gamba/data_processing/data/240-mammalian/",
        help="Directory to save the BED file",
    )
    args = parser.parse_args()

    # list of chromosomes to process
    chromosomes = [str(i) for i in range(1, 23)] + ["X"]

    # number of available CPU cores
    num_cores = min(len(chromosomes), multiprocessing.cpu_count())

    # multiprocessing pool with the number of cores
    pool = multiprocessing.Pool(processes=num_cores)

    # list of arguments for each process
    process_args = [
        (args.human_bed, args.maf_file, args.file_path, chrom) for chrom in chromosomes
    ]

    print("Starting multiprocessing to count gaps...")

    # multiprocessing pool to execute the function in parallel
    pool.starmap(count_seqs_in_maf, process_args)

    print("All processes completed.")


if __name__ == "__main__":
    main()
