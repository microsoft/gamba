import numpy as np
import os
import argparse
import pandas as pd
import multiprocessing
from Bio.AlignIO.MafIO import MafIndex
import json

def count_gaps_in_maf(maf_file, output_dir, split_name, chrom, chrom_size):
    """Count number of gaps in non-human species at every position in the human genome in a MAF alignment file and save as compressed .npz files by chromosome

    :param human_bed: str, path to BED file for human genome
    :param maf_file: str, path to MAF file
    :param output_dir: str, path to directory to save output .npz files
    :param chrom: str, chromosome to process
    """
    # read human bed and map chrom sizes
    print(f"Starting to process chromosome {chrom}, from split {split_name} in MAF file: {maf_file}")

    #strip the path to the maf file so its just the name at the end of last "/"
    maf_file_name = maf_file.split("/")[-1]
    #remove the ".maf" from the end of the maf file name
    maf_file_name = maf_file_name[:-4]

    #make Maf index file:
    os.makedirs(f"{output_dir}/maf_indexes/", exist_ok=True)
    maf_directory = f"{output_dir}/maf_indexes/"

    idx = MafIndex(f"{maf_directory}{maf_file_name}_{chrom}.mafindex",  maf_file, f"Homo_sapiens.{chrom}")
    
    split_dir = f"{output_dir}{split_name}/"
    os.makedirs(split_dir, exist_ok=True)

    # dictionary to store counts
    gap_counts = np.zeros(chrom_size, dtype=int)

    # search for alignments in the chromosome
    results = idx.search([0], [chrom_size])

    for multiple_alignment in results:
        human_seq = None
        non_human_seqs = []

        for seqrec in multiple_alignment:
            if seqrec.id.startswith("Homo_sapiens.{chrom}"):
                human_seq = seqrec
            else:
                non_human_seqs.append(seqrec)

        if human_seq:
            start = int(human_seq.annotations["start"])
            seq = str(human_seq.seq)
            for seqrec in non_human_seqs:
                gaps = np.array([1 if base == "-" else 0 for base in str(seqrec.seq)])
                required_length = start + len(seq)
                if required_length > gap_counts.size:
                    gap_counts = np.pad(
                        gap_counts,
                        (0, required_length - gap_counts.size),
                    )
                gap_counts[start : start + len(seq)] += gaps

    # Save the gap counts for the chromosome
    np.savez_compressed(
        os.path.join(split_dir, f"{chrom}_gap_counts.npz"), counts=gap_counts
    )
    print(f"Saved {chrom}_gap_counts.npz")

def main():
    # process command line arguments
    parser = argparse.ArgumentParser(
        description="Count gaps in the MAF file for each chromosome"
    )
    parser.add_argument(
        "--human_bed",
        type=str,
        default="/home/t-mconsens/gamba/data_processing/data/240-mammalian/hg38.bed",
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
        help="Directory to save the gaps file",
    )
    parser.add_argument(
        "--splits_file",
        type=str,
        default="/home/t-mconsens/gamba/data_processing/data/240-mammalian/splits.json",
        help="Path to the splits JSON file",
    )
    args = parser.parse_args()

    # chromosomes to process
    bed = pd.read_csv(args.human_bed, sep="\t", header=None, names=["chrom", "start", "end"])
    chromosomes = [str(i) for i in range(1, 23)] + ["X"]
    chrom_sizes = {
        "chr"  + chrom: bed[bed["chrom"] == ("chr" + chrom)]["end"].values
        for chrom in chromosomes
    }

    # multiprocessing pool with the number of chromosomes
    pool = multiprocessing.Pool(processes=len(chromosomes))

    with open(args.splits_file, "r") as f:
        splits = json.load(f)

    # create a dictionary to map chromosomes to splits
    chromosome_splits = {}
    for split, chroms in splits.items():
        for chrom in chroms:
            chromosome_splits[chrom] = split

    # list of arguments for each process
    process_args = [( args.maf_file, args.file_path, chromosome_splits[str(chrom)], f"chr{chrom}", chrom_sizes[f"chr{chrom}"]) for chrom in chromosomes]

    print("Starting multiprocessing to count gaps...")

    # multiprocessing pool to execute the function in parallel
    pool.starmap(count_gaps_in_maf, process_args)

    print("Completed gap counting for all chromosomes.")

if __name__ == "__main__":
    main()