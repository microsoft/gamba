import pyBigWig
import pandas as pd
import os
import argparse
import numpy as np
from pyfaidx import Fasta
import json
import logging
from evodiff.utils import Tokenizer

_logger = logging.getLogger(__name__)

# import gamba using sys.append
import sys
sys.path.append("../gamba")
from gamba.constants import DNA_ALPHABET_PLUS

def make_datasets(
    bigwig_file: str,
    bed: pd.DataFrame,
    file_path: str,
    genome_fasta: str,
    splits_file: str,
    chunk_size: int = 1000000,  # chunk size
    verbose: bool = True,
):
    # open the bigwig file
    bw = pyBigWig.open(bigwig_file)

    # open the genome fasta file
    genome = Fasta(genome_fasta)

    # create directories for train, test, and valid if they don't exist
    os.makedirs(f"{file_path}train", exist_ok=True)
    os.makedirs(f"{file_path}test", exist_ok=True)
    os.makedirs(f"{file_path}valid", exist_ok=True)

    # use the splits json to save the numpy array as a compressed numpy file by chrom_num
    with open(splits_file, "r") as f:
        splits = json.load(f)

    # create a dictionary to map chromosomes to splits
    chromosome_splits = {chrom: split for split, chroms in splits.items() for chrom in chroms}

    # iterate over the BED file
    for index, row in bed.iterrows():
        chrom = row["chrom"]
        size = row["end"]
        chrom_num = chrom.split("chr")[1]
        print(f"Processing chromosome: {chrom}")

        # Process chromosome in chunks
        for start in range(0, size, chunk_size):
            end = min(start + chunk_size, size)
            print(f"Processing chunk {start} - {end} for {chrom}")

            # Get the sequence for this chunk
            sequence = genome[chrom][start:end].seq

            # Tokenize the sequence
            tokenizer = Tokenizer(DNA_ALPHABET_PLUS)
            sequence = tokenizer.tokenizeMSA(sequence)

            # Initialize the conservation scores for the chunk
            vals = np.zeros(end - start, dtype=np.float64)

            # Get the conservation scores from the bigwig file for the chunk
            intervals = bw.intervals(chrom, start, end)
            
            # Check if there are intervals; if not, skip this chunk
            if intervals is not None:
                for i_start, i_end, value in intervals:
                    vals[i_start - start:i_end - start] = value
            else:
                print(f"No intervals found for chunk {start} - {end} on chromosome {chrom}")

            # Get the split for the current chromosome
            split_name = chromosome_splits[chrom_num]
            if verbose:
                _logger.info(f"Saving {split_name} data for chromosome: {chrom_num} chunk {start}-{end}")
            split_dir = f"{file_path}{split_name}/"
            seq_cons_file = f"{split_dir}{chrom_num}_{start}_{end}.npz"
            os.makedirs(split_dir, exist_ok=True)

            # Save the chunk
            np.savez_compressed(seq_cons_file, sequence=sequence, conservation=vals)

            # Release memory for the chunk
            del sequence, vals
            import gc
            gc.collect()

        print(f"Finished processing chromosome: {chrom}")

    # close the bigwig file
    bw.close()
    if verbose:
        _logger.info(f"Finished processing all chromosomes.")


def main():
    # process command line arguments
    parser = argparse.ArgumentParser(
        description="Generate data files for training, testing, and validation sets"
    )
    parser.add_argument(
        "--bigwig_file",
        type=str,
        default="/home/mica/gamba/data_processing/data/240-mammalian/241-mammalian-2020v2.bigWig",
        help="Path to the bigwig file with phyloP scores",
    )
    parser.add_argument(
        "--bed_file",
        type=str,
        default="/home/mica/gamba/data_processing/data/240-mammalian/hg38.bed",
        help="File name of the bed file",
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
        "--splits_file",
        type=str,
        default="/home/mica/gamba/data_processing/data/240-mammalian/splits.json",
        help="Path to the splits JSON file",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=1000000,  # chunk size as a parameter
        help="Chunk size for processing large chromosomes",
    )
    args = parser.parse_args()

    # load the BED file to pandas df
    bed = pd.read_csv(
        args.bed_file, sep="\t", header=None, names=["chrom", "start", "end"]
    )

    make_datasets(
        args.bigwig_file, bed, args.file_path, args.genome_fasta, args.splits_file, chunk_size=args.chunk_size
    )


if __name__ == "__main__":
    main()

