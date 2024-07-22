import os
import argparse
import pandas as pd
import multiprocessing
import os
from Bio.AlignIO.MafIO import MafIndex
import numpy as np
from evodiff.utils import Tokenizer
from gamba.constants import DNA_ALPHABET_PLUS
import sys
import subprocess


def save_chromosome_sequences(maf_file, file_path, chromosome, chrom_size):
    print(f"Starting to process chromosome {chromosome} in MAF file: {maf_file}")
    #strip the path to the maf file so its just the name at the end of last "/"
    maf_file_name = maf_file.split("/")[-1]
    #remove the ".maf" from the end of the maf file name
    maf_file_name = maf_file_name[:-4]

    #make Maf index file:
    os.makedirs(f"{file_path}maf_indexes/", exist_ok=True)
    maf_directory = f"{file_path}maf_indexes/"

    print(f"{maf_directory}{maf_file_name}_{chromosome}.mafindex")

    idx = MafIndex(f"{maf_directory}{maf_file_name}_{chromosome}.mafindex",  maf_file, f"Homo_sapiens.{chromosome}")
    
    results = idx.search([0], [chrom_size])
    tokenizer = Tokenizer(DNA_ALPHABET_PLUS)
    species_seqs = {} 


    for multiple_alignment in results:

        for seqrec in multiple_alignment:
            species = seqrec.id.split('.')[0]
            if species not in species_seqs:
                species_seqs[species] = []
                print("new species:", species)
            species_seq = np.array(list(str(seqrec.seq).replace("-", "").upper()))
            species_seq = tokenizer.tokenize(species_seq)
            species_seqs[species].append(species_seq)

    for species in species_seqs:
        species_seqs[species] = np.array(list(''.join(species_seqs[species])))
        # save the sequences for each species into file_path/chr{chromosome}/{species}.npz
        species_path = os.path.join(file_path, f"chr{chromosome}", f"{species}.npz")
        print(f"saving species {species} sequences to {species_path}")
        os.makedirs(species_path, exist_ok=True)
        np.savez_compressed(species_path, sequence=species_seq)

def main():
    # Process command line arguments
    parser = argparse.ArgumentParser(
        description="Generate a BED file from the hg38.chrom.sizes file"
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
        default="/data/mica/",
        help="Directory to save the multispecies file",
    )
    args = parser.parse_args()

    # chromosomes to process
    bed = pd.read_csv(args.human_bed, sep="\t", header=None, names=["chrom", "start", "end"])
    chromosomes = [str(i) for i in range(1, 23)] + ["X"]
    chrom_sizes = {
        "chr"  + chrom: bed[bed["chrom"] == ("chr" + chrom)]["end"].values
        for chrom in chromosomes
    }

    # make the directory to save the sequences
    os.makedirs(args.file_path, exist_ok=True)

    # multiprocessing pool with the number of chromosomes
    pool = multiprocessing.Pool(processes=len(chromosomes))

    if os.geteuid() == 0:
        print("We're root!")
    else:
        print("We're not root.")
        subprocess.call(['sudo', 'python3', *sys.argv])
        print("and now we are")

    # list of arguments for each process
    process_args = [( args.maf_file, args.file_path,  f"chr{chrom}", chrom_sizes[f"chr{chrom}"]) for chrom in chromosomes]

    print("Starting multiprocessing to save chromosome sequences...")

    # Multiprocessing pool to execute the function in parallel
    pool.starmap(save_chromosome_sequences, process_args)

    print("Completed saving sequences for all chromosomes.")

if __name__ == "__main__":
    main()