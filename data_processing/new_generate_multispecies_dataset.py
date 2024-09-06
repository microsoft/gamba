import numpy as np
import os
import argparse

def parse_maf(maf_file, species_list, chromosome, output_folder, start_position=0, region_size=10_000_000):
    # init arrays for each species with gap tokens '-' of length 10Mb
    arrays = {species: np.full(region_size, '-') for species in species_list}

    # human position
    current_human_pos = 0

    # read through the MAF file
    with open(maf_file, 'r') as f:
        for line in f:
            line = line.strip()

            if line.startswith('a'):
                # start of a new alignment block, reset current_human_pos for new alignment block
                continue

            elif line.startswith('s'):
                parts = line.split()
                species = parts[1].split('.')[0]  # species name before the dot (.)
                species_start = int(parts[2])  # position of the sequence in the species
                sequence_length = int(parts[3])  # length of the aligned sequence
                aligned_sequence = parts[6]  #  actual aligned sequence

                # Homo_sapiens as the reference
                if species == 'Homo_sapiens':
                    current_human_pos = species_start - start_position  # position relative to start
                    print(f"Reading homo_sapiens data at {current_human_pos}")
                    if current_human_pos < 0 or current_human_pos >= region_size:
                        # skip if position is outside the 10Mb region
                        continue

                # align all species to the same position as Homo_sapiens
                if species in species_list:
                    for i, base in enumerate(aligned_sequence):
                        pos = current_human_pos + i
                        if pos < region_size:
                            arrays[species][pos] = base

            elif line == '':  # end of block
                continue

    # create directory for saving arrays if it doesn't exist
    output_dir = chromosome
    os.makedirs(output_dir, exist_ok=True)

    # save each species array as a .npy file
    for species, array in arrays.items():
        filename = f'{output_folder}/{output_dir}/{species}_{start_position}_{start_position + region_size}.npy'
        np.save(filename, array)

def main():
    parser = argparse.ArgumentParser(
        description="Generate a multispecies dataset from MAF files"
    )
    parser.add_argument(
        "--maf_file",
        type=str,
        default="/data/mica/proj/uppstore2017228/KLT.04.200M/200m_MD/data/new_250_MAMMALS_v2_20201120/MAF/HUMAN/241MAMMALS/human_onlychr_v2_mdong_10Mb_241MAMMALS/chr1.150000000.10000000.maf",
        help="Path to the MAF file",
    )
    parser.add_argument(
        "--file_path",
        type=str,
        default="/data/mica/",
        help="Directory to save the multispecies file",
    )
    parser.add_argument(
        "--chrom_sizes",
        type=str,
        default="/home/t-mconsens/gamba/data_processing/data/240-mammalian/chrom_sizes.txt",
        help="File containing the sizes of each chromosome",
    )
    parser.add_argument(
        "--species_list",
        type=str,
        default="/home/t-mconsens/gamba/data_processing/data/240-mammalian/species_list.txt",
        help="File containing the list of species to include in the dataset",
    )
    args = parser.parse_args()
    
    # read chromosome sizes
    chrom_sizes = {}
    with open(args.chrom_sizes, 'r') as f:
        for line in f:
            chrom, size = line.strip().split()
            chrom_sizes[chrom] = int(size)

    # directory to save the sequences
    os.makedirs(args.file_path, exist_ok=True)

    #read the species list
    with open(args.species_list, 'r') as f:
        species_list = [line.strip() for line in f]

    # process the single MAF file
    maf_file = args.maf_file
    chromosome = "chr1"  #  chromosome 1 for testing
    #extract start position from the name of the file (chr1.150000000.10000000.maf  would be 150000000)
    start = int(maf_file.split('/')[-1].split('.')[1])
    #get end based on 10Mb after start
    end = start + 10000000  # 10Mb segment

    print(f"Maf file: {maf_file}, chromosome: {chromosome}, start: {start}, end: {end}")

    parse_maf(maf_file, species_list, chromosome, output_folder=args.file_path, start_position=start)

    print("Completed saving sequences for all species.")

if __name__ == "__main__":
    main()