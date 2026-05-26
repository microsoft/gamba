import numpy as np
import os
import argparse
import pandas as pd
import multiprocessing
import os.path as osp
import torch


def count_seqs_in_maf(maf_file, output_dir, chromosome):
    """Count number of gaps in non-human species at every position in the human genome in a MAF alignment file and save as compressed .npz files by chromosome

    :param human_bed: str, path to BED file for human genome
    :param maf_file: str, path to MAF file
    :param output_dir: str, path to directory to save output .npz files
    :param chromosome: str, chromosome to process
    """
    # read human bed and map chrom sizes
    print(f"Starting to process chromosome {chromosome} in MAF file: {maf_file}")
    # create empty files for chrom_entropy and chrom_species_count if they don't exist
    if not os.path.exists(os.path.join(output_dir, f"{chromosome}_entropy.npz")):
        print("making file for entropy")
        np.savez_compressed(
            os.path.join(output_dir, f"{chromosome}_entropy.npz"), entropy=np.array([])
        )
    if not os.path.exists(os.path.join(output_dir, f"{chromosome}_species_count.npz")):
        print("making file for species count")
        np.savez_compressed(
            os.path.join(output_dir, f"{chromosome}_species_count.npz"),
            counts=np.array([]),
        )

    # open MAF file
    with open(maf_file, "r") as maf:
        for line in maf:
            if line.startswith("a"):  # new alignment block
                human_seq = None

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
                    # intialize a torch tensor of length length
                    seq_mat = torch.tensor([], dtype=torch.float32)
                else:
                    # turn seq_vec into list of ints: A=0, T=2, G=3, C=4, and "-" or N=1
                    seq_vec = torch.tensor(
                        [
                            (
                                1
                                if base == "A"
                                else (
                                    2
                                    if base == "C"
                                    else 3 if base == "G" else 4 if base == "T" else 0
                                )
                            )
                            for base in seq
                        ]
                    )
                    # torch.stack seq_vec
                    seq_vec_unsqueezed = seq_vec.unsqueeze(0)  # seq_vec 2D
                    seq_mat = torch.cat(
                        (seq_mat, seq_vec_unsqueezed), dim=0
                    )  # concatenate vertically
                    # free memory of the seq_vec
                    del seq_vec
                    # free memory of the read in lines
                    del line
            elif line.startswith("\n"):  # end of alignment
                # free memory of line
                del line
                if human_seq is None:
                    continue
                chrom, start, length, seq = human_seq
                # check if chrom is the one we are processing
                if chrom == "chr" + chromosome:
                    # check if only human species in the alignment block (i.e. no non-human species, so seq_mat is 1D)
                    if seq_mat.dim() == 1:
                        # set entropy = inf and species_positions_aligned_count = 0 for the entire length of the sequence
                        seq_length = seq_mat.size(0)
                        entropy = torch.full((seq_length,), np.inf, dtype=torch.float32)
                        species_positions_aligned_count = torch.zeros(
                            seq_length, dtype=torch.int64
                        )
                    else:
                        # dimensions of seq_mat are non_human_species_number, sequence_length
                        non_human_species_number, sequence_length = seq_mat.size()
                        # to calculate the number of non-gapped bases aligned at every position to human,
                        # sum over the non-human species dimension where all non-zero values are present
                        species_positions_aligned_count = torch.count_nonzero(
                            seq_mat, dim=0
                        )

                        # one-hot encode seq_mat
                        # where ( A=1, C=2, G=3, T=4, -/N=0)
                        seq_mat_one_hot = torch.nn.functional.one_hot(
                            seq_mat.long(), num_classes=5
                        )
                        del seq_mat

                        # frequency of each base at each position
                        base_frequencies = (
                            torch.sum(seq_mat_one_hot, dim=0) / non_human_species_number
                        )
                        del seq_mat_one_hot

                        # calculate entropy at every position
                        # avoid log(0) by adding a small value to the frequencies
                        epsilon = 1e-9
                        entropy = -torch.sum(
                            base_frequencies * torch.log(base_frequencies + epsilon),
                            dim=1,
                        )

                    # load existing data from files
                    existing_entropy = np.load(
                        os.path.join(output_dir, f"{chromosome}_entropy.npz")
                    )["entropy"]
                    existing_species_count = np.load(
                        os.path.join(output_dir, f"{chromosome}_species_count.npz")
                    )["counts"]

                    # append new data to existing data
                    new_entropy = np.concatenate((existing_entropy, entropy.numpy()))
                    new_species_count = np.concatenate(
                        (
                            existing_species_count,
                            species_positions_aligned_count.numpy(),
                        )
                    )

                    # save updated data back to files
                    np.savez_compressed(
                        os.path.join(output_dir, f"{chromosome}_entropy.npz"),
                        entropy=new_entropy,
                    )
                    np.savez_compressed(
                        os.path.join(output_dir, f"{chromosome}_species_count.npz"),
                        counts=new_species_count,
                    )

                    # free everything else we don't need
                    del human_seq
                    del entropy
                    del species_positions_aligned_count
                    del base_frequencies

                    # save intermediate results periodically
                    if start % 1000000 == 0:
                        # save entropy and counts values
                        np.savez_compressed(
                            os.path.join(
                                output_dir, f"{chromosome}_entropy_and_species.npz"
                            ),
                            entropy=chrom_entropy.numpy(),
                            counts=chrom_species_count.numpy(),
                        )
                        print(
                            f"Saved intermediate {chromosome}_entropy_and_species.npz at position {start}"
                        )
                        del chrom_entropy
                        del chrom_species_count

    # save final results
    np.savez_compressed(
        os.path.join(output_dir, f"{chromosome}_entropy_and_species.npz"),
        entropy=chrom_entropy.numpy(),
        counts=chrom_species_count.numpy(),
    )
    print(f"Saved {chromosome}_entropy_and_species.npz")


def main():
    # process command line arguments
    parser = argparse.ArgumentParser(
        description="Generate a BED file from the hg38.chrom.sizes file"
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
        default="/data/mica",
        help="Directory to save the gaps file",
    )
    args = parser.parse_args()

    # list of chromosomes to process include X
    chromosomes = [str(i) for i in range(11, 24)]
      # [str(i) for i in range(1, 24)] + ["X"]

    # number of available CPU cores
    num_cores = min(len(chromosomes), multiprocessing.cpu_count())

    # multiprocessing pool with the number of cores
    pool = multiprocessing.Pool(processes=num_cores)

    # list of arguments for each process
    process_args = [(args.maf_file, args.file_path, chrom) for chrom in chromosomes]

    print(f"Starting multiprocessing to count species for chromosomes {chromosomes}...")

    # multiprocessing pool to execute the function in parallel
    pool.starmap(count_seqs_in_maf, process_args)

    print("All processes completed.")


if __name__ == "__main__":
    main()
