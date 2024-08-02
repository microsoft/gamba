import json
import numpy as np
import os
import os.path as osp
import pandas as pd
from scipy.spatial.distance import cdist
from sequence_models.utils import parse_fasta
from sequence_models.constants import MSA_ALPHABET, GAP, START, STOP
from torch.utils.data import Dataset


def parse_msa(path):
    parsed_msa = parse_fasta(path)
    parsed_msa = list(
        filter(None, parsed_msa)
    )  # get rid of any empty entries from commented inputs
    parsed_msa = [
        [char for char in seq if (char.isupper() or char == "-") and not char == "."]
        for seq in parsed_msa
    ]  # get rid of indels
    parsed_msa = ["".join(seq) for seq in parsed_msa]
    return parsed_msa


def msa_subsampling(sliced_msa, n_sequences, selection_type):
    """
    :param sliced_msa: msa sequences with query sliced out
    :param n_sequences: int, number of sequences in MSA to subsample to
    :return: constructed msa
    """
    if selection_type == "random":
        msa_depth = len(sliced_msa)
        random_idx = (
            np.random.choice(msa_depth - 1, size=n_sequences - 1, replace=False) + 1
        )
        msa_sequences = [list(sliced_msa[int(i)]) for i in random_idx]
    elif selection_type == "max_hamming":
        msa_sequences = []
        msa_subset = sliced_msa[1:]
        msa_subset_ind = np.arange(len(msa_subset))

        # start with rand seq to initialize for maxhamming subsampling
        random_ind = np.random.choice(msa_subset_ind)
        random_seq = msa_subset[random_ind]
        msa_sequences.append(list(random_seq))
        random_seq = np.expand_dims(random_seq, axis=0)
        msa_subset = np.delete(msa_subset, random_ind, axis=0)
        m = len(msa_subset)
        distance_matrix = np.ones((n_sequences - 2, m))
        # subsample new seqs using max distance between min(hamming) array
        for i in range(n_sequences - 2):
            curr_dist = cdist(random_seq, msa_subset, metric="hamming")
            distance_matrix[i] = curr_dist
            col_min = np.min(distance_matrix, axis=0)  # (1,num_choices)
            max_ind = np.argmax(col_min)
            random_ind = max_ind
            random_seq = msa_subset[random_ind]
            msa_sequences.append(list(random_seq))
            random_seq = np.expand_dims(random_seq, axis=0)
            msa_subset = np.delete(msa_subset, random_ind, axis=0)
            distance_matrix = np.delete(distance_matrix, random_ind, axis=1)
    else:
        raise Exception("Invalid selection type; choose from 'random' or 'max_hamming'")
    return msa_sequences


class ConservationDataset10000(Dataset):
    """
    Dataset that pulls sequence information and corresponding conservation scores in 10,000bp chunks

    The data folder should contain the following:
    - 'splits.json': a dict with keys 'train', 'valid', and 'test' mapping to lists of chromosomes
    - 'train', 'valid', 'test' folders with the following files:
    - '{chromosome}.npz': sequence data and conservation scores in 'sequence' and 'conservation' and 'error' keys
    where {chromosome} is the chromosome number, determined by splits.json to be in the correct split folder
    """

    def __init__(
        self,
        data_dir: str,
        split: str,
        max_len=2048,
        specific_chromosomes: list = None,
    ):
        self.data_dir = data_dir
        self.split = split
        with open(osp.join(data_dir, "splits.json"), "r") as f:
            self.chromosomes = json.load(f)[self.split]
        # load the bed file of chromosome sizes
        bed = pd.read_csv(
            osp.join(data_dir, "hg38.bed"),
            sep="\t",
            header=None,
            names=["chrom", "start", "end"],
        )
        self.max_len = max_len
        self.specific_chromosomes = specific_chromosomes
        # if specific chromosome provided, use it, otherwise use all in split
        if self.specific_chromosomes is not None:
            # if chromosome is not in split, raise error
            if not all(
                [
                    chromosome in self.chromosomes
                    for chromosome in self.specific_chromosomes
                ]
            ):
                raise ValueError("Chromosome not in split")
            self.chromosomes = self.specific_chromosomes
        # for each chromosome being used, make a mapping of chrom: size
        self.chrom_sizes = {
            chrom: bed[bed["chrom"] == ("chr" + chrom)]["end"].values
            for chrom in self.chromosomes
        }
        # split the size by arbitrary sequence length setting of 10,000 bp to determine how many 10,000bp sequences in each chromosome
        self.num_sequences = {
            chrom: (self.chrom_sizes[chrom] // 10000) for chrom in self.chromosomes
        }
        # indices from 0 to n, where n is the total number of 10,000bp sequences across all chromosomes (sum the number of sequences for each chromosome and then generate a range of that total)
        self.indices = list(
            range(int(sum(self.num_sequences[chrom] for chrom in self.chromosomes)))
        )
        self.file = None

    def __len__(self):
        return len(self.indices)

    def get_chrom_seq(self, idx: int):
        # based on idx, determine which chromosome the sequence is from
        chrom = None
        # sequence within chromosome
        seq_idx = None
        for chromosome in self.chromosomes:
            if idx < self.num_sequences[chromosome]:
                chrom = chromosome
                seq_idx = idx
                break
            else:
                idx -= self.num_sequences[chromosome]
        return chrom, int(seq_idx)

    def __getitem__(self, idx: int):
        chrom, seq_idx = self.get_chrom_seq(idx)
        if self.file is None or not self.file.endswith(f"test_{chrom}.npz"):
            self.file = osp.join(self.data_dir, self.split, f"test_{chrom}.npz")

        file_data = np.load(self.file)
        sequence = file_data["sequence"][seq_idx * 10000 : (seq_idx + 1) * 10000]
        conservation = file_data["conservation"][
            seq_idx * 10000 : (seq_idx + 1) * 10000
        ]
        gaps = file_data["gap"][seq_idx * 10000 : (seq_idx + 1) * 10000]

        # code to round conservation & scaling to two decimal places for prediction
        conservation = np.round(conservation, 2)
        gaps = np.round(gaps, 2)

        # right now random sampling, could change to some smarter way
        if len(sequence) - self.max_len > 0:
            start = np.random.choice(len(sequence) - self.max_len)
            stop = start + self.max_len
        else:
            start = 0
            stop = len(sequence)
        sequence = sequence[start:stop]
        conservation = conservation[start:stop]
        gaps = gaps[start:stop]
        return (sequence, conservation, gaps)


def chrom_sort_key(s):
    """Sort key for chromosomes."""
    import re

    return [int(text) if text.isdigit() else text for text in re.split(r"(\d+)", s)]


class ConservationDataset(Dataset):
    """
    Dataset that pulls sequence information and corresponding conservation scores randomly

    The data folder should contain the following:
    - 'splits.json': a dict with keys 'train', 'valid', and 'test' mapping to lists of chromosomes
    - 'train', 'valid', 'test' folders with the following files:
    - '{chromosome}.npz': sequence data and conservation scores in 'sequence' and 'conservation' and 'gap' keys
    where {chromosome} is the chromosome number, determined by splits.json to be in the correct split folder
    """

    def __init__(
        self,
        data_dir: str,
        split: str,
        max_len=2048,
        num_sequences: int = 100,
        specific_chromosomes: list = None,
    ):
        self.data_dir = data_dir
        self.split = split
        with open(osp.join(data_dir, "splits.json"), "r") as f:
            self.chromosomes = json.load(f)[self.split]
            # make sure self.chromosomes is sorted
            self.chromosomes = sorted(self.chromosomes, key=chrom_sort_key)
        # load the bed file of chromosome sizes
        bed = pd.read_csv(
            osp.join(data_dir, "hg38.bed"),
            sep="\t",
            header=None,
            names=["chrom", "start", "end"],
        )
        self.max_len = max_len
        self.specific_chromosomes = specific_chromosomes
        # if specific chromosome provided, use it, otherwise use all in split
        if self.specific_chromosomes is not None:
            # if chromosome is not in split, raise error
            if not all(
                [
                    chromosome in self.chromosomes
                    for chromosome in self.specific_chromosomes
                ]
            ):
                raise ValueError("Chromosome not in split")
            self.chromosomes = self.specific_chromosomes
        # for each chromosome being used, make a mapping of chrom: size
        self.chrom_sizes = {
            chrom: bed[bed["chrom"] == ("chr" + chrom)]["end"].values
            for chrom in self.chromosomes
        }   
        self.num_sequences = num_sequences
        # num sequences times: randomly sample a chromosome and a sequence start position from that chromosome (within chrom_size of that chromosome - max_len) and save it
        self.sequences = []
        for i in range(self.num_sequences):
            chrom = np.random.choice(self.chromosomes)
            start = np.random.choice(np.arange(self.chrom_sizes[chrom] - max_len))

            self.sequences.append((chrom, start))
        self.indices = list(range(self.num_sequences))
        self.file = None

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx: int):
        chrom, seq_idx = self.sequences[idx]
       
        # if self.file is None or not self.file.endswith(f"{chrom}.npz"):
        #     self.file = osp.join(self.data_dir, self.split, f"{chrom}.npz")

        if self.file is None or self.file != f"{chrom}.npz":
            self.file = osp.join(self.data_dir, self.split, f"{chrom}.npz")

        

        file_data = np.load(self.file)
        # print("current chromosome size:", self.chrom_sizes[chrom])
        # print("file data length of sequence:", len(file_data["sequence"]))
        # print("file data length of conservation:", len(file_data["conservation"]))
        sequence = file_data["sequence"][seq_idx : seq_idx + self.max_len]

        # check if sequence has over 10% composition of N nucleotides, represented as int 4,
        # if so, resample the sequence and save the new chrom and coordinates
        while np.count_nonzero(sequence == 4) > 0.1 * len(sequence):
            seq_idx = np.random.choice(
                np.arange(self.chrom_sizes[chrom] - self.max_len)
            )
            sequence = file_data["sequence"][seq_idx : seq_idx + self.max_len]
            self.sequences[idx] = (chrom, seq_idx)

        conservation = file_data["conservation"][seq_idx : seq_idx + self.max_len]

        # print(
        #     f"CHROM: {chrom}, SEQ_IDX: {seq_idx}, LENGTH OF CONSERVATION: {len(conservation)}, LENGTH OF SEQUENCE: {len(sequence)}"
        # )
        # gaps = file_data["gap"][seq_idx : seq_idx + self.max_len]

        # code to round conservation & scaling to two decimal places for prediction
        conservation = np.round(conservation, 2)
        # gaps = np.round(gaps, 2)

        return (sequence, conservation)  # , gaps)


class UniRefDataset(Dataset):
    """
    Dataset that pulls from UniRef/Uniclust downloads.

    The data folder should contain the following:
    - 'consensus.fasta': consensus sequences, no line breaks in sequences
    - 'splits.json': a dict with keys 'train', 'valid', and 'test' mapping to lists of indices
    - 'lengths_and_offsets.npz': byte offsets for the 'consensus.fasta' and sequence lengths
    """

    def __init__(self, data_dir: str, split: str, max_len=2048):
        self.data_dir = data_dir
        self.split = split
        with open(osp.join(data_dir, "splits.json"), "r") as f:
            self.indices = json.load(f)[self.split]
        metadata = np.load(osp.join(self.data_dir, "lengths_and_offsets.npz"))
        self.offsets = metadata["seq_offsets"]
        self.max_len = max_len
        self.file = None

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        idx = self.indices[idx]
        offset = self.offsets[idx]
        if self.file is None:
            self.file = open(self.data_dir + "consensus.fasta")

        self.file.seek(offset)
        consensus = self.file.readline()[:-1]
        if len(consensus) - self.max_len > 0:
            start = np.random.choice(len(consensus) - self.max_len)
            stop = start + self.max_len
        else:
            start = 0
            stop = len(consensus)
        consensus = consensus[start:stop]
        return (consensus,)


class OpenProteinDataset(Dataset):
    """
    Dataset that pulls from OpenFold OpenProteinSet https://registry.opendata.aws/openfold/
    Training data has been pre preprocessed into train/val/test sets via homology filtering against uniref sets

    The data folder should contain the following:
    - 'rtest_index.csv', 'test_index.csv', 'val_index.csv', 'train_index.csv', 4 csv files corresponding to the
       filepaths for MSAs in each split, MSA depths, and MSA sequence lengths

    Will return a subsampled MSA, with START/STOP tokens added to each seq in the MSA
    """

    def __init__(
        self,
        data_dir: str,
        split: str,
        selection_type: str,
        n_sequences: int,
        max_seq_len: int,
        min_depth=None,
        gap_fraction=4,
    ):
        """
        :param data_dir: str, path to directory containing openfold dataset
        :param split: str, split using for evaluation 'train', 'val', 'test', or 'rtest'
        :param selection_type: str, subsampling approach 'max_hamming' or 'random'
        :param n_sequences: int, number of sequences in MSA to subsample to
        :param max_seq_len: int, maximmum sequence length
        :param min_depth: minimum number of sequences needed to sample MSA
        :param gap_fraction: fraction of gap content to filter out
            (e.g 4 = filter out sequences with more than 1/4 (25%) gap content)
        """
        alphabet = list("".join(MSA_ALPHABET))
        self.a_to_i = {u: i for i, u in enumerate(alphabet)}
        self.i_to_a = np.array(list(alphabet))
        self.gap_id = self.a_to_i[GAP]
        self.start_id = self.a_to_i[START]
        self.stop_id = self.a_to_i[STOP]
        self.gap_fraction = gap_fraction

        # load in files from correct split
        split_path = os.path.join(data_dir, "out/" + split + "_index.csv")
        metadata = pd.read_csv(split_path, usecols=["path", "depth", "length"])

        # filter depths
        if min_depth is not None:
            print("filtering sequences less than", min_depth)
            metadata = metadata[metadata["depth"] >= min_depth]

        all_files = metadata["path"].values.tolist()
        self.filenames = [file for file in all_files]
        self.depths = metadata["depth"].values.tolist()
        self.lengths = metadata["length"].values.tolist()

        self.n_sequences = n_sequences
        self.max_seq_len = max_seq_len
        self.selection_type = selection_type
        self.min_depth = min_depth

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        path = self.filenames[idx]  # grab str from nested list
        msa_seq_len = self.lengths[idx]

        parsed_msa = parse_msa(path)
        tokenized_msa = np.vstack(
            [np.array([self.a_to_i[a] for a in seq]) for seq in parsed_msa]
        )

        if msa_seq_len > self.max_seq_len:
            slice_start = np.random.choice(msa_seq_len - self.max_seq_len + 1)
            seq_len = self.max_seq_len
        else:
            slice_start = 0
            seq_len = msa_seq_len

        # Slice sequence of max_seq_len
        sliced_msa = tokenized_msa[:, slice_start : slice_start + seq_len]
        # Reduce high-gap content in sliced sequences
        sliced_msa = [
            seq
            for seq in sliced_msa
            if (np.count_nonzero(seq == self.gap_id) < len(seq) / self.gap_fraction)
        ]
        msa_depth = len(sliced_msa)
        anchor_seq = sliced_msa[0]  # This is the query sequence in MSA

        if msa_depth <= self.n_sequences:
            output = sliced_msa
        else:
            msa_sequences = msa_subsampling(
                sliced_msa, self.n_sequences, selection_type=self.selection_type
            )
            output = [anchor_seq] + msa_sequences
        # Add start/stop tokens to each seq in MSA
        output = [
            "".join(self.i_to_a[[self.start_id] + list(seq) + [self.stop_id]])
            for seq in output
        ]
        return output
