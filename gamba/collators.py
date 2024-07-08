from typing import Callable, Literal, Optional, Sequence, Tuple

from evodiff.utils import Tokenizer
import numpy as np
from sequence_models.constants import START, STOP
import torch

from gamba.constants import FIM_MIDDLE, FIM_PREFIX, FIM_SUFFIX


def pad_to_mult(max_len, pad_to_mult):
    "helper function to pad to multiple of pad_to_mult"
    max_len = (
        max_len
        if pad_to_mult is None
        else pad_to_mult * torch.ceil(max_len / pad_to_mult).to(dtype=torch.int)
    )
    return max_len


class OAMaskCollator:
    def __init__(
        self, tokenizer: Callable, pad_to_multiple_of: Optional[int] = None
    ) -> None:
        self.tokenizer = tokenizer
        self.pad_to_mult = pad_to_multiple_of

    def __call__(
        self, sequences: Sequence[tuple]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Tokenize the input sequences, generate random masks, and convert into a tensor batch.

        Parameters:
        -----------
        sequences: Sequence[tuple]
            A sequence of tuples containing the input sequences as the first elem in each tuple.

        Returns:
        --------
        input_tokens: torch.Tensor
            The input tokens with the mask tokens in place.
        targets: torch.Tensor
            The target tokens.
        masks: torch.Tensor
            The mask tensor.
        timesteps: torch.Tensor
            The number of timesteps in the sequence.
        """
        # tokenize the samples (tokenizer accepts tuples of (seq,))
        tokenized = [torch.tensor(self.tokenizer.tokenize(s)) for s in sequences]

        # pad the max length if needed
        lens = torch.tensor([len(t) for t in tokenized])
        max_len = lens.max()
        max_len = pad_to_mult(max_len, self.pad_to_mult)

        # allocate the  output to fill
        input_tokens = torch.full(
            (len(tokenized), max_len), self.tokenizer.pad_id, dtype=torch.long
        )
        targets = input_tokens.clone()
        masks = torch.zeros(len(tokenized), max_len, dtype=torch.bool)

        # D - t + 1 where D is the length of the sequence and t is a random int in [1, D)
        timesteps = (
            lens - torch.tensor(np.random.randint(1, [max(2, lt) for lt in lens])) + 1
        )
        for i, (length, ts, toks) in enumerate(zip(lens, timesteps, tokenized)):
            input_tokens[i, : len(toks)] = toks
            targets[i, : len(toks)] = toks

            # generate the mask (num_timestep samples between [0, D-1])
            mask_idx = np.random.choice(length.item(), ts.numpy(), replace=False)
            masks[i, mask_idx] = True
            input_tokens[i, mask_idx] = self.tokenizer.mask_id

        return input_tokens, targets, masks, timesteps


class LMCollator:
    def __init__(
        self,
        tokenizer: Tokenizer,
        *,
        flip_prob: float = 0.0,
        fim_prob: float = 0.0,
        min_fim_prefix_len: int = 0,
        min_fim_suffix_len: int = 0,
        fim_mode: Literal["psm", "spm", "both"] = "both",
        simple_spm: bool = False,
        pad_to_multiple_of: Optional[int] = None,
        swap_bos_eos_on_flip: bool = True,
    ) -> None:
        """A collator which randomly converts a subset of samples into FIM samples.

        Parameters:
        -----------
        tokenizer: Callable
            A callable which tokenizes a string into a sequence of integers.
        fim_prob: float
            The probability of converting a sample into a FIM sample. Default is 0.5.
        min_fim_prefix_len: int
            The minimum length of the prefix for the FIM sample. Default is 0.
        min_fim_suffix_len: int
            The minimum length of the suffix for the FIM sample. Default is 0.
        fim_mode: Literal["psm", "spm", "both"]
            The mode of FIM to use. "psm" presents prefix-suffix-middle. "spm" presents suffix-prefix-middle.
            "both" presents both, switching between the two with equal probability. Default is both.
        simple_spm: bool
            If True, SPM samples are presented in the form <suffix>suffix-aa's<prefix>prefix-aa's<middle>middle-aa's.
            If False, SPM samples are presented in the form <prefix><suffix>suffix-aa's<middle>prefix-aa's middle-aa's.
            Default is False.
        flip_prob: float
            The probability of flipping the sample (always prior to FIM). Default is 0.5.
        pad_to_multiple_of: Optional[int]
            If not None, the length of the sequence will be padded to a multiple of this value.
        swap_bos_eos_on_flip: bool
            If True, the the sequence will be preceded by EOS (rather than BOS) when flipped. Default is True.
        """
        assert 0 <= fim_prob <= 1, "FIM probability must be in [0, 1]"
        assert 0 <= flip_prob <= 1, "Flip probability must be in [0, 1]"

        self.tokenizer = tokenizer
        self.fim_prob = fim_prob
        self.flip_prob = flip_prob
        self.pad_to_mult = pad_to_multiple_of
        self.fim_mode = fim_mode
        self.simple_spm = simple_spm
        self.swap_bos_eos_on_flip = swap_bos_eos_on_flip
        self.splitter = self.make_splitter(min_fim_prefix_len, min_fim_suffix_len)

        # intentionally keep them as arrays so we can concat later
        self.fim_pid = self.tokenizer.tokenize([FIM_PREFIX])
        self.fim_sid = self.tokenizer.tokenize([FIM_SUFFIX])
        self.fim_mid = self.tokenizer.tokenize([FIM_MIDDLE])
        self.start_id = self.tokenizer.tokenize([START])
        self.stop_id = self.tokenizer.tokenize([STOP])

    @staticmethod
    def make_splitter(min_prefix_len: int, min_suffix_len: int) -> Callable:
        def splitter(sequence: str) -> Tuple[str, str, str]:
            prefix_len = np.random.randint(
                min_prefix_len, len(sequence) - min_suffix_len
            )
            suffix_len = np.random.randint(min_suffix_len, len(sequence) - prefix_len)
            prefix = sequence[:prefix_len]
            suffix = sequence[-suffix_len:]
            middle = sequence[prefix_len:-suffix_len]
            return prefix, middle, suffix

        return splitter

    def _wrap(self, *args, flipped: bool):
        if flipped and self.swap_bos_eos_on_flip:
            return np.concatenate([self.stop_id, *args, self.start_id])
        return np.concatenate([self.start_id, *args, self.stop_id])

    def __call__(
        self, data: Sequence[str]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # tokenize
        tokenized = [self.tokenizer.tokenize(s) for s in data]

        # flip as needed
        if self.flip_prob > 0:
            maybe_flip = np.random.choice(
                [-1, 1], (len(tokenized),), p=[self.flip_prob, 1 - self.flip_prob]
            )
            tokenized = [s[::f] for s, f in zip(tokenized, maybe_flip)]

        is_flipped = (
            lambda i_: self.flip_prob > 0 and maybe_flip[i_] == -1
        )  # noqa: E731

        # FIM as needed
        maybe_fim = np.random.choice(
            [False, True], len(tokenized), p=[1 - self.fim_prob, self.fim_prob]
        )
        for i, fim in enumerate(maybe_fim):
            if fim:
                # randomly split the sample
                prefix, middle, suffix = self.splitter(tokenized[i])

                # select the fim type
                if self.fim_mode == "both":
                    fim_mode = np.random.choice(["psm", "spm"])
                else:
                    fim_mode = self.fim_mode

                if fim_mode == "psm":
                    tokenized[i] = self._wrap(
                        self.fim_pid,
                        prefix,
                        self.fim_sid,
                        suffix,
                        self.fim_mid,
                        middle,
                        flipped=is_flipped(i),
                    )
                else:
                    if self.simple_spm:
                        tokenized[i] = self._wrap(
                            self.fim_sid,
                            suffix,
                            self.fim_pid,
                            prefix,
                            self.fim_mid,
                            middle,
                            flipped=is_flipped(i),
                        )
                    else:
                        tokenized[i] = self._wrap(
                            self.fim_pid,
                            self.fim_sid,
                            suffix,
                            self.fim_mid,
                            prefix,
                            middle,
                            flipped=is_flipped(i),
                        )
            else:
                tokenized[i] = self._wrap(tokenized[i], flipped=is_flipped(i))

        # pad to a multiple of pad_to_mult
        max_len = max(len(s) for s in tokenized)

        # inflate to a mult of pad_to_mult
        if self.pad_to_mult is not None:
            max_len = (
                self.pad_to_mult * np.ceil(max_len / self.pad_to_mult).astype(int)
            ).item()

        out = torch.full(
            (len(tokenized), max_len), self.tokenizer.pad_id, dtype=torch.long
        )
        for i, s in enumerate(tokenized):
            out[i, : len(s)] = torch.tensor(s, device=out.device)

        lbls = out.clone()

        # no penalty for not predicting padding or FIM tokens
        lbls[lbls == self.tokenizer.pad_id] = -100
        lbls[lbls == self.fim_pid[0]] = -100
        lbls[lbls == self.fim_sid[0]] = -100
        lbls[lbls == self.fim_mid[0]] = -100

        return out, lbls


class gLMCollator:
    def __init__(
        self,
        tokenizer: Tokenizer,
        *,
        flip_prob: float = 0.0,
        fim_prob: float = 0.0,
        min_fim_prefix_len: int = 0,
        min_fim_suffix_len: int = 0,
        fim_mode: Literal["psm", "spm", "both"] = "both",
        simple_spm: bool = False,
        pad_to_multiple_of: Optional[int] = None,
        swap_bos_eos_on_flip: bool = True,
    ) -> None:
        """A collator which randomly converts a subset of samples into FIM samples.

        Parameters:
        -----------
        tokenizer: Callable
            A callable which tokenizes a string into a sequence of integers.
        fim_prob: float
            The probability of converting a sample into a FIM sample. Default is 0.5.
        min_fim_prefix_len: int
            The minimum length of the prefix for the FIM sample. Default is 0.
        min_fim_suffix_len: int
            The minimum length of the suffix for the FIM sample. Default is 0.
        fim_mode: Literal["psm", "spm", "both"]
            The mode of FIM to use. "psm" presents prefix-suffix-middle. "spm" presents suffix-prefix-middle.
            "both" presents both, switching between the two with equal probability. Default is both.
        simple_spm: bool
            If True, SPM samples are presented in the form <suffix>suffix-aa's<prefix>prefix-aa's<middle>middle-aa's.
            If False, SPM samples are presented in the form <prefix><suffix>suffix-aa's<middle>prefix-aa's middle-aa's.
            Default is False.
        flip_prob: float
            The probability of flipping the sample (always prior to FIM). Default is 0.5.
        pad_to_multiple_of: Optional[int]
            If not None, the length of the sequence will be padded to a multiple of this value.
        swap_bos_eos_on_flip: bool
            If True, the the sequence will be preceded by EOS (rather than BOS) when flipped. Default is True.
        """
        assert 0 <= fim_prob <= 1, "FIM probability must be in [0, 1]"
        assert 0 <= flip_prob <= 1, "Flip probability must be in [0, 1]"

        self.tokenizer = tokenizer
        self.fim_prob = fim_prob
        self.flip_prob = flip_prob
        self.pad_to_mult = pad_to_multiple_of
        self.fim_mode = fim_mode
        self.simple_spm = simple_spm
        self.swap_bos_eos_on_flip = swap_bos_eos_on_flip
        self.splitter = self.make_splitter(min_fim_prefix_len, min_fim_suffix_len)

        # intentionally keep them as arrays so we can concat later
        self.fim_pid = self.tokenizer.tokenize([FIM_PREFIX])
        self.fim_sid = self.tokenizer.tokenize([FIM_SUFFIX])
        self.fim_mid = self.tokenizer.tokenize([FIM_MIDDLE])
        self.start_id = self.tokenizer.tokenize([START])
        self.stop_id = self.tokenizer.tokenize([STOP])

    @staticmethod
    def make_splitter(min_prefix_len: int, min_suffix_len: int) -> Callable:
        def splitter(sequence: str) -> Tuple[str, str, str]:
            prefix_len = np.random.randint(
                min_prefix_len, len(sequence) - min_suffix_len
            )
            suffix_len = np.random.randint(min_suffix_len, len(sequence) - prefix_len)
            prefix = sequence[:prefix_len]
            suffix = sequence[-suffix_len:]
            middle = sequence[prefix_len:-suffix_len]
            return prefix, middle, suffix

        return splitter

    def __call__(self, data: Sequence[Tuple[np.ndarray, np.ndarray, np.ndarray]]):
        # unpack the input data
        sequence, scaling, gap = zip(*data)
        # sequence is already tokenized
        # wrap sequence in start and stop
        sequence = [
            np.concatenate([self.start_id, s, self.stop_id], axis=0) for s in sequence
        ]
        # add 0s as start and stop around the scaling and error params
        scaling = [
            np.pad(s, (1, 1), "constant", constant_values=(0, 0)) for s in scaling
        ]
        gap = [np.pad(g, (1, 1), "constant", constant_values=(0, 0)) for g in gap]
        # Pad each array type accordingly
        sequence, seq_lbls = self.pad_arrays(sequence, dtype=torch.long)
        scaling, scale_lbs = self.pad_arrays(scaling, dtype=torch.float32)
        gap, gap_lbs = self.pad_arrays(gap, dtype=torch.float32)

        out = torch.stack([sequence, scaling, gap])
        lbls = torch.stack([seq_lbls, scale_lbs, gap_lbs])

        return out, lbls

    def apply_transformations(self, sequence):
        # flip as needed
        if self.flip_prob > 0:
            maybe_flip = np.random.choice(
                [-1, 1], (len(sequence),), p=[self.flip_prob, 1 - self.flip_prob]
            )
            sequence = [s[::f] for s, f in zip(sequence, maybe_flip)]

        is_flipped = (
            lambda i_: self.flip_prob > 0 and maybe_flip[i_] == -1
        )  # noqa: E731

        # FIM as needed
        maybe_fim = np.random.choice(
            [False, True], len(sequence), p=[1 - self.fim_prob, self.fim_prob]
        )
        for i, fim in enumerate(maybe_fim):
            if fim:
                # randomly split the sample
                prefix, middle, suffix = self.splitter(sequence[i])

                # select the fim type
                if self.fim_mode == "both":
                    fim_mode = np.random.choice(["psm", "spm"])
                else:
                    fim_mode = self.fim_mode

                if fim_mode == "psm":
                    sequence[i] = self._wrap(
                        self.fim_pid,
                        prefix,
                        self.fim_sid,
                        suffix,
                        self.fim_mid,
                        middle,
                        flipped=is_flipped(i),
                    )
                else:
                    if self.simple_spm:
                        sequence[i] = self._wrap(
                            self.fim_sid,
                            suffix,
                            self.fim_pid,
                            prefix,
                            self.fim_mid,
                            middle,
                            flipped=is_flipped(i),
                        )
                    else:
                        sequence[i] = self._wrap(
                            self.fim_pid,
                            self.fim_sid,
                            suffix,
                            self.fim_mid,
                            prefix,
                            middle,
                            flipped=is_flipped(i),
                        )
            else:
                sequence[i] = self._wrap(sequence[i], flipped=is_flipped(i))
        return sequence

    def pad_arrays(self, sequence, dtype):
        # pad to a multiple of pad_to_mult
        max_len = max(len(s) for s in sequence)

        # inflate to a mult of pad_to_mult
        if self.pad_to_mult is not None:
            max_len = (
                self.pad_to_mult * np.ceil(max_len / self.pad_to_mult).astype(int)
            ).item()

        out = torch.full((len(sequence), max_len), self.tokenizer.pad_id, dtype=dtype)
        for i, s in enumerate(sequence):
            out[i, : len(s)] = torch.tensor(s, device=out.device)

        lbls = out.clone()

        # no penalty for not predicting padding
        lbls[lbls == self.tokenizer.pad_id] = -100

        return out, lbls


class MSAOAMasksCollator:
    def __init__(
        self, tokenizer: Callable, pad_to_multiple_of: Optional[int] = None
    ) -> None:
        self.tokenizer = tokenizer
        self.pad_to_mult = pad_to_multiple_of

    def __call__(
        self, batch_msa: "list"
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        depths = torch.tensor([len(msa) for msa in batch_msa])
        lens = torch.tensor([len(msa[0]) for msa in batch_msa])
        max_len = lens.max()
        max_len = pad_to_mult(max_len, self.pad_to_mult)
        max_depth = depths.max()
        max_depth = (
            max_depth
            if self.pad_to_mult is None
            else self.pad_to_mult * np.ceil(max_depth / self.pad_to_mult)
        )

        tokenized = [
            torch.tensor(np.vstack([self.tokenizer.tokenizeMSA(s) for s in msa]))
            for msa in batch_msa
        ]
        d = torch.tensor(
            [len(msa[:, 1:-1].flatten()) for msa in tokenized]
        )  # flattened msa shapes, excluding START/STOP tokens

        # allocate the  output to fill
        src = torch.full(
            (len(tokenized), max_depth, max_len),
            self.tokenizer.pad_id,
            dtype=torch.long,
        )
        targets = src.clone()
        masks = torch.zeros(len(tokenized), max_depth, max_len, dtype=torch.bool)

        # D - t + 1 where D is the length of the sequence and t is a random int in [1, D)
        timesteps = d - torch.tensor(np.random.randint(1, [max(2, lt) for lt in d])) + 1
        for i, (length, ts, msa) in enumerate(zip(d, timesteps, tokenized)):
            targets[i, : depths[i], : lens[i]] = msa
            input_tokens = msa[:, 1:-1].flatten()  # ignore START/STOP for masking

            # generate the mask on flattened MSAs
            mask_arr = torch.zeros(
                depths[i] * (lens[i] - 2), dtype=torch.bool
            )  # ignore START/STOP for masking
            mask_idx = np.random.choice(length.item(), ts.numpy(), replace=False)
            mask_arr[mask_idx] = True

            # reshape MSAs, being careful about START/STOP tokens
            mask_arr = mask_arr.reshape(depths[i], lens[i] - 2)
            masks[i, : depths[i], 1 : lens[i] - 1] = mask_arr
            input_tokens[mask_idx] = self.tokenizer.mask_id
            input_tokens = input_tokens.reshape(depths[i], lens[i] - 2)
            src[i, : depths[i], 1 : lens[i] - 1] = input_tokens
            src[i, : depths[i], 0] = self.tokenizer.start_id  # add back START
            src[i, : depths[i], lens[i] - 1] = self.tokenizer.stop_id  # add back STOP
        return src, timesteps, targets, masks


class MSAARCollator:
    def __init__(
        self, tokenizer: Callable, pad_to_multiple_of: Optional[int] = None
    ) -> None:
        self.tokenizer = tokenizer
        self.pad_to_mult = pad_to_multiple_of

    def __call__(self, batch_msa: "list") -> Tuple[torch.Tensor, torch.Tensor]:
        tokenized = [
            torch.tensor(np.vstack([self.tokenizer.tokenizeMSA(s) for s in msa]))
            for msa in batch_msa
        ]
        sep_tensor = torch.tensor([self.tokenizer.sep_id])

        refactored = [msa[:, 1:-1] for msa in tokenized]  # strip start/stop
        refactored = [
            torch.stack([torch.cat((seq, sep_tensor), 0) for seq in msa]).flatten()
            for msa in refactored
        ]
        max_len = max([len(msa) for msa in refactored])
        max_len = pad_to_mult(max_len, self.pad_to_mult)

        # pre-pad final array
        src = torch.full(
            (len(tokenized), max_len + 1), self.tokenizer.pad_id, dtype=torch.long
        )  # include start token
        targets = src.clone()
        src[:, 0] = self.tokenizer.start_id

        for i, msa in enumerate(refactored):
            src[i, 1 : len(msa[:-1]) + 1] = msa[
                :-1
            ]  # ignore last SEP token, offset inputs by 1
            targets[i, 0 : len(msa)] = msa
            targets[i, len(msa) - 1] = (
                self.tokenizer.stop_id
            )  # replace last SEP token with STOP

        return src, targets
