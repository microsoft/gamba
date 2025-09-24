#!/usr/bin/env python3
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import pyBigWig
from pyfaidx import Fasta
import json
from tqdm import tqdm
import pandas as pd
import logging
import sys
import random
from pathlib import Path
from scipy import stats

from Bio.Seq import Seq
#import Counter
from collections import Counter

import umap
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import os
from sklearn.neighbors import NearestNeighbors



from Bio.Seq import Seq  # make sure this import exists!

# def extract_context(bigwig_file, region, genome, model_type, context_window=2048):
#     """
#     Returns a dict(region + sequence, scores, feature_start_in_window, feature_end_in_window)
#     or None if the region cannot be extracted.
#     Rules:
#       gamba, hyenaDNA -> 2048 asymmetric (feature flush at window end for '+' and at start for '-')
#       caduceus, nt-ms, nt-human -> 2048 symmetric (feature centered)
#       phyloGPN -> 481 centered
#     """
#     chrom = region["chrom"]
#     chrom_length = len(genome[chrom])

#     # Normalize coords & strand
#     s0, e0 = int(region["start"]), int(region["end"])
#     strand = region.get("strand", "+")
#     if e0 < s0:
#         strand = "-"  # inverted coords imply '-' if not provided
#     feature_start, feature_end = min(s0, e0), max(s0, e0)  # half-open math uses these
#     feature_len = feature_end - feature_start
#     if feature_len <= 0:
#         return None

#     # Per-model window length + placement mode
#     if model_type in ("gamba", "hyenaDNA"):
#         max_len = 2048
#         # Asymmetric: put feature at the window edge depending on strand
#         max_ctx = max_len - min(feature_len, max_len)
#         if feature_len > max_len:
#             # clip to max_len but keep the terminal 1000bp like your previous logic
#             keep = min(1000, max_len)
#             if strand == "+":
#                 window_end = feature_end
#                 window_start = max(0, window_end - keep)
#                 fs, fe = 0, keep
#             else:
#                 window_start = feature_start
#                 window_end = min(chrom_length, window_start + keep)
#                 fs, fe = 0, window_end - window_start
#         else:
#             if strand == "+":
#                 window_end = feature_end
#                 window_start = max(0, window_end - (feature_len + max_ctx))
#                 fs = feature_start - window_start
#                 fe = feature_end - window_start
#             else:  # '-'
#                 window_start = feature_start
#                 window_end = min(chrom_length, window_start + (feature_len + max_ctx))
#                 fs = 0
#                 fe = feature_end - feature_start

#     elif model_type in ("caduceus", "nt-ms", "nt-human"):
#         max_len = 2048
#         # Symmetric: center feature in window
#         if feature_len >= max_len:
#             # clip to max_len around center
#             center = (feature_start + feature_end) // 2
#             window_start = max(0, center - max_len // 2)
#             window_end = min(chrom_length, window_start + max_len)
#             # recompute to ensure exact half-open bounds
#             window_start = max(0, window_end - max_len)
#         else:
#             total_ctx = max_len - feature_len
#             left = total_ctx // 2
#             right = total_ctx - left
#             window_start = max(0, feature_start - left)
#             window_end = min(chrom_length, feature_end + right)
#             # fix size exactly max_len
#             if window_end - window_start > max_len:
#                 window_end = window_start + max_len
#             elif window_end - window_start < max_len:
#                 window_start = max(0, window_end - max_len)
#         fs = feature_start - window_start
#         fe = feature_end - window_start

#     elif model_type == "phyloGPN":
#         max_len = 481
#         if feature_len >= max_len:
#             center = (feature_start + feature_end) // 2
#             window_start = max(0, center - max_len // 2)
#             window_end = min(chrom_length, window_start + max_len)
#             window_start = max(0, window_end - max_len)
#         else:
#             total_ctx = max_len - feature_len
#             left = total_ctx // 2
#             right = total_ctx - left
#             window_start = max(0, feature_start - left)
#             window_end = min(chrom_length, feature_end + right)
#             if window_end - window_start > max_len:
#                 window_end = window_start + max_len
#             elif window_end - window_start < max_len:
#                 window_start = max(0, window_end - max_len)
#         fs = feature_start - window_start
#         fe = feature_end - window_start

#     else:
#         raise ValueError(f"Unknown model_type: {model_type}")

#     # Clamp to chromosome and validate half-open invariants
#     window_start = max(0, window_start)
#     window_end = min(chrom_length, window_end)
#     region_len = window_end - window_start
#     if region_len <= 0:
#         return None
#     if not (0 <= fs <= fe <= region_len):
#         # If feature spills (e.g., near chrom ends), trim to window
#         fs = max(0, min(fs, region_len))
#         fe = max(fs, min(fe, region_len))
#         if fs == fe:  # degenerate
#             return None

#     # Extract sequence and phyloP safely
#     try:
#         seq = genome[chrom][window_start:window_end].seq  # pyfaidx end-exclusive
#         with pyBigWig.open(bigwig_file) as bw:
#             # ensure bigWig bounds
#             bw_start = max(0, window_start)
#             bw_end = min(chrom_length, window_end)
#             if bw_end <= bw_start:
#                 return None
#             scores = extract_phyloP_scores(bw, chrom, bw_start, bw_end)
#         if scores is None or len(scores) != region_len:
#             return None

#         # Handle minus strand: reverse complement + flip ROI (half-open)
#         if strand == "-":
#             seq = str(Seq(seq).reverse_complement())
#             scores = scores[::-1]
#             L = region_len
#             fs, fe = (L - fe, L - fs)

#         # Final sanity checks
#         if not (0 <= fs <= fe <= len(seq)):
#             return None
#         if len(seq) != len(scores):
#             return None

#         return {
#             **region,
#             "sequence": seq,
#             "scores": scores,
#             "feature_start_in_window": int(fs),
#             "feature_end_in_window": int(fe),
#         }

#     except Exception as e:
#         logging.warning(f"[ERROR] Failed to extract {chrom}:{window_start}-{window_end} - {e}")
#         return None

def extract_context(bigwig_file, region, genome, model_type=None, context_window=None):
    """
    Returns dict(region + sequence, scores, feature_start_in_window, feature_end_in_window)
    or None if the region cannot be extracted.

    Windowing rules:
      - model_type in {"gamba","hyenaDNA"}        -> 2048 asymmetric (feature at end for '+', at start for '-')
      - model_type in {"caduceus","nt-ms","nt-human"} -> 2048 symmetric (feature centered)
      - model_type == "phyloGPN"                  -> 481 symmetric (centered)
      - model_type is None (e.g., baselines)      -> 2048 symmetric (centered)
    """
    chrom = region["chrom"]
    chrom_length = len(genome[chrom])

    # Normalize coords & strand
    s0, e0 = int(region["start"]), int(region["end"])
    strand = region.get("strand", "+")
    if e0 < s0:
        strand = "-"  # inverted coords imply '-' if not provided
    feature_start, feature_end = min(s0, e0), max(s0, e0)
    feature_len = feature_end - feature_start
    if feature_len <= 0:
        return None

    # --- policy selection (baseline-friendly) ---
    # None => baseline => symmetric 2048
    if model_type in ("gamba"):
        policy = "asym"
        max_len = 2048
    elif model_type in ("caduceus", "baseline"):
        policy = "sym"
        max_len = 2048
    elif model_type in ("hyenaDNA"):
        policy = "asym"
        max_len = 160000
    elif model_type in ("caduceus-theirs"):
        policy = "sym"
        max_len = 131000
    elif model_type == "phyloGPN":
        policy = "sym"
        max_len = 481
    elif model_type in ("nt-ms", "nt-human"):
        policy = "sym"
        max_len = 6000
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    if context_window is not None and model_type != "phyloGPN":
        max_len = context_window

    # --- compute window per policy ---
    if policy == "asym":
        # Asymmetric: feature flush to window edge by strand
        if feature_len > max_len:
            keep = min(1000, max_len)
            if strand == "+":
                window_end = feature_end
                window_start = max(0, window_end - keep)
                fs, fe = 0, keep
            else:
                window_start = feature_start
                window_end = min(chrom_length, window_start + keep)
                fs, fe = 0, window_end - window_start
        else:
            max_ctx = max_len - feature_len
            if strand == "+":
                window_end = feature_end
                window_start = max(0, window_end - (feature_len + max_ctx))
                fs = feature_start - window_start
                fe = feature_end - window_start
            else:
                window_start = feature_start
                window_end = min(chrom_length, window_start + (feature_len + max_ctx))
                fs = 0
                fe = feature_end - feature_start

    else:
        # Symmetric policies (2048 or 481 or 1000): center feature in window
        if feature_len >= max_len:
            center = (feature_start + feature_end) // 2
            window_start = max(0, center - max_len // 2)
            window_end = min(chrom_length, window_start + max_len)
            window_start = max(0, window_end - max_len)  # ensure exact length
        else:
            total_ctx = max_len - feature_len
            left = total_ctx // 2
            right = total_ctx - left
            window_start = max(0, feature_start - left)
            window_end = min(chrom_length, feature_end + right)
            # fix to exact max_len
            if window_end - window_start > max_len:
                window_end = window_start + max_len
            elif window_end - window_start < max_len:
                window_start = max(0, window_end - max_len)
        fs = feature_start - window_start
        fe = feature_end - window_start

    # Clamp & validate
    window_start = max(0, window_start)
    window_end = min(chrom_length, window_end)
    region_len = window_end - window_start
    if region_len <= 0:
        return None
    if not (0 <= fs <= fe <= region_len):
        fs = max(0, min(fs, region_len))
        fe = max(fs, min(fe, region_len))
        if fs == fe:
            return None

    # Extract sequence + phyloP
    try:
        seq = genome[chrom][window_start:window_end].seq
        with pyBigWig.open(bigwig_file) as bw:
            bw_start = max(0, window_start)
            bw_end = min(chrom_length, window_end)
            if bw_end <= bw_start:
                return None
            scores = extract_phyloP_scores(bw, chrom, bw_start, bw_end)
        if scores is None or len(scores) != region_len:
            return None

        # Reverse-complement & flip ROI for minus strand
        if strand == "-":
            seq = str(Seq(seq).reverse_complement())
            scores = scores[::-1]
            L = region_len
            fs, fe = (L - fe, L - fs)

        if not (0 <= fs <= fe <= len(seq)):
            return None
        if len(seq) != len(scores):
            return None

        return {
            **region,
            "sequence": seq,
            "scores": scores,
            "feature_start_in_window": int(fs),
            "feature_end_in_window": int(fe),
        }

    except Exception as e:
        logging.warning(f"[ERROR] Failed to extract {chrom}:{window_start}-{window_end} - {e}")
        return None


def extract_phyloP_scores(bigwig: pyBigWig.pyBigWig, chrom: str, start: int, end: int) -> list[float]:
    """
    Extract phyloP conservation scores from bigWig using interval blocks.
    Missing values default to 0.0. Scores are rounded to 2 decimal places.

    Args:
        bigwig: Open pyBigWig.BigWigFile object
        chrom: Chromosome name (e.g., 'chr1')
        start: 0-based start coordinate
        end: 0-based end coordinate

    Returns:
        List of rounded phyloP scores for [start, end)
    """
    region_length = end - start
    vals = np.zeros(region_length, dtype=np.float64)

    try:
        intervals = bigwig.intervals(chrom, start, end)

        if intervals is None:
            logging.warning(f"phyloP intervals is None for {chrom}:{start}-{end}")
        else:
            for interval_start, interval_end, value in intervals:
                relative_start = interval_start - start
                relative_end = interval_end - start
                vals[relative_start:relative_end] = value

        return np.round(vals, 2).tolist()

    except RuntimeError as e:
        logging.error(f"RuntimeError when extracting phyloP scores for {chrom}:{start}-{end}: {e}")
        return np.round(vals, 2).tolist()


def extract_sequence_from_genome(genome: Fasta, chrom: str, start: int, end: int, strand: str) -> str:
    """
    Extract a sequence from the genome, reverse complementing it if on the minus strand.

    Args:
        genome: pyfaidx.Fasta object with loaded genome.
        chrom: Chromosome name (must match keys in genome, e.g., 'chr1').
        start: 0-based start coordinate (inclusive).
        end: 0-based end coordinate (exclusive).
        strand: '+' or '-'.

    Returns:
        DNA sequence as a string.
    """
    try:
        if chrom not in genome:
            raise ValueError(f"Chromosome {chrom} not found in genome FASTA.")

        seq = genome[chrom][start:end].seq.upper()

        if strand == '-':
            seq = str(Seq(seq).reverse_complement())

        return seq
    except Exception as e:
        print(f"Error extracting sequence from {chrom}:{start}-{end} ({strand}): {e}")
        return "N" * (end - start)

def load_bed_file(bed_path, category, genome, bw):
        regions = []
        with open(bed_path) as f:
            for line in f:
                if line.startswith("#"):
                    continue
                fields = line.strip().split("\t")
                if len(fields) < 3:
                    continue
                chrom = fields[0]
                start = int(fields[1])
                end = int(fields[2])
                strand = fields[5] if len(fields) >= 6 else "+"
                if chrom in genome and chrom in bw.chroms():
                    regions.append({
                        "chrom": chrom,
                        "start": start,
                        "end": end,
                        "strand": strand,
                        "category": category
                    })
        return regions