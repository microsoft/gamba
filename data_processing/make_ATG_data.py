#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from bisect import bisect_left, bisect_right

from pyfaidx import Fasta


# distance params
MIN_NEAR = 2000
MAX_NEAR = 5000
MIN_FAR = 100_000

_COMP = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def revcomp(s: str) -> str:
    return s.translate(_COMP)[::-1]


def parse_gtf_attrs(attr: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    parts = [p.strip() for p in attr.strip().split(";") if p.strip()]
    for p in parts:
        if " " not in p:
            continue
        k, v = p.split(" ", 1)
        out[k] = v.strip().strip('"')
    return out


def merge_intervals(intervals: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged = []
    cs, ce = intervals[0]
    for s, e in intervals[1:]:
        if s <= ce:
            ce = max(ce, e)
        else:
            merged.append((cs, ce))
            cs, ce = s, e
    merged.append((cs, ce))
    return merged


class IntervalMask:
    """O(logN) membership for merged half-open intervals [s,e)."""

    def __init__(self, intervals: List[Tuple[int, int]]):
        self.intervals = intervals
        self.starts = [s for s, _ in intervals]
        self.ends = [e for _, e in intervals]

    def window_fully_inside(self, start: int, end: int) -> bool:
        idx = bisect_right(self.starts, start) - 1
        if idx < 0:
            return False
        return end <= self.ends[idx]


@dataclass(frozen=True)
class CdsSeg:
    start: int  # 0-based, half-open
    end: int
    phase: int  # 0/1/2


@dataclass
class Transcript:
    transcript_id: str
    gene_id: str
    chrom: str
    strand: str
    cds: List[CdsSeg]
    blocks: Optional[List[Tuple[int, int, int]]] = None

    def build_blocks(self) -> None:
        if not self.cds:
            self.blocks = []
            return

        if self.strand == "+":
            segs = sorted(self.cds, key=lambda x: (x.start, x.end))
        else:
            segs = sorted(self.cds, key=lambda x: (x.start, x.end), reverse=True)

        first = segs[0]
        phase = int(first.phase) if first.phase in (0, 1, 2) else 0

        trimmed: List[Tuple[int, int]] = []
        for i, s in enumerate(segs):
            a, b = s.start, s.end
            if i == 0 and phase:
                if self.strand == "+":
                    a = min(b, a + phase)
                else:
                    b = max(a, b - phase)
            if a < b:
                trimmed.append((a, b))

        blocks: List[Tuple[int, int, int]] = []
        off = 0
        for a, b in trimmed:
            blocks.append((a, b, off))
            off += (b - a)
        self.blocks = blocks

    def cds_len(self) -> int:
        if self.blocks is None:
            self.build_blocks()
        if not self.blocks:
            return 0
        a, b, off0 = self.blocks[-1]
        return off0 + (b - a)

    def cds_offset_to_genomic(self, off: int) -> Optional[int]:
        if self.blocks is None:
            self.build_blocks()
        if off < 0:
            return None
        for a, b, off0 in (self.blocks or []):
            L = b - a
            if off0 <= off < off0 + L:
                d = off - off0
                if self.strand == "+":
                    return a + d
                else:
                    return (b - 1) - d
        return None

    def build_cds_seq(self, chrom_seq: str) -> str:
        if self.blocks is None:
            self.build_blocks()
        parts = [chrom_seq[a:b] for a, b, _ in (self.blocks or [])]
        s = "".join(parts)
        if self.strand == "-":
            s = revcomp(s)
        return s.upper()


def load_transcripts_from_gtf(gtf_path: str, chrom: str) -> List[Transcript]:
    tx: Dict[str, Transcript] = {}
    with open(gtf_path) as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9:
                continue
            seqname, _, feature, start1, end1, _, strand, phase, attrs = fields
            if seqname != chrom:
                continue
            if feature != "CDS":
                continue
            if strand not in ("+", "-"):
                continue

            attrs_str = attrs
            if 'tag "MANE_Select"' not in attrs_str:
                continue

            a = parse_gtf_attrs(attrs)
            tid = a.get("transcript_id")
            gid = a.get("gene_id")
            if not tid or not gid:
                continue

            s0 = int(start1) - 1
            e0 = int(end1)

            try:
                ph = int(phase)
            except Exception:
                ph = 0
            if ph not in (0, 1, 2):
                ph = 0

            if tid not in tx:
                tx[tid] = Transcript(
                    transcript_id=tid, gene_id=gid, chrom=chrom, strand=strand, cds=[]
                )
            tx[tid].cds.append(CdsSeg(start=s0, end=e0, phase=ph))

    out = list(tx.values())
    for t in out:
        t.build_blocks()
    out = [t for t in out if t.cds_len() >= 3]
    return out


def build_any_cds_mask(transcripts: List[Transcript]) -> IntervalMask:
    intervals: List[Tuple[int, int]] = []
    for t in transcripts:
        if t.blocks is None:
            t.build_blocks()
        for a, b, _ in (t.blocks or []):
            intervals.append((a, b))
    return IntervalMask(merge_intervals(intervals))


def find_all_atg_plusstrand(chrom_seq: str) -> List[int]:
    chrom_seq = chrom_seq.upper()
    hits: List[int] = []
    i = chrom_seq.find("ATG", 0)
    while i != -1:
        hits.append(i)
        i = chrom_seq.find("ATG", i + 1)
    return hits


def classify_noncoding_atg_sites(chrom_seq: str, any_cds_mask: IntervalMask) -> List[int]:
    atgs = find_all_atg_plusstrand(chrom_seq)
    out = []
    for p in atgs:
        if not any_cds_mask.window_fully_inside(p, p + 3):
            out.append(p)
    return out


def find_near(pos: int, arr: List[int], min_d: int, max_d: int) -> Optional[Tuple[int, int]]:
    if not arr:
        return None

    cand = []

    i_r = bisect_left(arr, pos + min_d)
    if i_r < len(arr):
        d = arr[i_r] - pos
        if min_d <= d <= max_d:
            cand.append((arr[i_r], d))

    i_l = bisect_right(arr, pos - min_d) - 1
    if i_l >= 0:
        d = pos - arr[i_l]
        if min_d <= d <= max_d:
            cand.append((arr[i_l], -d))

    if not cand:
        return None
    return min(cand, key=lambda x: abs(x[1]))


def find_far(pos: int, arr: List[int], min_d: int) -> Optional[Tuple[int, int]]:
    if not arr:
        return None

    cand = []

    i_r = bisect_left(arr, pos + min_d)
    if i_r < len(arr):
        cand.append((arr[i_r], arr[i_r] - pos))

    i_l = bisect_right(arr, pos - min_d) - 1
    if i_l >= 0:
        cand.append((arr[i_l], -(pos - arr[i_l])))

    if not cand:
        return None
    return min(cand, key=lambda x: abs(x[1]))


def _closest_by_abs_delta(anchor: int, positions: List[int]) -> Optional[Tuple[int, int]]:
    if not positions:
        return None
    best_p = min(positions, key=lambda p: abs(p - anchor))
    return best_p, best_p - anchor


def transcript_atg_sites(
    t: Transcript,
    chrom_seq: str,
) -> Tuple[Optional[int], List[int], List[int]]:
    """
    returns:
      start_genomic_pos (cds_offset 0), or None if start codon isn't ATG
      inframe_methionines_genomic (cds_offset%3==0, codon==ATG, offset>0)
      outframe_atg_motifs_genomic (cds_offset%3!=0, cds_seq[off:off+3]==ATG)
    """
    cds_seq = t.build_cds_seq(chrom_seq)
    if len(cds_seq) < 3 or cds_seq[:3] != "ATG":
        return None, [], []

    start_g = t.cds_offset_to_genomic(0)
    if start_g is None:
        return None, [], []

    inframe: List[int] = []
    outframe: List[int] = []

    L = len(cds_seq)
    for off in range(0, L - 2):
        if cds_seq[off : off + 3] != "ATG":
            continue
        g = t.cds_offset_to_genomic(off)
        if g is None:
            continue
        if off % 3 == 0:
            if off != 0:
                inframe.append(g)
        else:
            outframe.append(g)

    inframe.sort()
    outframe.sort()
    return start_g, inframe, outframe


def fmt_pair(p: Optional[Tuple[int, int]]) -> Tuple[str, str]:
    if p is None:
        return ".", "."
    return str(p[0]), str(p[1])


def process_chrom(
    chrom: str,
    gtf_path: str,
    genome_fa: str,
    out_path: str,
    min_near: int = MIN_NEAR,
    max_near: int = MAX_NEAR,
    min_far: int = MIN_FAR,
    n_sample: Optional[int] = None,
    random_seed: Optional[int] = None,
) -> None:
    """
    Simplified ATG label generation: 5 labels per transcript.
    
    For each sampled transcript:
    - label1: start codon position
    - label2: nearby noncoding ATG (2-5kb)
    - label3: far noncoding ATG (>100kb)
    - label4: same protein inframe methionine
    - label5: same protein outframe ATG
    """
    if random_seed is not None:
        random.seed(random_seed)
    
    fa = Fasta(genome_fa)
    if chrom not in fa:
        raise ValueError(f"{chrom} not found in fasta: {genome_fa}")
    chrom_seq = fa[chrom][:].seq.upper()

    transcripts = load_transcripts_from_gtf(gtf_path, chrom)
    any_cds_mask = build_any_cds_mask(transcripts)
    noncoding_atg = classify_noncoding_atg_sites(chrom_seq, any_cds_mask)
    noncoding_atg.sort()

    # precompute per-transcript sites (only those with true ATG start)
    tx_sites: List[Tuple[Transcript, int, List[int], List[int]]] = []
    for t in transcripts:
        start_g, inframe, outframe = transcript_atg_sites(t, chrom_seq)
        if start_g is None:
            continue
        tx_sites.append((t, start_g, inframe, outframe))

    print(f"[INFO] {chrom}: transcripts w/ ATG start = {len(tx_sites)}")
    print(f"[INFO] {chrom}: noncoding ATG sites (outside any CDS) = {len(noncoding_atg)}")

    # Sample transcripts if requested
    if n_sample is not None and n_sample < len(tx_sites):
        tx_sites = random.sample(tx_sites, n_sample)
        print(f"[INFO] {chrom}: sampled {n_sample} transcripts")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows_written = 0
    skipped = 0

    with out_path.open("w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(
            [
                "chrom",
                "transcript_id",
                "gene_id",
                "strand",
                "label1_start_pos",
                "label2_noncoding_near_pos",
                "label2_delta_bp",
                "label3_noncoding_far_pos",
                "label3_delta_bp",
                "label4_same_inframe_met_pos",
                "label4_delta_bp",
                "label5_same_outframe_atg_pos",
                "label5_delta_bp",
            ]
        )

        for ref_t, ref_start, ref_inframe, ref_outframe in tx_sites:
            anchor = ref_start

            # label2: noncoding near
            l2 = find_near(anchor, noncoding_atg, min_near, max_near)
            # label3: noncoding far
            l3 = find_far(anchor, noncoding_atg, min_far)
            # label4: same protein inframe
            l4 = _closest_by_abs_delta(anchor, ref_inframe)
            # label5: same protein outframe
            l5 = _closest_by_abs_delta(anchor, ref_outframe)

            # Skip if we're missing critical labels (optional - could keep all)
            # For now, we'll write all rows even if some labels are missing
            
            l2_pos, l2_delta = fmt_pair(l2)
            l3_pos, l3_delta = fmt_pair(l3)
            l4_pos, l4_delta = fmt_pair(l4)
            l5_pos, l5_delta = fmt_pair(l5)

            w.writerow(
                [
                    chrom,
                    ref_t.transcript_id,
                    ref_t.gene_id,
                    ref_t.strand,
                    anchor,
                    l2_pos,
                    l2_delta,
                    l3_pos,
                    l3_delta,
                    l4_pos,
                    l4_delta,
                    l5_pos,
                    l5_delta,
                ]
            )
            rows_written += 1

    print(f"[INFO] {chrom}: rows written = {rows_written}")


def main():
    ap = argparse.ArgumentParser(
        description="Simplified 5-way ATG label generation: start + noncoding near/far + inframe/outframe"
    )
    ap.add_argument("--chrom", required=True, help="e.g., chr1")
    ap.add_argument(
        "--gtf_dir",
        required=True,
        default="/data_processing/data/gtfs",
        help="directory containing chr1.gtf ... chrX.gtf",
    )
    ap.add_argument(
        "--genome",
        required=True,
        default="/data_processing/data/240-mammalian/hg38.ml.fa",
        help="genome fasta",
    )
    ap.add_argument(
        "--out",
        default="/data_processing/data/ATGs_simplified",
        help="output directory (tsv per chrom will be written here)",
    )
    ap.add_argument("--min_near", type=int, default=MIN_NEAR, help="min distance for near noncoding (default: 2kb)")
    ap.add_argument("--max_near", type=int, default=MAX_NEAR, help="max distance for near noncoding (default: 5kb)")
    ap.add_argument("--min_far", type=int, default=MIN_FAR, help="min distance for far noncoding (default: 100kb)")
    ap.add_argument(
        "--n_sample",
        type=int,
        default=None,
        help="randomly sample N transcripts per chromosome (default: use all)",
    )
    ap.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="random seed for reproducibility (default: 42)",
    )

    args = ap.parse_args()
    gtf_path = str(Path(args.gtf_dir) / f"{args.chrom}.gtf")
    out_path = str(Path(args.out) / f"{args.chrom}_atg_5way_labels.tsv")

    process_chrom(
        chrom=args.chrom,
        gtf_path=gtf_path,
        genome_fa=args.genome,
        out_path=out_path,
        min_near=args.min_near,
        max_near=args.max_near,
        min_far=args.min_far,
        n_sample=args.n_sample,
        random_seed=args.random_seed,
    )


if __name__ == "__main__":
    main()


# #!/usr/bin/env bash
# set -euo pipefail
# 
# SCRIPT="/home/mica/gamba/data_processing/make_ATG_data.py"

# # Generate 1000 examples per chromosome
# for i in $(seq 1 22); do
#   chrom="chr${i}"
#   echo "[RUN] ${chrom}"
#   python "$SCRIPT" --chrom "$chrom" --n_sample 10000
# done

# echo "[DONE] chr1–chr22"
