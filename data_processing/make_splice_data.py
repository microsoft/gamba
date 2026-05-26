#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from pyfaidx import Fasta


# ---------------------------------------------------------------------
# utils
# ---------------------------------------------------------------------

def parse_gtf_attrs(attr: str) -> Dict[str, str]:
    out = {}
    for p in [x.strip() for x in attr.split(";") if x.strip()]:
        if " " not in p:
            continue
        k, v = p.split(" ", 1)
        out[k] = v.strip().strip('"')
    return out


def merge_intervals(intervals: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if not intervals:
        return []
    intervals = sorted(intervals)
    out = []
    cs, ce = intervals[0]
    for s, e in intervals[1:]:
        if s <= ce:
            ce = max(ce, e)
        else:
            out.append((cs, ce))
            cs, ce = s, e
    out.append((cs, ce))
    return out


class IntervalMask:
    def __init__(self, intervals: List[Tuple[int, int]]):
        self.intervals = intervals
        self.starts = [s for s, _ in intervals]
        self.ends = [e for _, e in intervals]

    def contains_point(self, pos: int) -> bool:
        from bisect import bisect_right
        i = bisect_right(self.starts, pos) - 1
        return i >= 0 and pos < self.ends[i]


def find_all_motif_positions(seq: str, motif: str) -> List[int]:
    hits, i = [], seq.find(motif)
    while i != -1:
        hits.append(i)
        i = seq.find(motif, i + 1)
    return hits


def closest(anchor: int, xs: List[int]) -> Optional[Tuple[int, int]]:
    if not xs:
        return None
    p = min(xs, key=lambda x: abs(x - anchor))
    return p, p - anchor


def fmt(p: Optional[Tuple[int, int]]) -> Tuple[str, str]:
    return (".", ".") if p is None else (str(p[0]), str(p[1]))


def build_site_exclusion_mask(
    sites: List[int],
    motif_len: int,
    exclude_bp: int,
) -> IntervalMask:
    """
    builds a mask that excludes [pos-exclude_bp, pos+motif_len+exclude_bp)
    for each site.
    """
    intervals = []
    for p in sites:
        s = max(0, p - exclude_bp)
        e = p + motif_len + exclude_bp
        intervals.append((s, e))
    return IntervalMask(merge_intervals(intervals))


# ---------------------------------------------------------------------
# transcript
# ---------------------------------------------------------------------

@dataclass
class Transcript:
    transcript_id: str
    gene_id: str
    chrom: str
    strand: str
    exons: List[Tuple[int, int]]  # 0-based half-open

    def introns(self) -> List[Tuple[int, int]]:
        ex = sorted(self.exons)
        return [(ex[i][1], ex[i+1][0]) for i in range(len(ex) - 1) if ex[i][1] < ex[i+1][0]]

    def donor_sites_canonical(self, chrom_seq: str) -> List[int]:
        """
        true donor sites (genomic coordinate of intron start dinucleotide):
          + strand donor motif: GT at exon_end
          - strand donor motif: AC at intron start on +reference (reverse complement of GT)
        """
        if len(self.exons) < 2:
            return []
        ex = sorted(self.exons)
        out = []
        for i in range(len(ex) - 1):
            if self.strand == "+":
                pos = ex[i][1]  # exon end = intron start
                if chrom_seq[pos:pos + 2] == "GT":
                    out.append(pos)
            else:
                pos = ex[i + 1][0]  # intron start on reference for -strand
                if chrom_seq[pos:pos + 2] == "AC":
                    out.append(pos)
        return out

    def acceptor_sites_canonical(self, chrom_seq: str) -> List[int]:
        """
        true acceptor sites (genomic coordinate of acceptor dinucleotide):
          + strand acceptor motif: AG at exon_start (of downstream exon)
          - strand acceptor motif: CT on +reference (reverse complement of AG)
        """
        if len(self.exons) < 2:
            return []
        ex = sorted(self.exons)
        out = []
        for i in range(1, len(ex)):
            pos = ex[i][0]  # downstream exon start
            if self.strand == "+":
                if chrom_seq[pos:pos + 2] == "AG":
                    out.append(pos)
            else:
                if chrom_seq[pos:pos + 2] == "CT":
                    out.append(pos)
        return out


# ---------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------

def load_transcripts(gtf: str, chrom: str) -> List[Transcript]:
    tx: Dict[str, Transcript] = {}
    with open(gtf) as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            c, _, feat, s1, e1, _, strand, _, attrs = line.rstrip().split("\t")
            if c != chrom or feat != "exon":
                continue
            if strand not in "+-":
                continue
            if 'tag "MANE_Select"' not in attrs:
                continue

            a = parse_gtf_attrs(attrs)
            tid, gid = a.get("transcript_id"), a.get("gene_id")
            if not tid or not gid:
                continue

            s0, e0 = int(s1) - 1, int(e1)  # 0-based half-open
            if tid not in tx:
                tx[tid] = Transcript(tid, gid, chrom, strand, [])
            tx[tid].exons.append((s0, e0))

    return [t for t in tx.values() if len(t.exons) >= 2]


# ---------------------------------------------------------------------
# 5-way sampling logic
# ---------------------------------------------------------------------

@dataclass
class Example5Way:
    chrom: str
    ref_transcript_id: str
    ref_gene_id: str
    ref_strand: str

    label1_true_pos: int

    label2_near_decoy_pos: int
    label3_far_decoy_pos: int
    label4_other_same_gene_pos: int

    label5_other_diff_gene_pos: int
    label5_other_diff_gene_id: str
    label5_other_diff_transcript_id: str
    label5_other_diff_strand: str


def sample_5way_for_site(
    *,
    anchor: int,
    same_gene_sites: List[int],
    other_gene_sites: List[int],
    decoy_candidates: List[int],
    exclude_mask: IntervalMask,
    window: int,
    near_max: int,
    far_min: int,
) -> Optional[Tuple[int, int, int, int]]:
    """
    returns (l2, l3, l4, l5_pos) for a given anchor, or None if cannot form full 5-way.
    NOTE: does not decide which gene/transcript l5 belongs to (caller does).
    """
    # -------- decoys (same motif, not splice) --------
    decoys = []
    for p in decoy_candidates:
        if p == anchor:
            continue
        if exclude_mask.contains_point(p):
            continue
        if abs(p - anchor) > window:
            continue
        decoys.append(p)

    if not decoys:
        return None

    near_decoys = [p for p in decoys if 1 <= abs(p - anchor) <= near_max]
    far_decoys = [p for p in decoys if abs(p - anchor) >= far_min]

    l2 = closest(anchor, near_decoys)
    l3 = closest(anchor, far_decoys)
    if l2 is None or l3 is None:
        return None

    l2_pos = l2[0]
    l3_pos = l3[0]

    # -------- other functional same gene --------
    same_gene_other = [p for p in same_gene_sites if p != anchor]
    l4 = closest(anchor, same_gene_other)
    if l4 is None:
        return None
    l4_pos = l4[0]
    target_dist = abs(l4_pos - anchor)

    # -------- other functional different gene --------
    if not other_gene_sites:
        return None
    l5_pos = min(other_gene_sites, key=lambda p: abs(abs(p - anchor) - target_dist))

    return l2_pos, l3_pos, l4_pos, l5_pos

def process_chrom_5way(
    *,
    chrom: str,
    gtf: str,
    genome: str,
    out_tsv: str,
    site_type: str,  # "donor" | "acceptor"
    max_examples: int = 1000,
    seed: int = 42,
    window: int = 2000,
    near_max: int = 100,
    far_min: int = 500,
    exclude_bp: int = 50,
    l5_max_dist: Optional[int] = None,  # <- new: cap label5 distance from anchor (None = no cap)
):
    """
    writes 5-way examples for either donor or acceptor, per chromosome.

    labels:
      1 true site (anchor)
      2 near decoy same motif, not annotated splice (within near_max)
      3 far decoy same motif, not annotated splice (>= far_min)
      4 other true site same gene
      5 other true site different gene (distance-matched to label4)

    IMPORTANT CHANGE:
      - label5 metadata is now sourced from the transcript that generated the site,
        not from a "representative transcript per gene".
      - optional l5_max_dist caps how far label5 can be from the anchor.
    """
    import random

    rng = random.Random(seed)

    fa = Fasta(genome)
    chrom_seq = fa[chrom][:].seq.upper()

    transcripts = load_transcripts(gtf, chrom)

    # motif in +reference coordinates, depends on site_type + transcript strand
    def motif_for(site_type_: str, strand: str) -> str:
        if site_type_ == "donor":
            return "GT" if strand == "+" else "AC"
        if site_type_ == "acceptor":
            return "AG" if strand == "+" else "CT"
        raise ValueError(site_type_)

    motif_len = 2

    # collect true sites and build per-gene pools (across transcripts)
    per_gene_sites: Dict[str, List[int]] = {}
    all_true_sites: List[int] = []
    tx_anchors: List[Tuple[Transcript, List[int]]] = []

    for t in transcripts:
        if site_type == "donor":
            sites = t.donor_sites_canonical(chrom_seq)
        elif site_type == "acceptor":
            sites = t.acceptor_sites_canonical(chrom_seq)
        else:
            raise ValueError(f"unknown site_type: {site_type}")

        if not sites:
            continue

        tx_anchors.append((t, sites))
        per_gene_sites.setdefault(t.gene_id, []).extend(sites)
        all_true_sites.extend(sites)

    outp = Path(out_tsv)
    outp.parent.mkdir(parents=True, exist_ok=True)

    header = [
        "chrom", "ref_transcript_id", "ref_gene_id", "ref_strand",
        "label1_true_pos",
        "label2_near_decoy_pos", "label2_delta_bp",
        "label3_far_decoy_pos", "label3_delta_bp",
        "label4_other_same_gene_pos", "label4_delta_bp",
        "label5_other_diff_gene_pos", "label5_delta_bp",
        "label5_other_diff_gene_id",
        "label5_other_diff_transcript_id",
        "label5_other_diff_strand",
    ]

    # write header even if empty
    if not tx_anchors:
        with outp.open("w", newline="") as f:
            csv.writer(f, delimiter="\t").writerow(header)
        return

    # dedupe pools
    for gid in list(per_gene_sites.keys()):
        per_gene_sites[gid] = sorted(set(per_gene_sites[gid]))
    all_true_sites = sorted(set(all_true_sites))

    exclude_mask = build_site_exclusion_mask(all_true_sites, motif_len=motif_len, exclude_bp=exclude_bp)

    # NEW: build a position -> (gene_id, transcript_id, strand) map from the actual transcripts
    # If multiple transcripts share the same genomic position, first one wins (rare; acceptable).
    pos_meta: Dict[int, Tuple[str, str, str]] = {}
    for t2, sites2 in tx_anchors:
        for p in sites2:
            pos_meta.setdefault(p, (t2.gene_id, t2.transcript_id, t2.strand))

    rng.shuffle(tx_anchors)

    written = 0
    with outp.open("w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(header)

        for t, anchors in tx_anchors:
            anchors = anchors[:]
            rng.shuffle(anchors)

            motif = motif_for(site_type, t.strand)
            same_gene_sites = per_gene_sites.get(t.gene_id, [])

            # other-gene pool = all true sites not in this gene
            other_gene_sites_all = [p for p in all_true_sites if pos_meta.get(p, (None, None, None))[0] != t.gene_id]
            if not other_gene_sites_all:
                continue

            for anchor in anchors:
                if written >= max_examples:
                    return

                # decoy candidates: same motif in local window, excluded if near ANY true splice site
                lo = max(0, anchor - window)
                hi = min(len(chrom_seq), anchor + window + motif_len)
                sub = chrom_seq[lo:hi]
                decoy_candidates = [lo + i for i in find_all_motif_positions(sub, motif)]

                # optionally cap label5 candidates by distance from anchor
                if l5_max_dist is None:
                    other_gene_sites = other_gene_sites_all
                else:
                    other_gene_sites = [p for p in other_gene_sites_all if abs(p - anchor) <= l5_max_dist]
                    if not other_gene_sites:
                        continue

                picked = sample_5way_for_site(
                    anchor=anchor,
                    same_gene_sites=same_gene_sites,
                    other_gene_sites=other_gene_sites,
                    decoy_candidates=decoy_candidates,
                    exclude_mask=exclude_mask,
                    window=window,
                    near_max=near_max,
                    far_min=far_min,
                )
                if picked is None:
                    continue

                l2_pos, l3_pos, l4_pos, l5_pos = picked

                # label5 metadata: use the actual strand/transcript that produced l5_pos
                meta = pos_meta.get(l5_pos, None)
                if meta is None:
                    # shouldn't happen, but be safe
                    continue
                l5_gene_id, l5_transcript_id, l5_strand = meta

                w.writerow([
                    chrom, t.transcript_id, t.gene_id, t.strand,
                    anchor,
                    l2_pos, l2_pos - anchor,
                    l3_pos, l3_pos - anchor,
                    l4_pos, l4_pos - anchor,
                    l5_pos, l5_pos - anchor,
                    l5_gene_id,
                    l5_transcript_id,
                    l5_strand,
                ])
                written += 1



# ---------------------------------------------------------------------
# cli
# ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="5-way splice task label generation (donor or acceptor), per chromosome (MANE_Select)"
    )
    ap.add_argument("--chrom", required=True, help="e.g., chr1")
    ap.add_argument(
        "--site_type",
        required=True,
        choices=["donor", "acceptor"],
        help="generate donor 5-way or acceptor 5-way",
    )
    ap.add_argument(
        "--gtf_dir",
        default="/home/mica/gamba/data_processing/data/gtfs",
        help="directory containing chr1.gtf ... chrX.gtf",
    )
    ap.add_argument(
        "--genome",
        default="/home/mica/gamba/data_processing/data/240-mammalian/hg38.ml.fa",
        help="genome fasta",
    )
    ap.add_argument(
        "--out",
        default="/home/mica/gamba/data_processing/data/splice_sites",
        help="output directory (tsv per chrom will be written here)",
    )
    ap.add_argument("--l5_max_dist", type=int, default=None)
    ap.add_argument("--max_examples", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)

    # hardness knobs
    ap.add_argument("--window", type=int, default=2000)
    ap.add_argument("--near_max", type=int, default=100)
    ap.add_argument("--far_min", type=int, default=500)
    ap.add_argument("--exclude_bp", type=int, default=50)

    args = ap.parse_args()

    gtf_path = str(Path(args.gtf_dir) / f"{args.chrom}.gtf")
    out_path = str(Path(args.out) / f"{args.chrom}_{args.site_type}_5way.tsv")

    process_chrom_5way(
        chrom=args.chrom,
        gtf=gtf_path,
        genome=args.genome,
        out_tsv=out_path,
        site_type=args.site_type,
        max_examples=args.max_examples,
        seed=args.seed,
        window=args.window,
        near_max=args.near_max,
        far_min=args.far_min,
        exclude_bp=args.exclude_bp,
        l5_max_dist=args.l5_max_dist
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env bash
# set -euo pipefail

# SCRIPT="/home/mica/gamba/data_processing/make_splice_data.py"

# for i in $(seq 1 22); do
#   chrom="chr${i}"
#   echo "[RUN] ${chrom} donor"
#   python "$SCRIPT" --chrom "$chrom" --site_type donor --max_examples 1000

#   echo "[RUN] ${chrom} acceptor"
#   python "$SCRIPT" --chrom "$chrom" --site_type acceptor --max_examples 1000
# done

# echo "[DONE] chr1–chr22 donor+acceptor"
