import os
import pandas as pd
import pytest
from pyfaidx import Fasta

#!/usr/bin/env python3

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from bisect import bisect_left, bisect_right

from pyfaidx import Fasta


# distance params (same as before)
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
    # derived blocks in transcript order after phase trim:
    # (g_start, g_end, cds_offset_start)
    blocks: Optional[List[Tuple[int, int, int]]] = None

    def build_blocks(self) -> None:
        if not self.cds:
            self.blocks = []
            return

        # order CDS segments in transcript direction
        if self.strand == "+":
            segs = sorted(self.cds, key=lambda x: (x.start, x.end))
        else:
            segs = sorted(self.cds, key=lambda x: (x.start, x.end), reverse=True)

        # apply gtf phase trim to FIRST segment only
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

    def genomic_to_cds_offset(self, gpos: int) -> Optional[int]:
        if self.blocks is None:
            self.build_blocks()
        if not self.blocks:
            return None

        if self.strand == "+":
            for a, b, off0 in self.blocks:
                if a <= gpos < b:
                    return off0 + (gpos - a)
            return None
        else:
            for a, b, off0 in self.blocks:
                if a <= gpos < b:
                    return off0 + ((b - 1) - gpos)
            return None

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

            # --- canonical filter (MANE_Select) ---
            attrs_str = attrs
            if 'tag "MANE_Select"' not in attrs_str:
                continue

            a = parse_gtf_attrs(attrs)
            tid = a.get("transcript_id")
            gid = a.get("gene_id")
            if not tid or not gid:
                continue

            # gtf is 1-based inclusive. convert to 0-based half-open
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

    # scan cds_seq for "ATG" motifs; classify by off % 3
    # cap scanning length to avoid pathological huge transcripts
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

    # keep sorted (genomic coords)
    inframe.sort()
    outframe.sort()
    return start_g, inframe, outframe


def pick_diff_protein_transcript(
    anchor: int,
    ref_gene_id: str,
    candidates: List[Tuple[Transcript, int, List[int], List[int]]],
) -> Optional[Tuple[Transcript, int, List[int], List[int]]]:
    """
    pick the transcript (different gene_id) whose START is closest to anchor.
    candidates items are (t, start_g, inframe_list, outframe_list)
    """
    best = None
    best_d = None
    for t, start_g, inframe, outframe in candidates:
        if t.gene_id == ref_gene_id:
            continue
        d = abs(start_g - anchor)
        if best is None or d < best_d:
            best = (t, start_g, inframe, outframe)
            best_d = d
    return best


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
    max_ref_tx: Optional[int] = None,
) -> None:
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

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(
            [
                "chrom",
                "ref_transcript_id",
                "ref_gene_id",
                "ref_strand",
                "label1_start_pos",
                "label2_noncoding_near_pos",
                "label2_delta_bp",
                "label3_noncoding_far_pos",
                "label3_delta_bp",
                "label4_same_inframe_met_pos",
                "label4_delta_bp",
                "label5_same_outframe_atg_pos",
                "label5_delta_bp",
                "diff_transcript_id",
                "diff_gene_id",
                "diff_strand",
                "label6_diff_start_pos",
                "label6_delta_bp",
                "label7_diff_inframe_met_pos",
                "label7_delta_bp",
                "label8_diff_outframe_atg_pos",
                "label8_delta_bp",
            ]
        )

        # iterate ref transcripts (optionally cap)
        ref_iter = tx_sites if max_ref_tx is None else tx_sites[:max_ref_tx]
        for ref_t, ref_start, ref_inframe, ref_outframe in ref_iter:
            anchor = ref_start

            # label2/3: noncoding near/far relative to anchor
            l2 = find_near(anchor, noncoding_atg, min_near, max_near)
            l3 = find_far(anchor, noncoding_atg, min_far)

            # label4/5: same protein inframe/outframe relative to anchor (choose closest)
            l4 = _closest_by_abs_delta(anchor, ref_inframe)
            l5 = _closest_by_abs_delta(anchor, ref_outframe)

            # pick a diff protein transcript and its labels (relative to anchor)
            diff = pick_diff_protein_transcript(anchor, ref_t.gene_id, tx_sites)
            if diff is None:
                # no diff gene found (rare if gtf small), still write row with missing diff fields
                diff_t = None
                l6 = l7 = l8 = None
            else:
                diff_t, diff_start, diff_inframe, diff_outframe = diff
                l6 = (diff_start, diff_start - anchor)
                l7 = _closest_by_abs_delta(anchor, diff_inframe)
                l8 = _closest_by_abs_delta(anchor, diff_outframe)

            l2_pos, l2_delta = fmt_pair(l2)
            l3_pos, l3_delta = fmt_pair(l3)
            l4_pos, l4_delta = fmt_pair(l4)
            l5_pos, l5_delta = fmt_pair(l5)

            if diff is None:
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
                        ".",
                        ".",
                        ".",
                        ".",
                        ".",
                        ".",
                        ".",
                        ".",
                        ".",
                    ]
                )
            else:
                diff_t, diff_start, diff_inframe, diff_outframe = diff
                l6_pos, l6_delta = fmt_pair(l6)
                l7_pos, l7_delta = fmt_pair(l7)
                l8_pos, l8_delta = fmt_pair(l8)
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
                        diff_t.transcript_id,
                        diff_t.gene_id,
                        diff_t.strand,
                        l6_pos,
                        l6_delta,
                        l7_pos,
                        l7_delta,
                        l8_pos,
                        l8_delta,
                    ]
                )


GTF_DIR = "/home/mica/gamba/data_processing/data/gtfs"
GENOME_FA = "/home/mica/gamba/data_processing/data/240-mammalian/hg38.ml.fa"
CHROM = "chr1"


def _load_chr_seq():
    fa = Fasta(GENOME_FA)
    return fa[CHROM][:].seq.upper()


@pytest.fixture(scope="session")
def chr_seq():
    return _load_chr_seq()


@pytest.fixture(scope="session")
def transcripts():
    gtf_path = os.path.join(GTF_DIR, f"{CHROM}.gtf")
    txs = load_transcripts_from_gtf(gtf_path, CHROM)
    assert len(txs) > 0
    return txs


@pytest.fixture(scope="session")
def any_cds_mask(transcripts):
    return build_any_cds_mask(transcripts)


@pytest.fixture(scope="session")
def tx_map(transcripts, chr_seq):
    """
    map transcript_id -> (Transcript, start_g, inframe_list, outframe_list)
    only for transcripts with ATG start
    """
    out = {}
    for t in transcripts:
        start_g, inframe, outframe = transcript_atg_sites(t, chr_seq)
        if start_g is None:
            continue
        out[t.transcript_id] = (t, start_g, inframe, outframe)
    assert len(out) > 0
    return out


def _as_int(x):
    if x == "." or pd.isna(x):
        return None
    return int(x)


def test_script_generates_rows_and_labels_are_valid(tmp_path, chr_seq, any_cds_mask, tx_map):
    out_tsv = tmp_path / "chr1_atg_8way_labels.tsv"
    gtf_path = os.path.join(GTF_DIR, f"{CHROM}.gtf")

    # cap for test speed
    process_chrom(
        chrom=CHROM,
        gtf_path=gtf_path,
        genome_fa=GENOME_FA,
        out_path=str(out_tsv),
        max_ref_tx=300,
    )

    df = pd.read_csv(out_tsv, sep="\t")
    assert len(df) > 0

    # build noncoding atg set for membership checks
    noncoding_atg = set(classify_noncoding_atg_sites(chr_seq, any_cds_mask))

    for _, r in df.iterrows():
        ref_tid = r["ref_transcript_id"]
        assert ref_tid in tx_map
        ref_t, ref_start, ref_inframe, ref_outframe = tx_map[ref_tid]
        anchor = int(r["label1_start_pos"])
        assert anchor == ref_start

        # label1 must be ATG start on strand-aware sequence
        if ref_t.strand == "+":
            assert chr_seq[anchor:anchor+3] == "ATG"
        else:
            trip = chr_seq[anchor-2:anchor+1]
            assert revcomp(trip) == "ATG"

        # label2/3: if present, must be noncoding (outside any CDS window) and distance constraints
        l2 = _as_int(r["label2_noncoding_near_pos"])
        if l2 is not None:
            assert l2 in noncoding_atg
            assert 2000 <= abs(l2 - anchor) <= 5000

        l3 = _as_int(r["label3_noncoding_far_pos"])
        if l3 is not None:
            assert l3 in noncoding_atg
            assert abs(l3 - anchor) >= 100000

        # label4: same transcript inframe methionine (internal)
        l4 = _as_int(r["label4_same_inframe_met_pos"])
        if l4 is not None:
            assert l4 in set(ref_inframe)
            assert l4 != anchor

        # label5: same transcript out-of-frame atg motif
        l5 = _as_int(r["label5_same_outframe_atg_pos"])
        if l5 is not None:
            assert l5 in set(ref_outframe)

        # diff transcript: if present, must be different gene_id
        diff_tid = r["diff_transcript_id"]
        if diff_tid != ".":
            assert diff_tid in tx_map
            diff_t, diff_start, diff_inframe, diff_outframe = tx_map[diff_tid]
            assert diff_t.gene_id != ref_t.gene_id

            l6 = _as_int(r["label6_diff_start_pos"])
            if l6 is not None:
                assert l6 == diff_start

            l7 = _as_int(r["label7_diff_inframe_met_pos"])
            if l7 is not None:
                assert l7 in set(diff_inframe)

            l8 = _as_int(r["label8_diff_outframe_atg_pos"])
            if l8 is not None:
                assert l8 in set(diff_outframe)


def test_mane_one_transcript_per_gene(transcripts):
    # transcripts fixture is after MANE filter if you changed load_transcripts_from_gtf
    from collections import Counter
    c = Counter(t.gene_id for t in transcripts)
    assert max(c.values()) == 1

def test_missingness_not_insane(tmp_path):
    import pandas as pd
    out_tsv = tmp_path / "chr1_atg_8way_labels.tsv"
    gtf_path = os.path.join(GTF_DIR, f"{CHROM}.gtf")
    process_chrom(
        chrom=CHROM,
        gtf_path=gtf_path,
        genome_fa=GENOME_FA,
        out_path=str(out_tsv),
        max_ref_tx=2000,  # more stable stats
    )
    df = pd.read_csv(out_tsv, sep="\t")
    assert len(df) > 100  # sanity: you got real scale output

    def miss_rate(col):
        return (df[col].astype(str) == ".").mean()

    # these thresholds are deliberately loose; you tighten after you see real numbers
    assert miss_rate("label2_noncoding_near_pos") < 0.50
    assert miss_rate("label3_noncoding_far_pos") < 0.05
    assert miss_rate("label4_same_inframe_met_pos") < 0.80
    assert miss_rate("label5_same_outframe_atg_pos") < 0.50
    assert miss_rate("diff_transcript_id") < 0.05
