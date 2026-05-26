#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Iterable
from bisect import bisect_left, bisect_right

# -----------------------
# small helpers
# -----------------------

_COMP = str.maketrans("ACGTNacgtn", "TGCANtgcan")

def revcomp(s: str) -> str:
    return s.translate(_COMP)[::-1]

def parse_gtf_attrs(attr: str) -> Dict[str, str]:
    # minimal gtf attribute parser: key "value";
    out: Dict[str, str] = {}
    parts = [p.strip() for p in attr.strip().split(";") if p.strip()]
    for p in parts:
        if " " not in p:
            continue
        k, v = p.split(" ", 1)
        v = v.strip().strip('"')
        out[k] = v
    return out

# -----------------------
# intervals
# -----------------------

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
    """
    O(log N) membership for half-open intervals [start,end).
    """
    def __init__(self, intervals: List[Tuple[int, int]]):
        self.intervals = intervals
        self.starts = [s for s, _ in intervals]
        self.ends = [e for _, e in intervals]

    def contains(self, pos: int) -> bool:
        idx = bisect_right(self.starts, pos) - 1
        if idx < 0:
            return False
        return pos < self.ends[idx]

    def window_fully_inside(self, start: int, end: int) -> bool:
        # require [start,end) fully inside some interval
        idx = bisect_right(self.starts, start) - 1
        if idx < 0:
            return False
        return end <= self.ends[idx]

# -----------------------
# transcript model
# -----------------------

@dataclass(frozen=True)
class CdsSeg:
    start: int   # 0-based, half-open
    end: int     # 0-based, half-open
    phase: int   # 0,1,2 per gtf

@dataclass
class Transcript:
    transcript_id: str
    gene_id: str
    chrom: str
    strand: str          # '+' or '-'
    cds: List[CdsSeg]    # raw cds segments (untrimmed)
    # derived:
    blocks: List[Tuple[int,int,int]] = None
    # blocks: list of (g_start, g_end, cds_offset_start) in transcript order,
    # after applying phase trim to the first block

    def build_blocks(self) -> None:
        """
        Build phase-trimmed blocks in transcript order with cumulative cds offsets.
        """
        if not self.cds:
            self.blocks = []
            return

        # order cds in transcript order
        if self.strand == "+":
            segs = sorted(self.cds, key=lambda x: (x.start, x.end))
        else:
            segs = sorted(self.cds, key=lambda x: (x.start, x.end), reverse=True)

        # apply phase trim to FIRST segment only (gtf convention)
        first = segs[0]
        phase = int(first.phase)
        if phase not in (0,1,2):
            phase = 0

        trimmed: List[Tuple[int,int]] = []
        for i, s in enumerate(segs):
            a, b = s.start, s.end
            if i == 0 and phase:
                if self.strand == "+":
                    a = min(b, a + phase)
                else:
                    b = max(a, b - phase)
            if a < b:
                trimmed.append((a,b))

        blocks: List[Tuple[int,int,int]] = []
        off = 0
        for (a,b) in trimmed:
            blocks.append((a,b,off))
            off += (b - a)
        self.blocks = blocks

    def cds_len(self) -> int:
        if self.blocks is None:
            self.build_blocks()
        return 0 if not self.blocks else (self.blocks[-1][2] + (self.blocks[-1][1] - self.blocks[-1][0]))

    def genomic_to_cds_offset(self, gpos: int) -> Optional[int]:
        """
        Return cds offset for genomic base gpos (0-based), or None if not in CDS.
        Offset increases along transcript 5'->3' coding direction.
        """
        if self.blocks is None:
            self.build_blocks()
        if not self.blocks:
            return None

        if self.strand == "+":
            for a,b,off0 in self.blocks:
                if a <= gpos < b:
                    return off0 + (gpos - a)
            return None
        else:
            for a,b,off0 in self.blocks:
                if a <= gpos < b:
                    # transcript direction runs from (b-1) down to a
                    return off0 + ((b - 1) - gpos)
            return None

    def cds_offset_to_genomic(self, off: int) -> Optional[int]:
        """
        Map cds offset (0-based) to genomic coordinate of that base.
        """
        if self.blocks is None:
            self.build_blocks()
        if off < 0:
            return None
        for a,b,off0 in self.blocks:
            L = b - a
            if off0 <= off < off0 + L:
                d = off - off0
                if self.strand == "+":
                    return a + d
                else:
                    return (b - 1) - d
        return None

    def build_cds_seq(self, chrom_seq: str) -> str:
        """
        Construct phase-trimmed spliced CDS sequence on the coding strand (5'->3').
        """
        if self.blocks is None:
            self.build_blocks()
        parts = []
        for a,b,_ in self.blocks:
            parts.append(chrom_seq[a:b])
        s = "".join(parts)
        if self.strand == "-":
            s = revcomp(s)
        return s.upper()

    def codon_at_cds_offset(self, cds_seq: str, off: int) -> Optional[str]:
        if off < 0 or off + 3 > len(cds_seq):
            return None
        return cds_seq[off:off+3]

# -----------------------
# gtf loading (CDS only)
# -----------------------

def load_transcripts_from_gtf(gtf_path: str, chrom: str) -> List[Transcript]:
    """
    Load protein-coding transcript CDS structure from a per-chrom gtf.
    Keeps only rows with feature == 'CDS' and requiring transcript_id + gene_id.
    """
    tx: Dict[str, Transcript] = {}

    with open(gtf_path) as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9:
                continue
            seqname, source, feature, start1, end1, score, strand, phase, attrs = fields
            if seqname != chrom:
                continue
            if feature != "CDS":
                continue
            if strand not in ("+","-"):
                continue

            a = parse_gtf_attrs(attrs)
            tid = a.get("transcript_id")
            gid = a.get("gene_id")
            if not tid or not gid:
                continue

            # gtf is 1-based inclusive. convert to 0-based half-open
            s0 = int(start1) - 1
            e0 = int(end1)

            ph = 0
            try:
                ph = int(phase)
            except Exception:
                ph = 0

            if tid not in tx:
                tx[tid] = Transcript(transcript_id=tid, gene_id=gid, chrom=chrom, strand=strand, cds=[])
            tx[tid].cds.append(CdsSeg(start=s0, end=e0, phase=ph))

    out = list(tx.values())
    for t in out:
        t.build_blocks()
    # filter transcripts with non-empty cds after trimming
    out = [t for t in out if t.cds_len() >= 3]
    return out

def build_any_cds_mask(transcripts: List[Transcript]) -> IntervalMask:
    """
    Merge all CDS blocks (after phase trim) across transcripts into an IntervalMask (per-base CDS coverage).
    """
    intervals: List[Tuple[int,int]] = []
    for t in transcripts:
        if t.blocks is None:
            t.build_blocks()
        for a,b,_ in (t.blocks or []):
            intervals.append((a,b))
    merged = merge_intervals(intervals)
    return IntervalMask(merged)

# -----------------------
# genome-wide ATG scanning (plus-strand motif)
# -----------------------

def find_all_atg_plusstrand(chrom_seq: str) -> List[int]:
    chrom_seq = chrom_seq.upper()
    hits: List[int] = []
    i = chrom_seq.find("ATG", 0)
    while i != -1:
        hits.append(i)
        i = chrom_seq.find("ATG", i + 1)
    return hits

def classify_noncoding_atg_sites(chrom_seq: str, any_cds_mask: IntervalMask) -> List[int]:
    """
    "noncoding ATG" here means: plus-strand 'ATG' where [pos,pos+3) is NOT fully inside any CDS interval.
    (this is about being outside CDS, not strand-aware translation)
    """
    atgs = find_all_atg_plusstrand(chrom_seq)
    out = []
    for p in atgs:
        if not any_cds_mask.window_fully_inside(p, p+3):
            out.append(p)
    return out

import os
import pytest
from pyfaidx import Fasta

# paths you gave / implied
GTF_DIR = "/home/mica/gamba/data_processing/data/gtfs"
GENOME_FA = "/home/mica/gamba/data_processing/data/240-mammalian/hg38.ml.fa"
CHROM = "chr1"

# keep tests from taking forever
MAX_TX_SCAN = 5000
MAX_CDS_SCAN = 200_000  # scan at most this many cds bases per transcript for motif discovery


def _load_chr_seq():
    fa = Fasta(GENOME_FA)
    assert CHROM in fa
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


def _find_tx_with_start_atg(transcripts, chr_seq):
    for t in transcripts[:MAX_TX_SCAN]:
        cds_seq = t.build_cds_seq(chr_seq)
        if len(cds_seq) >= 3 and cds_seq[:3] == "ATG":
            return t, cds_seq
    return None, None

def _find_internal_inframe_atg(t, cds_seq):
    # return cds_offset if found, else None
    limit = min(len(cds_seq) - 2, MAX_CDS_SCAN)
    for off in range(3, limit, 3):
        if cds_seq[off:off+3] == "ATG":
            return off
    return None

def _find_outofframe_atg(t, cds_seq):
    # return cds_offset if found, else None
    limit = min(len(cds_seq) - 2, MAX_CDS_SCAN)
    for off in range(1, limit):
        if (off % 3) != 0 and cds_seq[off:off+3] == "ATG":
            return off
    return None


def test_true_start_codon_is_inframe_and_strandaware(transcripts, chr_seq):
    t, cds_seq = _find_tx_with_start_atg(transcripts, chr_seq)
    if t is None:
        pytest.skip("could not find a transcript in chr1 with cds_seq starting ATG (within scan limit)")

    # start codon in cds space
    assert cds_seq[:3] == "ATG"

    # genomic position of cds_offset 0
    g0 = t.cds_offset_to_genomic(0)
    assert g0 is not None

    if t.strand == "+":
        assert chr_seq[g0:g0+3] == "ATG"
    else:
        # on plus strand, the 3bp in increasing coords should revcomp to ATG
        # since g0 is the first translated base in transcript direction, for '-' codon spans g0,g0-1,g0-2
        assert g0 - 2 >= 0
        trip = chr_seq[g0-2:g0+1]  # length 3
        assert revcomp(trip) == "ATG"


def test_internal_inframe_methionine_is_multiple_of_3_not_start(transcripts, chr_seq):
    t, cds_seq = _find_tx_with_start_atg(transcripts, chr_seq)
    if t is None:
        pytest.skip("no start-ATG transcript found in chr1 scan window")

    off = _find_internal_inframe_atg(t, cds_seq)
    if off is None:
        pytest.skip("found start-ATG transcript but no internal in-frame ATG within scan window")

    assert off % 3 == 0
    assert off != 0
    assert cds_seq[off:off+3] == "ATG"

    g = t.cds_offset_to_genomic(off)
    assert g is not None

    # sanity: mapping back lands on the same offset
    assert t.genomic_to_cds_offset(g) == off


def test_out_of_frame_atg_is_not_multiple_of_3(transcripts, chr_seq):
    t, cds_seq = _find_tx_with_start_atg(transcripts, chr_seq)
    if t is None:
        pytest.skip("no start-ATG transcript found in chr1 scan window")

    off = _find_outofframe_atg(t, cds_seq)
    if off is None:
        pytest.skip("no out-of-frame ATG motif in this transcript within scan window")

    assert off % 3 != 0
    assert cds_seq[off:off+3] == "ATG"

    g = t.cds_offset_to_genomic(off)
    assert g is not None
    assert t.genomic_to_cds_offset(g) == off


def test_near_and_far_noncoding_atg_sites_exist(chr_seq, transcripts, any_cds_mask):
    """
    Uses plus-strand ATG motifs outside ANY CDS, then checks distance constraints
    from a real start codon anchor.
    """
    t, cds_seq = _find_tx_with_start_atg(transcripts, chr_seq)
    if t is None:
        pytest.skip("no start-ATG transcript found in chr1 scan window")

    anchor = t.cds_offset_to_genomic(0)
    assert anchor is not None

    noncoding_atg = classify_noncoding_atg_sites(chr_seq, any_cds_mask)
    if not noncoding_atg:
        pytest.skip("no noncoding ATG sites found (unexpected)")

    # find any within 2-5kb and any >=100kb
    near = [p for p in noncoding_atg if 2000 <= abs(p - anchor) <= 5000]
    far = [p for p in noncoding_atg if abs(p - anchor) >= 100000]

    if not near:
        pytest.skip("no near noncoding ATG within 2-5kb of anchor in chr1")
    if not far:
        pytest.skip("no far noncoding ATG >=100kb from anchor in chr1")

    # verify they are truly outside any CDS (3bp window not fully inside)
    p_near = near[0]
    p_far = far[0]
    assert not any_cds_mask.window_fully_inside(p_near, p_near + 3)
    assert not any_cds_mask.window_fully_inside(p_far, p_far + 3)
