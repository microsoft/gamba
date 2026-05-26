#!/usr/bin/env python3
import sys
from pathlib import Path

import pandas as pd
from pyfaidx import Fasta

GENOME = "/home/mica/gamba/data_processing/data/240-mammalian/hg38.ml.fa"
fa = Fasta(GENOME)

TSV_DIR = Path(sys.argv[1])
EXPECTED_PER_CHROM = 1000

# set this if you used different output naming
# new generator writes: {chrom}_{site_type}_5way.tsv
DONOR_SUFFIX = "donor_5way.tsv"
ACCEPTOR_SUFFIX = "acceptor_5way.tsv"

bad = False
coverage = {"donor": {}, "acceptor": {}}


def _motif_for(site_type: str, strand: str) -> str:
    if site_type == "donor":
        return "GT" if strand == "+" else "AC"
    if site_type == "acceptor":
        return "AG" if strand == "+" else "CT"
    raise ValueError(site_type)


def _check_anchor_motif(seq, pos: int, motif: str) -> bool:
    """
    exact check at pos..pos+2, and tolerant check pos-1..pos+1.
    (this matches your previous “robust to ±1 bp boundary” idea)
    """
    m1 = seq[pos:pos + 2].seq.upper()
    m2 = seq[pos - 1:pos + 1].seq.upper() if pos > 0 else ""
    return motif in (m1, m2)


def _check_pos_has_motif(seq, pos: int, motif: str) -> bool:
    if pos < 0:
        return False
    return seq[pos:pos + 2].seq.upper() == motif


def _validate_one_df(df: pd.DataFrame, chrom: str, site_type: str):
    global bad

    # required cols (5-way)
    req = [
        "chrom", "ref_transcript_id", "ref_gene_id", "ref_strand",
        "label1_true_pos",
        "label2_near_decoy_pos", "label2_delta_bp",
        "label3_far_decoy_pos", "label3_delta_bp",
        "label4_other_same_gene_pos", "label4_delta_bp",
        "label5_other_diff_gene_pos", "label5_delta_bp",
    ]
    missing = [c for c in req if c not in df.columns]
    if missing:
        print(f"[FAIL] {chrom} {site_type}: missing columns: {missing}")
        bad = True
        return

    n = len(df)
    if n == 0:
        print(f"[FAIL] {chrom} {site_type}: empty")
        bad = True
        return

    if n > EXPECTED_PER_CHROM:
        print(f"[FAIL] {chrom} {site_type}: {n} rows > {EXPECTED_PER_CHROM}")
        bad = True

    seq = fa[chrom]
    anchor = pd.to_numeric(df["label1_true_pos"], errors="raise")

    # ------------------------------------------------------------
    # anchor motif validation (strand-aware)
    # ------------------------------------------------------------
    for _, r in df.iterrows():
        pos = int(r["label1_true_pos"])
        strand = r["ref_strand"]
        motif = _motif_for(site_type, strand)

        if not _check_anchor_motif(seq, pos, motif):
            m1 = seq[pos:pos + 2].seq.upper()
            m2 = seq[pos - 1:pos + 1].seq.upper() if pos > 0 else ""
            print(f"[FAIL] {chrom} {site_type}: anchor not {motif} near {pos} ({m1},{m2})")
            bad = True
            break

    # ------------------------------------------------------------
    # delta consistency checks
    # ------------------------------------------------------------
    for col_pos, col_delta in [
        ("label2_near_decoy_pos", "label2_delta_bp"),
        ("label3_far_decoy_pos", "label3_delta_bp"),
        ("label4_other_same_gene_pos", "label4_delta_bp"),
        ("label5_other_diff_gene_pos", "label5_delta_bp"),
    ]:
        pos = pd.to_numeric(df[col_pos], errors="coerce")
        delta = pd.to_numeric(df[col_delta], errors="coerce")

        mask = pos.notna()
        if ((pos[mask] - anchor[mask]) != delta[mask]).any():
            print(f"[FAIL] {chrom} {site_type}: delta mismatch in {col_pos}")
            bad = True

        coverage[site_type].setdefault(col_pos, 0)
        coverage[site_type][col_pos] += int(mask.sum())

    # ------------------------------------------------------------
    # motif sanity for labels 2-5 (exact 2bp match)
    # ------------------------------------------------------------
    for _, r in df.iterrows():
        strand = r["ref_strand"]
        motif = _motif_for(site_type, strand)

        for col in ["label2_near_decoy_pos", "label3_far_decoy_pos", "label4_other_same_gene_pos", "label5_other_diff_gene_pos"]:
            p = r[col]
            if pd.isna(p) or str(p) == ".":
                print(f"[FAIL] {chrom} {site_type}: missing {col}")
                bad = True
                break
            p = int(p)
            if not _check_pos_has_motif(seq, p, motif):
                got = seq[p:p + 2].seq.upper() if p >= 0 else ""
                print(f"[FAIL] {chrom} {site_type}: {col} not {motif} at {p} (got {got})")
                bad = True
                break
        if bad:
            break

    # ------------------------------------------------------------
    # distance distribution sanity
    # ------------------------------------------------------------
    for col in df.columns:
        if col.endswith("_delta_bp"):
            d = pd.to_numeric(df[col], errors="coerce").dropna().abs()
            if len(d) == 0:
                continue
            if d.quantile(0.99) > 1e6:
                print(f"[WARN] {chrom} {site_type}: huge deltas in {col}")
            if (d == 0).mean() > 0.5:
                print(f"[WARN] {chrom} {site_type}: too many zero deltas in {col}")

    # ------------------------------------------------------------
    # near/far sanity (matches generator intent)
    # ------------------------------------------------------------
    d2 = pd.to_numeric(df["label2_delta_bp"], errors="coerce").dropna().abs()
    d3 = pd.to_numeric(df["label3_delta_bp"], errors="coerce").dropna().abs()
    if len(d2) and d2.quantile(0.5) > 500:
        print(f"[WARN] {chrom} {site_type}: label2 looks not-so-near (median |delta|={d2.median():.1f})")
    if len(d3) and d3.quantile(0.5) < 200:
        print(f"[WARN] {chrom} {site_type}: label3 looks not-so-far (median |delta|={d3.median():.1f})")

    print(f"[OK] {chrom} {site_type}: {n} rows")


# ------------------------------------------------------------
# main loop
# ------------------------------------------------------------
for chrom in [f"chr{i}" for i in range(1, 23)]:
    for site_type, suffix in [("donor", DONOR_SUFFIX), ("acceptor", ACCEPTOR_SUFFIX)]:
        path = TSV_DIR / f"{chrom}_{suffix}"
        if not path.exists():
            print(f"[FAIL] missing {path}")
            bad = True
            continue

        df = pd.read_csv(path, sep="\t")
        _validate_one_df(df, chrom=chrom, site_type=site_type)

# ------------------------------------------------------------
# global coverage summary
# ------------------------------------------------------------
print("\n[coverage summary]")
for site_type in ["donor", "acceptor"]:
    print(f"\n[{site_type}]")
    for k, v in sorted(coverage[site_type].items()):
        print(f"{k}: {v}")

if bad:
    sys.exit(1)

print("\n[ALL CHECKS PASSED]")

# python /home/mica/gamba/data_processing/test_splice_tsvs.py \
#   /home/mica/gamba/data_processing/data/splice_sites
