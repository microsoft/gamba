#!/usr/bin/env python3
"""
filter_splice_labels.py

Filter existing chr*_{donor,acceptor}_labels.tsv to high-quality rows and
subsample N per chromosome.

High-quality criteria:
- label1_pos is motif-correct in transcript orientation (GT for donor, AG for acceptor)
- optionally require labels 2-5 present
- optionally require motifs correct for labels 2-5 as well (recommended)
- optionally require near labels (2 and 4) present (to avoid missing_L2/L4)

Writes:
  <out_dir>/chr1_donor_labels.filtered.tsv
  <out_dir>/chr1_acceptor_labels.filtered.tsv
  ...

Then point splice_reps.py at --splice_tsv_dir <out_dir>.

Example:
  python filter_splice_labels.py \
    --in_dir  /home/mica/gamba/data_processing/data/splice_sites \
    --out_dir /home/mica/gamba/data_processing/data/splice_sites_filtered \
    --genome_fasta /home/mica/gamba/data_processing/data/240-mammalian/hg38.ml.fa \
    --chromosomes chr1 chr2 chr3 chr22 \
    --n_per_chrom 1000 \
    --require_all_labels \
    --require_all_motifs \
    --require_near
"""

import argparse
import os
import glob
from pathlib import Path
from collections import Counter, defaultdict

import pandas as pd
from pyfaidx import Fasta

RC = str.maketrans("ACGT", "TGCA")

LABEL_COLS = {
    1: "label1_pos",
    2: "label2_pos_cryptic_near",
    3: "label3_pos_cryptic_far",
    4: "label4_pos_annotated_near",
    5: "label5_pos_annotated_far",
}

def revcomp(s: str) -> str:
    return s.translate(RC)[::-1]

def fetch_2bp(genome: Fasta, chrom: str, pos0: int) -> str:
    s = str(genome[chrom][pos0:pos0+2]).upper()
    if len(s) != 2:
        return "??"
    if any(b not in "ACGT" for b in s):
        return "NN"
    return s

def motif_ok(genome, chrom, pos0, strand, expected_tx):
    """
    expected_tx: expected motif in transcript orientation (GT or AG)
    Checks dinuc at genome[chrom][pos:pos+2], reverse-complements on '-' strand,
    and compares to expected_tx.
    """
    d = fetch_2bp(genome, chrom, pos0)
    if d in ("??", "NN"):
        return False, d
    if strand == "-":
        d = revcomp(d)
    return (d == expected_tx), d

def load_one(in_dir, chrom, site_type):
    pattern = os.path.join(in_dir, f"{chrom}_{site_type}_labels.tsv")
    m = glob.glob(pattern)
    if not m:
        return None
    return pd.read_csv(m[0], sep="\t")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--genome_fasta", required=True)
    ap.add_argument("--chromosomes", nargs="+", required=True)
    ap.add_argument("--n_per_chrom", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--require_all_labels", action="store_true",
                    help="require labels 2-5 to be present (not '.')")
    ap.add_argument("--require_all_motifs", action="store_true",
                    help="require motif correctness for labels 2-5 as well as label1")
    ap.add_argument("--require_near", action="store_true",
                    help="require near controls present (label2 and label4 not '.')")

    ap.add_argument("--site_types", nargs="+", default=["donor", "acceptor"])
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    genome = Fasta(args.genome_fasta, as_raw=True, sequence_always_upper=True)

    summary = []

    for site_type in args.site_types:
        expected = "GT" if site_type == "donor" else "AG"

        for chrom in args.chromosomes:
            df = load_one(args.in_dir, chrom, site_type)
            if df is None or len(df) == 0:
                print(f"[warn] missing/empty {chrom} {site_type}")
                continue

            n0 = len(df)

            # optional label presence filters
            if args.require_all_labels:
                mask = (df[LABEL_COLS[2]] != ".") & (df[LABEL_COLS[3]] != ".") & (df[LABEL_COLS[4]] != ".") & (df[LABEL_COLS[5]] != ".")
                df = df[mask].copy()

            if args.require_near:
                mask = (df[LABEL_COLS[2]] != ".") & (df[LABEL_COLS[4]] != ".")
                df = df[mask].copy()

            # motif correctness
            keep = []
            bad_counts = Counter()

            for i, r in df.iterrows():
                strand = r.get("strand", "+")
                if strand not in ["+", "-"]:
                    strand = "+"

                # label1 must be correct
                try:
                    p1 = int(r[LABEL_COLS[1]])
                except Exception:
                    bad_counts["bad_label1_int"] += 1
                    continue
                ok1, d1 = motif_ok(genome, r["chrom"], p1, strand, expected)
                if not ok1:
                    bad_counts[f"label1_bad_{d1}"] += 1
                    continue

                if args.require_all_motifs:
                    ok_all = True
                    for lid in (2, 3, 4, 5):
                        v = r[LABEL_COLS[lid]]
                        if v == "." or pd.isna(v):
                            ok_all = False
                            bad_counts[f"missing_L{lid}"] += 1
                            break
                        try:
                            p = int(v)
                        except Exception:
                            ok_all = False
                            bad_counts[f"badint_L{lid}"] += 1
                            break
                        ok, d = motif_ok(genome, r["chrom"], p, strand, expected)
                        if not ok:
                            ok_all = False
                            bad_counts[f"L{lid}_bad_{d}"] += 1
                            break
                    if not ok_all:
                        continue

                keep.append(i)

            df_f = df.loc[keep].copy()
            n1 = len(df_f)

            # subsample per chrom
            if args.n_per_chrom is not None and n1 > args.n_per_chrom:
                df_f = df_f.sample(args.n_per_chrom, random_state=args.seed)
            n2 = len(df_f)

            out_path = out_dir / f"{chrom}_{site_type}_labels.filtered.tsv"
            df_f.to_csv(out_path, sep="\t", index=False)

            print(f"[{chrom} {site_type}] start={n0} after_presence={len(df)} after_motif={n1} saved={n2} -> {out_path}")
            if bad_counts:
                top = bad_counts.most_common(6)
                print(f"  top drop reasons: {top}")

            summary.append({
                "chrom": chrom,
                "site_type": site_type,
                "expected": expected,
                "n_start": n0,
                "n_after_presence": len(df),
                "n_after_motif": n1,
                "n_saved": n2,
            })

    # write summary
    summary_path = out_dir / "filter_summary.tsv"
    pd.DataFrame(summary).to_csv(summary_path, sep="\t", index=False)
    print(f"\nwrote summary: {summary_path}")

if __name__ == "__main__":
    main()

# python /home/mica/gamba/data_processing/filter_splice_labels.py \
#   --in_dir  /home/mica/gamba/data_processing/data/splice_sites \
#   --out_dir /home/mica/gamba/data_processing/data/splice_sites_filtered \
#   --genome_fasta /home/mica/gamba/data_processing/data/240-mammalian/hg38.ml.fa \
#   --chromosomes chr1 chr2 chr3 chr4 chr5 chr6 chr7 chr8 chr9 chr10 chr11 chr12 chr13 chr14 chr15 chr16 chr17 chr18 chr19 chr20 chr21 chr22 chrX \
#   --n_per_chrom 1000 \
#   --require_all_labels \
#   --require_all_motifs \
#   --require_near
