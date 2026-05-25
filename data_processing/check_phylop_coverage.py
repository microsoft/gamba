"""
check_phylop_coverage.py

Fast PhyloP missingness audit. Skips tokenization, fasta loading,
and array allocation entirely — just counts covered bases from
bigwig interval lengths.

Usage (single chrom):
    python check_phylop_coverage.py --chromosome chr1

Usage (all chroms in parallel):
    python check_phylop_coverage.py --all --workers 12
"""

import pyBigWig
import pandas as pd
import argparse
import csv
import os
import sys
from multiprocessing import Pool
from functools import partial

# ── core per-chromosome function ─────────────────────────────

def check_chromosome(chrom: str, bigwig_file: str, bed_df: pd.DataFrame) -> dict:
    """
    Returns coverage stats for one chromosome.
    No fasta, no tokenization, no array allocation.
    Covered bases are computed by summing interval lengths directly.
    """
    bw = pyBigWig.open(bigwig_file)

    chrom_bed = bed_df[bed_df["chrom"] == chrom]
    total_bp = 0
    covered_bp = 0

    for _, row in chrom_bed.iterrows():
        start, end = int(row["start"]), int(row["end"])
        region_len = end - start
        total_bp += region_len

        intervals = bw.intervals(chrom, start, end)
        if intervals:
            # sum interval lengths — no boolean array needed
            covered_bp += sum(iv_end - iv_start for iv_start, iv_end, _ in intervals)

    bw.close()

    missing_bp = total_bp - covered_bp
    pct_missing = 100.0 * missing_bp / total_bp if total_bp > 0 else 0.0

    return {
        "chromosome": chrom,
        "total_bp": total_bp,
        "covered_bp": covered_bp,
        "missing_bp": missing_bp,
        "pct_missing": round(pct_missing, 4),
    }


def _worker(chrom, bigwig_file, bed_df):
    """Thin wrapper so Pool.map can pickle the call."""
    print(f"  [{chrom}] starting...", flush=True)
    result = check_chromosome(chrom, bigwig_file, bed_df)
    pct = result["pct_missing"]
    print(f"  [{chrom}] done — {pct:.2f}% missing", flush=True)
    return result


# ── output helpers ────────────────────────────────────────────

def sort_key(r):
    c = r["chromosome"].replace("chr", "")
    return (0, int(c)) if c.isdigit() else (1, c)


def print_table(rows):
    header = (
        f"{'Chrom':<8} {'Total bp':>14} {'Covered bp':>14} "
        f"{'Missing bp':>14} {'% Missing':>10}"
    )
    sep = "─" * len(header)
    print("\n" + sep)
    print(header)
    print(sep)

    total = covered = 0
    for r in sorted(rows, key=sort_key):
        total   += r["total_bp"]
        covered += r["covered_bp"]
        print(
            f"{r['chromosome']:<8} {r['total_bp']:>14,} {r['covered_bp']:>14,} "
            f"{r['missing_bp']:>14,} {r['pct_missing']:>9.2f}%"
        )

    missing = total - covered
    pct = 100.0 * missing / total if total else 0.0
    print(sep)
    print(f"{'TOTAL':<8} {total:>14,} {covered:>14,} {missing:>14,} {pct:>9.2f}%")
    print(sep + "\n")


def save_csv(rows, path):
    fields = ["chromosome", "total_bp", "covered_bp", "missing_bp", "pct_missing"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(sorted(rows, key=sort_key))
    print(f"Stats saved → {path}")


# ── main ──────────────────────────────────────────────────────

ALL_CHROMS = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]

def main():
    parser = argparse.ArgumentParser(description="Fast PhyloP coverage audit")
    parser.add_argument("--bigwig_file", type=str,
        default="/home/mica/gamba/data_processing/data/240-mammalian/241-mammalian-2020v2.bigWig")
    parser.add_argument("--bed_file", type=str,
        default="/home/mica/gamba/data_processing/data/240-mammalian/regions.bed")
    parser.add_argument("--stats_file", type=str,
        default="/home/mica/gamba/data_processing/data/240-mammalian/phylop_coverage_stats.csv")
    parser.add_argument("--chromosome", type=str, default=None,
        help="Single chromosome to check (e.g. chr1). Omit when using --all.")
    parser.add_argument("--all", action="store_true",
        help="Check all chromosomes in parallel.")
    parser.add_argument("--workers", type=int, default=12,
        help="Number of parallel workers when using --all (default: 12).")
    args = parser.parse_args()

    bed = pd.read_csv(args.bed_file, sep="\t", header=None,
                      names=["chrom", "start", "end"])

    if args.all:
        # only process chroms that actually appear in the bed file
        chroms = [c for c in ALL_CHROMS if c in bed["chrom"].values]
        print(f"Checking {len(chroms)} chromosomes with {args.workers} workers...\n")

        worker_fn = partial(_worker, bigwig_file=args.bigwig_file, bed_df=bed)
        with Pool(processes=args.workers) as pool:
            rows = pool.map(worker_fn, chroms)

    elif args.chromosome:
        rows = [check_chromosome(args.chromosome, args.bigwig_file, bed)]
    else:
        parser.error("Provide --chromosome <chrom> or --all")

    print_table(rows)
    save_csv(rows, args.stats_file)


if __name__ == "__main__":
    main()