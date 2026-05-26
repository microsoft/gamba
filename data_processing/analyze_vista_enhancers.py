#!/usr/bin/env python3
import argparse
import glob
import os
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

def load_vista_regions(regions_dir):
    """
    loads all /regions/vista_enhancer/chr*.bed files into a DataFrame
    columns: chrom, start, end
    """
    pattern = os.path.join(regions_dir, "vista_enhancer", "chr*.bed")
    beds = glob.glob(pattern)
    if not beds:
        raise FileNotFoundError(f"no VISTA enhancer beds found: {pattern}")

    dfs = []
    for bed in beds:
        df = pd.read_csv(
            bed,
            sep="\t",
            header=None,
            comment="#"
        )
        df = df.rename(columns={0: "chrom", 1: "start", 2: "end"})
        dfs.append(df)

    out = pd.concat(dfs, ignore_index=True)
    out["length"] = out["end"] - out["start"]
    return out


def compute_nearest_distances(df):
    """
    for each enhancer, compute distance to nearest other enhancer on same chromosome.
    
    distance = min( start[i+1] - end[i], start[i] - end[i-1] ), clipped at >= 0
    """
    df = df.sort_values(["chrom", "start"]).reset_index(drop=True)

    df["dist_prev"] = float("inf")
    df["dist_next"] = float("inf")

    # distance to previous
    for i in range(1, len(df)):
        if df.loc[i, "chrom"] == df.loc[i-1, "chrom"]:
            dist = df.loc[i, "start"] - df.loc[i-1, "end"]
            df.loc[i, "dist_prev"] = max(dist, 0)

    # distance to next
    for i in range(len(df)-1):
        if df.loc[i, "chrom"] == df.loc[i+1, "chrom"]:
            dist = df.loc[i+1, "start"] - df.loc[i, "end"]
            df.loc[i, "dist_next"] = max(dist, 0)

    df["nearest_dist"] = df[["dist_prev", "dist_next"]].min(axis=1)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--regions_dir",
        type=str,
        default="/home/mica/gamba/data_processing/data/regions",
        help="path to regions directory containing vista_enhancer/*.bed",
    )
    parser.add_argument(
        "--output_tsv",
        type=str,
        default="/home/mica/gamba/data_processing/data/regions/vista_enhancer_distance_summary.tsv",
    )
    parser.add_argument(
        "--out_prefix",
        type=str,
        default="/home/mica/gamba/data_processing/data/regions/vista_enhancer_distance_summary",
        help="prefix for output plots",
    )
    args = parser.parse_args()

    df = load_vista_regions(args.regions_dir)
    df = compute_nearest_distances(df)

    df.to_csv(args.output_tsv, sep="\t", index=False)

    import numpy as np

    df_sorted = df.sort_values(["chrom", "start"]).reset_index(drop=True)

    overlaps = (df_sorted["start"].shift(-1) < df_sorted["end"]) & \
            (df_sorted["chrom"].shift(-1) == df_sorted["chrom"])

    print("number overlapping:", overlaps.sum())
    print("fraction overlapping:", overlaps.sum() / len(df_sorted))

    print("\n=== nearest distance counts (under 50bp) ===")
    df[df["nearest_dist"] < 50]["nearest_dist"].value_counts().sort_index().head(20)
    print(df["nearest_dist"].min())
    print((df["nearest_dist"] <= 5).sum())
    print((df["nearest_dist"] <= 10).sum())
    print((df["nearest_dist"] <= 100).sum())

    print("\n=== nearest distance summary statistics ===")
    print(df["nearest_dist"].describe(percentiles=[0.01, 0.05, 0.1, 0.25]))

    print("\n=== summary statistics ===")
    print(df["length"].describe())
    print("\n=== distance to nearest enhancer ===")
    print(df["nearest_dist"].describe())

    if "length" not in df.columns or "nearest_dist" not in df.columns:
        raise ValueError("TSV must contain 'length' and 'nearest_dist' columns")

    # 1) histogram of lengths
    plt.figure()
    plt.hist(df["length"], bins=50)
    plt.xlabel("vista enhancer length (bp)")
    plt.ylabel("count")
    plt.title("distribution of vista enhancer lengths")
    plt.tight_layout()
    plt.savefig(f"{args.out_prefix}_length_hist.png", dpi=200)
    plt.close()

    # 2) histogram of nearest distances (linear)
    plt.figure()
    plt.hist(df["nearest_dist"], bins=100)
    plt.xlabel("distance to nearest vista enhancer (bp)")
    plt.ylabel("count")
    plt.title("distribution of nearest enhancer distances (linear)")
    plt.tight_layout()
    plt.ticklabel_format(style="plain", axis="x")
    plt.savefig(f"{args.out_prefix}_nearest_dist_hist_linear.png", dpi=200)
    plt.close()

    # 2b) histogram of nearest distances (log x-scale)
    # filter out zeros to avoid log(0) issues
    df_nonzero = df[df["nearest_dist"] > 0].copy()
    if not df_nonzero.empty:
        plt.figure()
        plt.hist(df_nonzero["nearest_dist"], bins=100)
        plt.xscale("log")
        plt.xlabel("distance to nearest vista enhancer (bp, log scale)")
        plt.ylabel("count")
        plt.title("distribution of nearest enhancer distances (log x)")
        plt.tight_layout()
        plt.savefig(f"{args.out_prefix}_nearest_dist_hist_log.png", dpi=200)
        plt.close()

    # 3) scatter: length vs nearest_dist
    plt.figure()
    plt.scatter(df["length"], df["nearest_dist"], alpha=0.3, s=5)
    plt.yscale("log")
    plt.xlabel("vista enhancer length (bp)")
    plt.ylabel("distance to nearest enhancer (bp, log scale)")
    plt.title("vista enhancer length vs nearest neighbour distance")
    plt.tight_layout()
    plt.savefig(f"{args.out_prefix}_length_vs_nearest_dist_scatter.png", dpi=200)
    plt.close()



if __name__ == "__main__":
    main()
