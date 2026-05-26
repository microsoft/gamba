#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd


def parse_coord(coord: str):
    """parse strings like 'chr16:86396481-86397120' -> (chrom, start, end)."""
    if pd.isna(coord):
        return None, None, None
    chrom, rest = coord.split(":")
    start, end = rest.split("-")
    return chrom, int(start), int(end)


def chrom_sort_key(chrom: str) -> int:
    """map chr1..chr22, chrX, chrY, chrM to sortable ints."""
    if chrom is None or not isinstance(chrom, str):
        return 999

    name = chrom.lower().replace("chr", "")
    if name == "x":
        return 23
    if name == "y":
        return 24
    if name in ("m", "mt"):
        return 25
    try:
        return int(name)
    except ValueError:
        return 998  # weird contigs etc


def sort_by_coord(df: pd.DataFrame, coord_col: str) -> pd.DataFrame:
    parsed = df[coord_col].apply(parse_coord)
    coord_df = pd.DataFrame(parsed.tolist(), index=df.index, columns=["chrom", "start", "end"])

    tmp = df.copy()
    tmp[["chrom", "start", "end"]] = coord_df
    tmp["chrom_order"] = tmp["chrom"].apply(chrom_sort_key)

    tmp = tmp.sort_values(["chrom_order", "start", "end"], kind="mergesort")
    return tmp.drop(columns=["chrom", "start", "end", "chrom_order"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="/home/mica/gamba/data_processing/data/VISTA_enhancers/experiments_new.tsv",
        help="path to VISTA experiments tsv",
    )
    parser.add_argument(
        "--outdir",
        default="/home/mica/gamba/data_processing/data/VISTA_enhancers/subsets",
        help="directory to write subset tsvs",
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # read tsv
    df = pd.read_csv(in_path, sep="\t", dtype=str)

    # normalize a couple of columns
    df["organism_norm"] = df["organism"].str.strip().str.lower()
    df["curation_norm"] = df["curation_status"].str.strip().str.lower()

    # subsets
    human = df[df["organism_norm"] == "human"]
    mouse = df[df["organism_norm"] == "mouse"]

    human_pos = human[human["curation_norm"] == "positive"]
    human_neg = human[human["curation_norm"] == "negative"]
    mouse_pos = mouse[mouse["curation_norm"] == "positive"]
    mouse_neg = mouse[mouse["curation_norm"] == "negative"]

    # sort by appropriate genome coordinates
    human_pos_sorted = sort_by_coord(human_pos, "coordinate_hg38")
    human_neg_sorted = sort_by_coord(human_neg, "coordinate_hg38")
    mouse_pos_sorted = sort_by_coord(mouse_pos, "coordinate_mm10")
    mouse_neg_sorted = sort_by_coord(mouse_neg, "coordinate_mm10")

    # write out
    human_pos_sorted.drop(columns=["organism_norm", "curation_norm"]).to_csv(
        outdir / "vista_human_positive.tsv", sep="\t", index=False
    )
    human_neg_sorted.drop(columns=["organism_norm", "curation_norm"]).to_csv(
        outdir / "vista_human_negative.tsv", sep="\t", index=False
    )
    mouse_pos_sorted.drop(columns=["organism_norm", "curation_norm"]).to_csv(
        outdir / "vista_mouse_positive.tsv", sep="\t", index=False
    )
    mouse_neg_sorted.drop(columns=["organism_norm", "curation_norm"]).to_csv(
        outdir / "vista_mouse_negative.tsv", sep="\t", index=False
    )


if __name__ == "__main__":
    main()
