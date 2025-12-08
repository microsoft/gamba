#!/usr/bin/env python3
import argparse
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "for each ROI category, compute the % of upstream regions whose "
            "category-based annotation matches the ROI category"
        )
    )
    parser.add_argument(
        "--input_tsv",
        type=str,
        default="/home/mica/gamba/data_processing/data/240-mammalian/upstream_region_annotations.tsv",
        help="input TSV produced by upstream_region_annotations script",
    )
    parser.add_argument(
        "--output_tsv",
        type=str,
        default="/home/mica/gamba/data_processing/data/240-mammalian/upstream_region_match_summary.tsv",
        help="optional path to write summary TSV; if empty, no file is written",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    input_path = Path(args.input_tsv)
    if not input_path.exists():
        raise FileNotFoundError(f"input TSV not found: {input_path}")

    logging.info(f"loading {input_path}")
    df = pd.read_csv(input_path, sep="\t")

    required_cols = [
        "category_its_upstream_of",
        "region_identified_by_category",
    ]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"missing required column in TSV: {col}")

    # normalize region_identified_by_category to string
    df["region_identified_by_category"] = df["region_identified_by_category"].fillna(
        "unknown"
    ).astype(str)

    # helper: does this upstream region include the same category label as the ROI category?
    def has_same_category(row) -> bool:
        roi_cat = row["category_its_upstream_of"]
        upstream_cats = row["region_identified_by_category"]
        if upstream_cats == "unknown" or upstream_cats.strip() == "":
            return False
        cats = [c.strip() for c in upstream_cats.split(";") if c.strip()]
        return roi_cat in cats

    df["same_category_upstream"] = df.apply(has_same_category, axis=1)

    # for extra context, mark unknowns
    df["upstream_unknown"] = df["region_identified_by_category"].eq("unknown")

    # aggregate per ROI category
    grouped = df.groupby("category_its_upstream_of").agg(
        total_upstream=("same_category_upstream", "size"),
        num_same_category=("same_category_upstream", "sum"),
        num_unknown=("upstream_unknown", "sum"),
    )
    grouped["pct_same_category"] = (
        grouped["num_same_category"] / grouped["total_upstream"] * 100.0
    )
    grouped["pct_unknown"] = (
        grouped["num_unknown"] / grouped["total_upstream"] * 100.0
    )

    grouped = grouped.reset_index().rename(
        columns={"category_its_upstream_of": "category"}
    )

    # print nicely
    pd.set_option("display.max_rows", None)
    pd.set_option("display.float_format", lambda x: f"{x:0.2f}")
    print(grouped.sort_values("category").to_string(index=False))

    # optionally write to file
    if args.output_tsv:
        out_path = Path(args.output_tsv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        logging.info(f"writing summary to {out_path}")
        grouped.to_csv(out_path, sep="\t", index=False)


if __name__ == "__main__":
    main()
