#!/usr/bin/env python3
"""
build regions/ dataset with a single interval convention:

- all outputs are 0-based, half-open [start, end)
- bed inputs are assumed already 0-based half-open
- gtf inputs are 1-based, inclusive -> converted to 0-based half-open
- vista coordinate strings like "chr1:3274017-3274864" are treated as 1-based, inclusive -> converted

outputs:
regions/
  CATEGORY/
  CATEGORY_upstream/
  CATEGORY_random-noannot/
  CATEGORY_random/
  manifest.tsv
  manifest.json
"""

import argparse
import os
import json
import logging
import random
from pathlib import Path
from collections import defaultdict, Counter
from glob import glob
from bisect import bisect_left

import pandas as pd
import numpy as np
from tqdm import tqdm
from pyfaidx import Fasta

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# -----------------------
# interval helpers
# -----------------------

def gtf_to_bed(start1: int, end1: int) -> tuple[int, int]:
    """
    gtf is 1-based inclusive [start1, end1]
    convert to 0-based half-open [start0, end0)
    """
    start0 = start1 - 1
    end0 = end1
    return start0, end0

def parse_coord_str_1based_inclusive(coord: str) -> tuple[str, int, int]:
    """
    coord like "chr1:3274017-3274864" (assume 1-based inclusive)
    returns (chrom, start0, end0) in bed coords
    """
    coord = str(coord).strip()
    chrom_part, pos_part = coord.split(":")
    chrom = chrom_part if chrom_part.startswith("chr") else "chr" + chrom_part
    s1_str, e1_str = pos_part.split("-")
    s1 = int(s1_str)
    e1 = int(e1_str)
    s0, e0 = gtf_to_bed(s1, e1)
    return chrom, s0, e0

def _overlaps(iv, start, end):
    # iv sorted list of (start,end) merged, half-open
    i = bisect_left(iv, (start, end))
    if i > 0 and iv[i - 1][1] > start:
        return True
    if i < len(iv) and iv[i][0] < end:
        return True
    return False

def _insert_merge(iv, start, end):
    # keep iv sorted, merged; do NOT merge abutting
    i = bisect_left(iv, (start, end))
    s, e = start, end

    j = i - 1
    while j >= 0 and iv[j][1] > s:
        s = min(s, iv[j][0])
        e = max(e, iv[j][1])
        j -= 1
    j += 1

    k = i
    while k < len(iv) and iv[k][0] < e:
        s = min(s, iv[k][0])
        e = max(e, iv[k][1])
        k += 1

    iv[j:k] = [(s, e)]

def ensure_half_open(region: dict) -> dict:
    s = int(region["start"])
    e = int(region["end"])
    if e < s:
        s, e = e, s
    if e == s:
        return None
    region["start"] = s
    region["end"] = e
    return region

# -----------------------
# bed loading
# -----------------------

def build_canonical_set(genome_fasta: str) -> set[str]:
    fa = Fasta(genome_fasta)
    return set(fa.keys())

def normalize_chrom(chrom: str, canonical: set[str]) -> str:
    if chrom in canonical:
        return chrom
    if chrom.startswith("chr"):
        nochr = chrom[3:]
        if nochr in canonical:
            return nochr
    else:
        withchr = "chr" + chrom
        if withchr in canonical:
            return withchr
    mito_aliases = {"M", "MT", "chrM", "chrMT"}
    if chrom in mito_aliases:
        for cand in ("chrM", "MT", "M"):
            if cand in canonical:
                return cand
    return chrom

def normalize_chrom_list(chroms, canonical):
    return [normalize_chrom(c, canonical) for c in chroms]

def load_name_blocklist(txt_path: str) -> set[str]:
    if not os.path.exists(txt_path):
        logging.warning(f"ucne paralogue list not found: {txt_path}")
        return set()

    to_drop = set()
    with open(txt_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            group = [x.strip() for x in line.replace(",", " ").split() if x.strip()]
            if len(group) > 1:
                to_drop.update(group[1:])
    logging.info(f"loaded paralogue drop list: {len(to_drop)} names")
    return to_drop

def load_bed_file_filtered(path, category, keep_chroms=None, drop_names=None, canonical=None):
    df = pd.read_csv(path, sep="\t", header=None, comment="#")
    if df.shape[1] < 3:
        raise ValueError(f"BED file {path} must have >= 3 columns")

    all_cols = [
        "chrom","start","end","name","score","strand",
        "thickStart","thickEnd","itemRgb","blockCount","blockSizes","blockStarts"
    ]
    df.columns = all_cols[: df.shape[1]]

    if "name" not in df.columns: df["name"] = category
    if "score" not in df.columns: df["score"] = 0.0
    if "strand" not in df.columns: df["strand"] = "."

    if canonical is not None:
        df["chrom"] = df["chrom"].map(lambda c: normalize_chrom(str(c), canonical))

    if keep_chroms is not None:
        keep_set = set(keep_chroms)
        df = df[df["chrom"].isin(keep_set)]

    if drop_names:
        if df["name"].nunique() == 1 and next(iter(df["name"].unique())) == category:
            logging.warning(f"[{category}] bed lacks distinct names; cannot drop by name reliably")
        else:
            df = df[~df["name"].isin(drop_names)]

    df["start"] = df["start"].astype(int)
    df["end"] = df["end"].astype(int)

    out = []
    for _, r in df.iterrows():
        reg = {
            "chrom": r["chrom"],
            "start": int(r["start"]),
            "end": int(r["end"]),
            "name": str(r["name"]),
            "score": float(r["score"]),
            "strand": str(r["strand"]),
            "category": category,
        }
        reg = ensure_half_open(reg)
        if reg is not None:
            out.append(reg)

    logging.info(f"[{category}] {path}: kept {len(out)}/{len(df)} rows")
    return out

# -----------------------
# vista loading (1-based inclusive coords -> bed)
# -----------------------

def load_vista_coordinates(tsv_path: str, canonical: set[str], keep_chroms: list[str]):
    df = pd.read_csv(tsv_path, sep="\t")
    if "coordinate_hg38" in df.columns:
        coord_col = "coordinate_hg38"
    elif "coord" in df.columns:
        coord_col = "coord"
    else:
        raise ValueError(f"no coord column in vista tsv: {tsv_path}")

    id_col = "vista_id" if "vista_id" in df.columns else df.columns[0]
    strand_col = "strand" if "strand" in df.columns else None

    keep_set = set(keep_chroms)

    out = []
    for _, row in df.iterrows():
        coord = row[coord_col]
        if pd.isna(coord):
            continue
        chrom, s0, e0 = parse_coord_str_1based_inclusive(coord)

        chrom = normalize_chrom(chrom, canonical)
        if chrom not in keep_set:
            continue

        reg = {
            "chrom": chrom,
            "start": s0,
            "end": e0,
            "name": str(row[id_col]),
            "score": 0.0,
            "strand": str(row[strand_col]) if strand_col else ".",
            "category": "vista_enhancer",
        }
        reg = ensure_half_open(reg)
        if reg is not None:
            out.append(reg)

    logging.info(f"[vista_enhancer] loaded {len(out)} regions from {tsv_path}")
    return out

# -----------------------
# gtf parsing (FIXED)
# -----------------------

def list_gtf_files(gtf_dir):
    files = sorted(glob(os.path.join(gtf_dir, "*.gtf"))) + sorted(glob(os.path.join(gtf_dir, "*.gtf.gz")))
    if not files:
        raise FileNotFoundError(f"no gtf files in {gtf_dir}")
    return files

class GTFParser:
    """
    FIX: convert all gtf coords (1-based inclusive) -> bed coords (0-based half-open)
    FIX: introns computed in half-open space (no +1/-1)
    FIX: upstream_TSS computed in half-open space
    """

    def __init__(self, gtf_files):
        self.features = {
            "coding_regions": [],
            "exons": [],
            "introns": [],
            "upstream_TSS": [],
        }
        self.all_intervals_by_chrom = defaultdict(list)
        self._parse_gtf_files(gtf_files)

    def _parse_attributes(self, attr_string):
        attrs = {}
        for attr in attr_string.strip().split(";"):
            a = attr.strip()
            if not a:
                continue
            if " " not in a:
                continue
            key, value = a.split(" ", 1)
            attrs[key] = value.strip().strip('"')
        return attrs

    def _open_gtf(self, path):
        if path.endswith(".gz"):
            import gzip
            return gzip.open(path, "rt")
        return open(path, "r")

    def _parse_gtf_files(self, gtf_files):
        logging.info(f"parsing {len(gtf_files)} gtf files")

        for gtf_file in gtf_files:
            logging.info(f"parsing: {gtf_file}")

            genes = {}
            transcripts = {}

            with self._open_gtf(gtf_file) as f:
                for line in tqdm(f, desc=f"read {os.path.basename(gtf_file)}"):
                    if not line or line.startswith("#"):
                        continue
                    fields = line.rstrip("\n").split("\t")
                    if len(fields) < 9:
                        continue

                    chrom, source, feature_type, start1, end1, score, strand, frame, attributes = fields
                    try:
                        start1 = int(start1)
                        end1 = int(end1)
                    except ValueError:
                        continue

                    # FIX: gtf -> bed conversion
                    start0, end0 = gtf_to_bed(start1, end1)
                    if end0 <= start0:
                        continue

                    try:
                        attrs = self._parse_attributes(attributes)
                    except Exception:
                        continue

                    transcript_id = attrs.get("transcript_id")
                    gene_id = attrs.get("gene_id")
                    transcript_type = attrs.get("transcript_type")

                    if feature_type == "gene":
                        gene_type = attrs.get("gene_type")
                        if gene_id and gene_type:
                            genes[gene_id] = {
                                "chrom": chrom,
                                "start": start0,
                                "end": end0,
                                "strand": strand,
                                "type": gene_type,
                                "transcripts": [],
                            }

                    elif feature_type == "transcript":
                        if gene_id and transcript_id and transcript_type:
                            is_canonical = ('tag "Ensembl_canonical"' in attributes) or ('tag "CCDS"' in attributes)
                            transcripts[transcript_id] = {
                                "gene_id": gene_id,
                                "chrom": chrom,
                                "start": start0,
                                "end": end0,
                                "strand": strand,
                                "type": transcript_type,
                                "is_canonical": is_canonical,
                                "exons": [],
                                "cds_regions": [],
                            }
                            if gene_id in genes:
                                genes[gene_id]["transcripts"].append(transcript_id)

                    elif feature_type == "exon" and transcript_id in transcripts:
                        transcripts[transcript_id]["exons"].append({"start": start0, "end": end0})

                    elif feature_type == "CDS" and transcript_id in transcripts:
                        transcripts[transcript_id]["cds_regions"].append({"start": start0, "end": end0})

            # finalize per canonical transcript
            for transcript_id, tx in transcripts.items():
                if not tx["is_canonical"]:
                    continue

                chrom = tx["chrom"]
                strand = tx["strand"]

                # coding regions (CDS segments)
                if tx["type"] == "protein_coding" and tx["cds_regions"]:
                    for cds in tx["cds_regions"]:
                        self.features["coding_regions"].append({
                            "chrom": chrom,
                            "start": int(cds["start"]),
                            "end": int(cds["end"]),
                            "strand": strand,
                            "name": transcript_id,
                            "score": 0.0,
                        })

                # exons
                for exon in tx["exons"]:
                    self.features["exons"].append({
                        "chrom": chrom,
                        "start": int(exon["start"]),
                        "end": int(exon["end"]),
                        "strand": strand,
                        "name": transcript_id,
                        "score": 0.0,
                    })

                # FIX: introns in half-open space: [prev_exon_end, next_exon_start)
                if len(tx["exons"]) >= 2:
                    exons_sorted = sorted(tx["exons"], key=lambda x: x["start"])
                    for i in range(len(exons_sorted) - 1):
                        intron_start = int(exons_sorted[i]["end"])
                        intron_end = int(exons_sorted[i + 1]["start"])
                        if intron_end > intron_start:
                            self.features["introns"].append({
                                "chrom": chrom,
                                "start": intron_start,
                                "end": intron_end,
                                "strand": strand,
                                "name": transcript_id,
                                "score": 0.0,
                            })

                # FIX: upstream_TSS in half-open space
                tx_start = int(tx["start"])
                tx_end = int(tx["end"])  # exclusive

                if strand == "+":
                    up_start = tx_start - 2000
                    up_end = tx_start
                else:
                    up_start = tx_end
                    up_end = tx_end + 2000

                if up_end > up_start:
                    self.features["upstream_TSS"].append({
                        "chrom": chrom,
                        "start": up_start,
                        "end": up_end,
                        "strand": strand,
                        "name": transcript_id,
                        "score": 0.0,
                    })

                # track all annotated (for later subtraction / random-noannot)
                for feat in tx["cds_regions"] + tx["exons"]:
                    self.all_intervals_by_chrom[chrom].append((int(feat["start"]), int(feat["end"])))

    def get_regions_by_type(self, feature_type, chrom=None):
        if feature_type not in self.features:
            raise ValueError(f"unknown feature type: {feature_type}")
        regs = self.features[feature_type]
        if chrom:
            return [r for r in regs if r["chrom"] == chrom]
        return regs

def sample_regions_by_feature(gtf_parser, feature_type, chromosomes, canonical=None):
    chrom_set = set(chromosomes)
    out = []
    for chrom in chromosomes:
        feats = gtf_parser.get_regions_by_type(feature_type, chrom=chrom)
        for f in feats:
            c = f["chrom"]
            if canonical is not None:
                c = normalize_chrom(c, canonical)
            if c not in chrom_set:
                continue

            reg = {
                "chrom": c,
                "start": int(f["start"]),
                "end": int(f["end"]),
                "name": f.get("name", feature_type),
                "score": float(f.get("score", 0.0)),
                "strand": f.get("strand", "."),
                "category": feature_type,
            }
            reg = ensure_half_open(reg)
            if reg is not None:
                out.append(reg)

    logging.info(f"[{feature_type}] kept {len(out)} rows")
    return out

# -----------------------
# non-overlap + upstream pairing
# -----------------------

def ensure_nonoverlap(categories, order, limit_per_category=None, seed=42):
    rng = random.Random(seed)
    occupied = defaultdict(list)  # chrom -> merged non-overlapping ivs
    out = {}

    for cat in order:
        pool = list(categories.get(cat, []))
        rng.shuffle(pool)

        kept = []
        for r in pool:
            c = r["chrom"]
            s = int(r["start"])
            e = int(r["end"])
            if e <= s:
                continue

            iv = occupied[c]
            if not _overlaps(iv, s, e):
                kept.append(r)
                _insert_merge(iv, s, e)
                if limit_per_category and len(kept) >= limit_per_category:
                    break

        out[cat] = kept
        logging.info(f"[NON-OVERLAP] {cat}: kept {len(kept)}")
    return out

def build_upstream_regions(categories, chrom_lengths, upstream_len=2000):
    """
    for each ROI, require a strand-aware upstream region of identical length, located upstream_len upstream.
    upstream must NOT overlap any ROI from the same category.
    """
    # build per-category occupied intervals for overlap checks
    occ_by_cat = defaultdict(lambda: defaultdict(list))  # cat -> chrom -> merged ivs
    for cat, regs in categories.items():
        for r in regs:
            c = r["chrom"]
            s = int(r["start"])
            e = int(r["end"])
            _insert_merge(occ_by_cat[cat][c], s, e)

    filtered = {cat: [] for cat in categories}
    upstream = {f"{cat}_upstream": [] for cat in categories}

    pair_id = 0
    for cat, regs in categories.items():
        up_cat = f"{cat}_upstream"

        for r in regs:
            chrom = r["chrom"]
            strand = r.get("strand", ".")
            s = int(r["start"])
            e = int(r["end"])
            L = e - s
            if L <= 0:
                continue
            if chrom not in chrom_lengths:
                continue

            # anchor 2kb upstream, same length as ROI
            if strand == "-":
                # upstream is increasing coords
                us_start = e + upstream_len
                us_end = us_start + L
            else:
                us_end = s - upstream_len
                us_start = us_end - L

            if us_start < 0 or us_end > chrom_lengths[chrom]:
                continue
            if us_end <= us_start:
                continue

            # must not overlap same category
            if _overlaps(occ_by_cat[cat][chrom], us_start, us_end):
                continue

            pid = pair_id
            pair_id += 1

            anchor = dict(r)
            anchor["pair_id"] = pid
            filtered[cat].append(anchor)

            upstream[up_cat].append({
                "chrom": chrom,
                "start": int(us_start),
                "end": int(us_end),
                "name": f"{r.get('name', cat)}_up",
                "score": 0.0,
                "strand": strand,
                "category": up_cat,
                "pair_id": pid,
            })

    for cat in filtered:
        logging.info(f"[UPSTREAM] {cat}: kept {len(filtered[cat])} anchors")
    return filtered, upstream

# -----------------------
# random sampling (matched length)
# -----------------------

def build_occupied_all(categories):
    occ = defaultdict(list)
    for cat, regs in categories.items():
        for r in regs:
            c = r["chrom"]
            s = int(r["start"])
            e = int(r["end"])
            _insert_merge(occ[c], s, e)
    return occ

def build_occupied_by_cat(categories):
    occ = defaultdict(lambda: defaultdict(list))
    for cat, regs in categories.items():
        for r in regs:
            c = r["chrom"]
            s = int(r["start"])
            e = int(r["end"])
            _insert_merge(occ[cat][c], s, e)
    return occ

def sample_random_interval(chrom, length, chrom_size, occupied_iv, rng, max_attempts=2000):
    if length <= 0 or length > chrom_size:
        return None
    for _ in range(max_attempts):
        start = rng.randint(0, chrom_size - length)
        end = start + length
        if not _overlaps(occupied_iv, start, end):
            _insert_merge(occupied_iv, start, end)
            return start, end
    return None

def build_random_sets(functional_categories, chrom_lengths, seed=42, max_attempts=2000):
    """
    for each ROI in each category:
      - random-noannot: avoid overlap with ANY functional ROI (all categories)
      - random: avoid overlap with same category only
    """
    rng = random.Random(seed)

    occ_all = build_occupied_all(functional_categories)
    occ_by_cat = build_occupied_by_cat(functional_categories)

    # also make occupancies for already-sampled randoms (so randoms don't overlap each other)
    occ_rno = defaultdict(list)   # chrom -> merged
    occ_rcat = defaultdict(list)  # chrom -> merged

    random_noannot = {f"{cat}_random-noannot": [] for cat in functional_categories}
    random_cat = {f"{cat}_random": [] for cat in functional_categories}

    for cat, regs in functional_categories.items():
        for r in regs:
            chrom = r["chrom"]
            if chrom not in chrom_lengths:
                continue
            L = int(r["end"]) - int(r["start"])
            if L <= 0:
                continue

            # random-noannot: avoid ALL + previous random-noannot
            occ_tmp = occ_all[chrom]
            # create a combined occupancy by merging into a copy
            combined_noannot = list(occ_tmp)
            for iv in occ_rno[chrom]:
                _insert_merge(combined_noannot, iv[0], iv[1])

            sampled = sample_random_interval(
                chrom, L, chrom_lengths[chrom], combined_noannot, rng, max_attempts=max_attempts
            )
            if sampled is not None:
                s0, e0 = sampled
                _insert_merge(occ_rno[chrom], s0, e0)
                random_noannot[f"{cat}_random-noannot"].append({
                    "chrom": chrom,
                    "start": s0,
                    "end": e0,
                    "name": f"{r.get('name', cat)}_rno",
                    "score": 0.0,
                    "strand": r.get("strand", "."),
                    "category": f"{cat}_random-noannot",
                    "pair_id": r.get("pair_id", None),
                })

            # category-matched random: avoid SAME CAT + previous category-randoms
            combined_cat = list(occ_by_cat[cat][chrom])
            for iv in occ_rcat[chrom]:
                _insert_merge(combined_cat, iv[0], iv[1])

            sampled2 = sample_random_interval(
                chrom, L, chrom_lengths[chrom], combined_cat, rng, max_attempts=max_attempts
            )
            if sampled2 is not None:
                s1, e1 = sampled2
                _insert_merge(occ_rcat[chrom], s1, e1)
                random_cat[f"{cat}_random"].append({
                    "chrom": chrom,
                    "start": s1,
                    "end": e1,
                    "name": f"{r.get('name', cat)}_rcat",
                    "score": 0.0,
                    "strand": r.get("strand", "."),
                    "category": f"{cat}_random",
                    "pair_id": r.get("pair_id", None),
                })

    return random_noannot, random_cat

# -----------------------
# writing + manifest
# -----------------------

def write_bed(regions, output_dir, category):
    output_dir = Path(output_dir)
    out_path = output_dir / category
    out_path.mkdir(parents=True, exist_ok=True)

    by_chrom = defaultdict(list)
    for r in regions:
        by_chrom[r["chrom"]].append(r)

    for chrom, regs in by_chrom.items():
        out_file = out_path / f"{chrom}.bed"
        with out_file.open("w") as f:
            for r in regs:
                fields = [
                    r["chrom"],
                    str(int(r["start"])),
                    str(int(r["end"])),
                    str(r.get("name", category)),
                    str(r.get("score", 0.0)),
                    str(r.get("strand", ".")),
                ]
                if r.get("pair_id", None) is not None:
                    fields.append(str(r["pair_id"]))
                f.write("\t".join(fields) + "\n")
        logging.info(f"saved {len(regs)} to {out_file}")

def read_bed(path: Path):
    rows = []
    with path.open() as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            chrom = parts[0]
            start = int(parts[1])
            end = int(parts[2])
            strand = parts[5] if len(parts) >= 6 else "."
            pair_id = parts[6] if len(parts) >= 7 else None
            if end <= start:
                continue
            rows.append((chrom, start, end, strand, pair_id))
    return rows

def quantiles(arr, qs=(0, 5, 25, 50, 75, 95, 100)):
    if len(arr) == 0:
        return {f"p{q}": None for q in qs}
    a = np.asarray(arr)
    return {f"p{q}": float(np.percentile(a, q)) for q in qs}

def write_manifest(regions_root: str):
    root = Path(regions_root)
    out_tsv = root / "manifest.tsv"
    out_json = root / "manifest.json"

    splits = [d for d in root.iterdir() if d.is_dir()]

    header = [
        "split",
        "n_regions",
        "len_min","len_p5","len_p25","len_median","len_p75","len_p95","len_max",
        "len_mean","len_std",
        "pair_id_frac","n_unique_pair_id","pair_id_duplicates",
        "strand_counts_json",
        "top_chroms_json",
    ]

    all_stats = {}
    lines = ["\t".join(header)]

    for d in sorted(splits, key=lambda x: x.name):
        bed_files = sorted(d.glob("*.bed"))
        if not bed_files:
            continue

        lens = []
        chrom_counts = Counter()
        strand_counts = Counter()
        pair_ids = []
        n = 0

        for bf in bed_files:
            for chrom, start, end, strand, pid in read_bed(bf):
                n += 1
                lens.append(end - start)
                chrom_counts[chrom] += 1
                strand_counts[strand] += 1
                if pid is not None:
                    pair_ids.append(pid)

        lens = np.asarray(lens, dtype=np.int64)
        q = quantiles(lens, qs=(0,5,25,50,75,95,100))

        stats = {
            "n_regions": int(n),
            "len_min": q["p0"],
            "len_p5": q["p5"],
            "len_p25": q["p25"],
            "len_median": q["p50"],
            "len_p75": q["p75"],
            "len_p95": q["p95"],
            "len_max": q["p100"],
            "len_mean": float(lens.mean()) if len(lens) else None,
            "len_std": float(lens.std(ddof=0)) if len(lens) else None,
            "strand_counts": dict(strand_counts),
            "top_chroms": dict(chrom_counts.most_common(10)),
            "n_with_pair_id": int(len(pair_ids)),
            "pair_id_frac": float(len(pair_ids) / n) if n else None,
            "n_unique_pair_id": int(len(set(pair_ids))),
            "pair_id_duplicates": int(len(pair_ids) - len(set(pair_ids))),
        }
        all_stats[d.name] = stats

        row = [
            d.name,
            str(stats["n_regions"]),
            str(stats["len_min"]), str(stats["len_p5"]), str(stats["len_p25"]),
            str(stats["len_median"]), str(stats["len_p75"]), str(stats["len_p95"]), str(stats["len_max"]),
            str(stats["len_mean"]), str(stats["len_std"]),
            str(stats["pair_id_frac"]), str(stats["n_unique_pair_id"]), str(stats["pair_id_duplicates"]),
            json.dumps(stats["strand_counts"], sort_keys=True),
            json.dumps(stats["top_chroms"], sort_keys=True),
        ]
        lines.append("\t".join(row))

    out_tsv.write_text("\n".join(lines) + "\n")
    out_json.write_text(json.dumps(all_stats, indent=2, sort_keys=True) + "\n")
    logging.info(f"wrote manifest: {out_tsv}")
    logging.info(f"wrote manifest: {out_json}")

# -----------------------
# main
# -----------------------

def main():
    p = argparse.ArgumentParser("build region benchmark beds (interval-fixed)")
    p.add_argument("--genome_fasta", required=True, default="/data_processing/data/240-mammalian/hg38.ml.fa")
    p.add_argument("--gtf_dir", required=True, default="/data_processing/data/for_sampling/gtfs/")
    p.add_argument("--vista_tsv", required=True, default="data_processing/data/for_sampling/VISTA_enhancers/subsets/vista_human_positive.tsv")
    p.add_argument("--promoters_bed", required=True, default="data_processing/data/for_sampling/promoters/promoters.bed")
    p.add_argument("--utr5_bed", required=True, default="data_processing/data/for_sampling/UCSC coordinates/UCSC_5UTR_exons.bed")
    p.add_argument("--utr3_bed", required=True, default="data_processing/data/for_sampling/UCSC coordinates/UCSC_3UTR_exons.bed")
    p.add_argument("--repeats_bed", required=True, default="data_processing/data/for_sampling/repeats_hg38.bed")
    p.add_argument("--ucne_bed", required=True, default="data_processing/data/for_sampling/hg38_UCNE_coordinates.bed")
    p.add_argument("--ucne_paralogues", required=True, default="data_processing/data/for_sampling/conserved_elements/ucne_paralogues.txt")
    p.add_argument("--output_dir", required=True, default="data_processing/data/regions")
    p.add_argument("--chromosomes", nargs="+", default=["auto"])
    p.add_argument("--limit_per_category", type=int, default=10000)
    p.add_argument("--upstream_length", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_random_attempts", type=int, default=2000)
    args = p.parse_args()

    fa = Fasta(args.genome_fasta)
    canonical = set(fa.keys())

    # chrom lengths
    chrom_lengths = {c: len(fa[c]) for c in canonical}

    # chromosomes
    if len(args.chromosomes) == 1 and args.chromosomes[0] == "auto":
        chromosomes = sorted(canonical)
    else:
        chromosomes = normalize_chrom_list(args.chromosomes, canonical)

    # gtf files (by chrom)
    gtf_files_all = list_gtf_files(args.gtf_dir)
    chrom_set = set(chromosomes)
    gtf_files = []
    for fpath in gtf_files_all:
        fname = os.path.basename(fpath)
        chrom_name = fname.split(".")[0]
        if chrom_name in chrom_set:
            gtf_files.append(fpath)
    logging.info(f"using {len(gtf_files)} gtf files")

    gtf_parser = GTFParser(gtf_files)

    ucne_blocklist = load_name_blocklist(args.ucne_paralogues)

    categories = {
        # curated / non-gtf
        "repeats": load_bed_file_filtered(args.repeats_bed, "repeats", keep_chroms=chromosomes, canonical=canonical),
        "UCNE": load_bed_file_filtered(args.ucne_bed, "UCNE", keep_chroms=chromosomes, drop_names=ucne_blocklist, canonical=canonical),
        "vista_enhancer": load_vista_coordinates(args.vista_tsv, canonical=canonical, keep_chroms=chromosomes),
        "promoters": load_bed_file_filtered(args.promoters_bed, "promoters", keep_chroms=chromosomes, canonical=canonical),
        "UTR5": load_bed_file_filtered(args.utr5_bed, "UTR5", keep_chroms=chromosomes, canonical=canonical),
        "UTR3": load_bed_file_filtered(args.utr3_bed, "UTR3", keep_chroms=chromosomes, canonical=canonical),

        # gtf-derived (interval-fixed)
        "coding_regions": sample_regions_by_feature(gtf_parser, "coding_regions", chromosomes, canonical=canonical),
        "exons": sample_regions_by_feature(gtf_parser, "exons", chromosomes, canonical=canonical),
        "introns": sample_regions_by_feature(gtf_parser, "introns", chromosomes, canonical=canonical),
        "upstream_TSS": sample_regions_by_feature(gtf_parser, "upstream_TSS", chromosomes, canonical=canonical),
    }

    # non-overlap priority (specific -> broad, biologically)
    priority_order = [
        "repeats",
        "UCNE",
        "vista_enhancer",
        "promoters",
        "UTR5",
        "UTR3",
        "coding_regions",
        "exons",
        "introns",
        "upstream_TSS",
    ]

    # normalize + half-open enforce (belt+suspenders)
    for cat, rs in categories.items():
        fixed = []
        for r in rs:
            r["chrom"] = normalize_chrom(r["chrom"], canonical)
            rr = ensure_half_open(r)
            if rr is not None and rr["chrom"] in chrom_set:
                fixed.append(rr)
        categories[cat] = fixed
        logging.info(f"[SUMMARY] {cat}: {len(categories[cat])}")

    categories = ensure_nonoverlap(
        categories,
        priority_order,
        limit_per_category=args.limit_per_category,
        seed=args.seed,
    )

    # upstream pairing (same length, 2kb upstream)
    categories, upstream_categories = build_upstream_regions(
        categories,
        chrom_lengths=chrom_lengths,
        upstream_len=args.upstream_length,
    )

    # merge (functional + upstream) for writing
    all_to_write = {}
    all_to_write.update(categories)
    all_to_write.update(upstream_categories)

    # write functional + upstream
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, regs in all_to_write.items():
        write_bed(regs, out_dir, name)

    # build random sets (length-matched) using ONLY retained anchors (with pair_id)
    random_noannot, random_cat = build_random_sets(
        functional_categories=categories,
        chrom_lengths=chrom_lengths,
        seed=args.seed,
        max_attempts=args.max_random_attempts,
    )

    for name, regs in random_noannot.items():
        write_bed(regs, out_dir, name)
    for name, regs in random_cat.items():
        write_bed(regs, out_dir, name)

    # manifest
    write_manifest(str(out_dir))

if __name__ == "__main__":
    main()
