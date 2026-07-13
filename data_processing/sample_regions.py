#!/usr/bin/env python3

import argparse
import os
import json
from pathlib import Path
from collections import defaultdict
import pandas as pd

import pyBigWig
from pyfaidx import Fasta
import logging
import random
from tqdm import tqdm
import numpy as np

from glob import glob


from bisect import bisect_left
import random


def find_overlapping_categories(intervals, start, end):
    """
    intervals: sorted list of (start, end, category) for a chromosome.
    returns set of categories that overlap [start, end) (half-open).
    """
    cats = set()
    if not intervals:
        return cats

    # bisect on start coordinate
    i = bisect_left(intervals, (start, end, ""))

    # walk left
    j = i - 1
    while j >= 0 and intervals[j][1] > start:
        cats.add(intervals[j][2])
        j -= 1

    # walk right
    k = i
    n = len(intervals)
    while k < n and intervals[k][0] < end:
        cats.add(intervals[k][2])
        k += 1

    return cats


def _overlaps(iv, start, end):
    # intervals are [start, end) half-open
    i = bisect_left(iv, (start, end))
    if i > 0 and iv[i-1][1] > start:   # left end > new start
        return True
    if i < len(iv) and iv[i][0] < end: # right start < new end
        return True
    return False

def _insert_merge(iv, start, end):
    # keep iv sorted, non-overlapping (do NOT merge abutting)
    i = bisect_left(iv, (start, end))
    s, e = start, end
    j = i-1
    while j >= 0 and iv[j][1] > s:     # strictly overlap
        s = min(s, iv[j][0]); e = max(e, iv[j][1]); j -= 1
    j += 1
    k = i
    while k < len(iv) and iv[k][0] < e: # strictly overlap
        s = min(s, iv[k][0]); e = max(e, iv[k][1]); k += 1
    iv[j:k] = [(s, e)]


def ensure_nonoverlap(categories, order, limit_per_category=None, seed=42):
    """
    categories: dict[str, list[dict(chrom,start,end,...)]]
    order: list[str] priority; earlier keeps more
    """
    rng = random.Random(seed)
    occupied = defaultdict(list)  # chrom -> sorted, merged intervals
    out = {}

    for cat in order:
        pool = list(categories.get(cat, []))
        rng.shuffle(pool)
        kept = []
        for r in pool:
            c = r['chrom']; s = int(r['start']); e = int(r['end'])
            if s > e: s, e = e, s
            iv = occupied[c]
            if not _overlaps(iv, s, e):
                kept.append(r)
                _insert_merge(iv, s, e)
                if limit_per_category and len(kept) >= limit_per_category:
                    break
        out[cat] = kept
        logging.info(f"[NON-OVERLAP] {cat}: kept {len(kept)} non-overlapping regions")
    return out

def build_upstream_regions(categories, chrom_lengths, upstream_len=2000):
    """
    For each region in each category, create a strand-aware upstream window.
    Both anchor and upstream receive a shared pair_id.

    Returns:
        filtered_categories: anchors WITH pair_id
        upstream_categories: upstream windows WITH pair_id
    """
    intervals_by_chrom = defaultdict(list)
    for cat, regs in categories.items():
        for r in regs:
            intervals_by_chrom[r['chrom']].append((r['start'], r['end'], cat))

    for chrom in intervals_by_chrom:
        intervals_by_chrom[chrom].sort(key=lambda x: x[0])

    filtered_categories = {cat: [] for cat in categories}
    upstream_categories = {f"{cat}_upstream": [] for cat in categories}

    pair_id = 0  # global monotonic id across all categories

    for cat, regs in categories.items():
        up_cat = f"{cat}_upstream"

        for r in regs:
            chrom = r['chrom']
            strand = r.get('strand', '.')
            start = r['start']
            end = r['end']

            # strand-aware upstream
            if strand == '-':
                us_start = end
                us_end = end + upstream_len
            else:
                us_start = start - upstream_len
                us_end = start

            if chrom not in chrom_lengths:
                continue
            if us_start < 0 or us_end > chrom_lengths[chrom]:
                continue
            if us_end <= us_start:
                continue

            # check overlap categories
            cats_over = find_overlapping_categories(intervals_by_chrom[chrom], us_start, us_end)
            if cat in cats_over:
                continue

            # assign unique pair_id to anchor + upstream
            pid = pair_id
            pair_id += 1

            anchor = dict(r)
            anchor["pair_id"] = pid
            filtered_categories[cat].append(anchor)

            upstream = {
                "chrom": chrom,
                "start": us_start,
                "end": us_end,
                "name": f"{r.get('name', cat)}_up",
                "score": 0.0,
                "strand": strand,
                "category": up_cat,
                "pair_id": pid,
            }
            upstream_categories[up_cat].append(upstream)

    for cat in filtered_categories:
        logging.info(f"[UPSTREAM] {cat}: kept {len(filtered_categories[cat])} anchors with upstreams")
    for up_cat in upstream_categories:
        logging.info(f"[UPSTREAM] {up_cat}: {len(upstream_categories[up_cat])} upstreams")

    return filtered_categories, upstream_categories


def list_gtf_files(gtf_dir):
    files = sorted(glob(os.path.join(gtf_dir, "*.gtf"))) + sorted(glob(os.path.join(gtf_dir, "*.gtf.gz")))
    if not files:
        raise FileNotFoundError(f"No GTF files found in {gtf_dir}")
    return files

def discover_chromosomes(genome_fasta, bigwig_file):
    # Prefer genome FASTA (pyfaidx) since that’s source of truth
    try:
        fa = Fasta(genome_fasta)
        chroms = list(fa.keys())
        if chroms:
            return chroms
    except Exception:
        pass
    # Fallback to bigWig
    bw = pyBigWig.open(bigwig_file)
    chroms = list(bw.chroms().keys())
    bw.close()
    if not chroms:
        raise RuntimeError("Could not discover chromosomes from FASTA or bigWig.")
    return chroms


def load_name_blocklist(txt_path):
    """
    Reads UCNE paralogue groups and returns a set of UCNE names to drop.
    Keeps exactly one UCNE from each group, removes the rest.

    Format of txt_path:
        One group per line, UCNE IDs separated by whitespace or commas.
    """
    if not os.path.exists(txt_path):
        logging.warning(f"Paralogue list not found: {txt_path}")
        return set()

    to_drop = set()
    with open(txt_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            group = [item.strip() for item in line.replace(",", " ").split() if item.strip()]
            if len(group) > 1:
                to_drop.update(group[1:])
    logging.info(f"Loaded paralogue drop list: {len(to_drop)} UCNEs will be removed.")
    return to_drop

def load_bed_file_filtered(path, category, keep_chroms=None, drop_names=None, canonical=None):
    df = pd.read_csv(path, sep='\t', header=None, comment='#')
    if df.shape[1] < 3:
        raise ValueError(f"BED file {path} must have at least 3 columns.")
    all_cols = ['chrom', 'start', 'end', 'name', 'score', 'strand',
                'thickStart', 'thickEnd', 'itemRgb', 'blockCount',
                'blockSizes', 'blockStarts']
    df.columns = all_cols[:df.shape[1]]

    if 'name' not in df.columns: df['name'] = category
    if 'score' not in df.columns: df['score'] = 0.0
    if 'strand' not in df.columns: df['strand'] = '.'

    n_before = len(df)

    if canonical is not None:
        df['chrom'] = df['chrom'].map(lambda c: normalize_chrom(str(c), canonical))

    if keep_chroms is not None:
        keep_set = set(keep_chroms)
        df = df[df['chrom'].isin(keep_set)]

    if drop_names:
        if df['name'].nunique() == 1 and next(iter(df['name'].unique())) == category:
            logging.warning(f"[{category}] BED lacks distinct names; cannot drop paralogues by name.")
        else:
            df = df[~df['name'].isin(drop_names)]

    df['start'] = df['start'].astype(int)
    df['end'] = df['end'].astype(int)

    logging.info(f"[{category}] {path}: kept {len(df)}/{n_before} rows after chrom+paralogue filters")

    return [
        {'chrom': r['chrom'], 'start': r['start'], 'end': r['end'],
         'name': r['name'], 'score': float(r['score']), 'strand': r['strand'],
         'category': category}
        for _, r in df.iterrows()
    ]


class GTFParser:
    """Parser for GTF annotation files."""

    def __init__(self, gtf_files):
        self.features = {
            'coding_regions': [],
            'noncoding_regions': [],
            'exons': [],
            'introns': [],
            'upstream_TSS': [],
            'start_codon': [],
            'stop_codon': []
        }
        self.all_intervals_by_chrom = defaultdict(list)
        self._parse_gtf_files(gtf_files)
        self._finalize_noncoding_regions()

    def _parse_attributes(self, attr_string):
        attrs = {}
        for attr in attr_string.strip().split(';'):
            if not attr.strip():
                continue
            key, value = attr.strip().split(' ', 1)
            attrs[key] = value.strip('"')
        return attrs

    def _parse_gtf_files(self, gtf_files):
        logging.info(f"Parsing {len(gtf_files)} GTF files...")

        for gtf_file in gtf_files:
            logging.info(f"Parsing GTF file: {gtf_file}")
            genes = {}
            transcripts = {}

            with open(gtf_file, 'r') as f:
                for line in tqdm(f, desc=f"Reading {os.path.basename(gtf_file)}"):
                    if line.startswith('#'):
                        continue
                    fields = line.strip().split('\t')
                    if len(fields) < 9:
                        continue
                    chrom, source, feature_type, start, end, score, strand, frame, attributes = fields
                    start, end = int(start), int(end)

                    try:
                        attrs = self._parse_attributes(attributes)
                    except:
                        continue

                    transcript_id = attrs.get('transcript_id')
                    gene_id = attrs.get('gene_id')
                    transcript_type = attrs.get('transcript_type')

                    if feature_type == 'gene':
                        gene_type = attrs.get('gene_type')
                        if gene_id and gene_type:
                            genes[gene_id] = {
                                'chrom': chrom,
                                'start': start,
                                'end': end,
                                'strand': strand,
                                'type': gene_type,
                                'transcripts': []
                            }

                    elif feature_type == 'transcript':
                        if gene_id and transcript_id and transcript_type:
                            is_canonical = 'tag "Ensembl_canonical"' in attributes or 'tag "CCDS"' in attributes
                            transcripts[transcript_id] = {
                                'gene_id': gene_id,
                                'chrom': chrom,
                                'start': start,
                                'end': end,
                                'strand': strand,
                                'type': transcript_type,
                                'is_canonical': is_canonical,
                                'exons': []
                            }
                            if gene_id in genes:
                                genes[gene_id]['transcripts'].append(transcript_id)

                    elif feature_type == 'exon' and transcript_id in transcripts:
                        transcripts[transcript_id]['exons'].append({
                            'start': start,
                            'end': end
                        })

                    elif feature_type == 'CDS' and transcript_id in transcripts:
                        transcripts[transcript_id].setdefault('cds_regions', []).append({
                            'start': start,
                            'end': end
                        })

                    elif feature_type == 'start_codon' and transcript_id:
                        self.features['start_codon'].append({
                            'chrom': chrom,
                            'start': start,
                            'end': end,
                            'strand': strand,
                            'gene_id': gene_id,
                            'transcript_id': transcript_id
                        })

                    elif feature_type == 'stop_codon' and transcript_id:
                        self.features['stop_codon'].append({
                            'chrom': chrom,
                            'start': start,
                            'end': end,
                            'strand': strand,
                            'gene_id': gene_id,
                            'transcript_id': transcript_id
                        })

            for transcript_id, transcript in transcripts.items():
                if not transcript['is_canonical']:
                    continue

                chrom = transcript['chrom']
                strand = transcript['strand']

                if 'cds_regions' in transcript and transcript['type'] == 'protein_coding':
                    for cds in transcript['cds_regions']:
                        self.features['coding_regions'].append({
                            'chrom': chrom,
                            'start': cds['start'],
                            'end': cds['end'],
                            'strand': strand,
                            'gene_id': transcript['gene_id'],
                            'transcript_id': transcript_id
                        })

                for exon in transcript['exons']:
                    self.features['exons'].append({
                        'chrom': chrom,
                        'start': exon['start'],
                        'end': exon['end'],
                        'strand': strand,
                        'gene_id': transcript['gene_id'],
                        'transcript_id': transcript_id
                    })

                if len(transcript['exons']) >= 2:
                    sorted_exons = sorted(transcript['exons'], key=lambda x: x['start'])
                    for i in range(len(sorted_exons) - 1):
                        intron_start = sorted_exons[i]['end'] + 1
                        intron_end = sorted_exons[i + 1]['start'] - 1
                        if intron_end > intron_start:
                            self.features['introns'].append({
                                'chrom': chrom,
                                'start': intron_start,
                                'end': intron_end,
                                'strand': strand,
                                'gene_id': transcript['gene_id'],
                                'transcript_id': transcript_id
                            })

                if strand == '+':
                    upstream_start = max(1, transcript['start'] - 2000)
                    upstream_end = transcript['start'] - 1
                else:
                    upstream_start = transcript['end'] + 1
                    upstream_end = transcript['end'] + 2000

                if upstream_end > upstream_start:
                    self.features['upstream_TSS'].append({
                        'chrom': chrom,
                        'start': upstream_start,
                        'end': upstream_end,
                        'strand': strand,
                        'gene_id': transcript['gene_id'],
                        'transcript_id': transcript_id
                    })

                # Track all annotated regions for subtraction later
                for feat in transcript.get('cds_regions', []) + transcript['exons']:
                    self.all_intervals_by_chrom[chrom].append((feat['start'], feat['end']))
                for feat in self.features['start_codon'] + self.features['stop_codon']:
                    if feat['chrom'] == chrom:
                        self.all_intervals_by_chrom[chrom].append((feat['start'], feat['end']))

    def _finalize_noncoding_regions(self, max_noncoding=10000):
        """Infer noncoding regions as parts of the genome not covered by any annotated features."""
        logging.info("Constructing noncoding regions...")
        all_noncoding = []

        for chrom, intervals in self.all_intervals_by_chrom.items():
            if not intervals:
                continue

            merged = sorted(intervals)
            merged_intervals = []
            current_start, current_end = merged[0]
            for start, end in merged[1:]:
                if start <= current_end:
                    current_end = max(current_end, end)
                else:
                    merged_intervals.append((current_start, current_end))
                    current_start, current_end = start, end
            merged_intervals.append((current_start, current_end))

            chrom_max = max(end for _, end in merged_intervals) + 2000
            last_end = 1
            for start, end in merged_intervals:
                if start > last_end:
                    all_noncoding.append({
                        'chrom': chrom,
                        'start': last_end,
                        'end': start - 1,
                        'strand': '.',
                        'gene_id': None,
                        'transcript_id': None
                    })
                last_end = end + 1
            if last_end < chrom_max:
                all_noncoding.append({
                    'chrom': chrom,
                    'start': last_end,
                    'end': chrom_max,
                    'strand': '.',
                    'gene_id': None,
                    'transcript_id': None
                })

        logging.info(f"Total candidate noncoding regions: {len(all_noncoding)}")

        # Random sample up to the desired number
        if len(all_noncoding) > max_noncoding:
            all_noncoding = random.sample(all_noncoding, max_noncoding)

        self.features['noncoding_regions'].extend(all_noncoding)
        logging.info(f"Selected {len(self.features['noncoding_regions'])} noncoding regions")

    def get_regions_by_type(self, feature_type, chrom=None):
        """
        Return regions of a given feature type, optionally filtered by chromosome.

        Valid feature types include:
            - 'exons'
            - 'introns'
            - 'noncoding_regions'
            - 'coding_regions'
            - 'upstream_TSS'
            - 'start_codon'
            - 'stop_codon'

        Args:
            feature_type: string name of the feature
            chrom: optional string name of chromosome to filter

        Returns:
            List of dicts with at least keys: chrom, start, end, strand
        """
        if feature_type not in self.features:
            raise ValueError(f"Unknown feature type '{feature_type}'")

        regions = self.features[feature_type]
        if chrom:
            return [r for r in regions if r["chrom"] == chrom]
        return regions

def get_bigwig_values(bw, chrom, start, end):
    """
    Get values from bigWig file using intervals method.
    
    Args:
        bw: pyBigWig object
        chrom: Chromosome name
        start: Start position (0-based)
        end: End position (exclusive)
        
    Returns:
        numpy array of values
    """
    # Initialize vals with zeros
    vals = np.zeros(end - start, dtype=np.float64)
    
    try:
        # Get intervals from the bigwig file
        intervals = bw.intervals(chrom, start, end)
        
        # Check if intervals is None
        if intervals is None:
            # Return zeros if no intervals found
            return vals
        
        # Fill in values from intervals
        for interval_start, interval_end, value in intervals:
            rel_start = max(0, interval_start - start)
            rel_end = min(end - start, interval_end - start)
            vals[rel_start:rel_end] = value
            
        return vals
    except Exception as e:
        logging.debug(f"Error getting values for {chrom}:{start}-{end}: {e}")
        return vals
    

def get_phylop_score_ranges(bigwig_file, chromosomes, num_samples=10000, region_length=1000):
    """
    Get ranges of phyloP scores by sampling random regions.
    
    Args:
        bigwig_file: Path to bigwig file with phyloP scores
        chromosomes: List of chromosomes to sample from
        num_samples: Number of regions to sample
        region_length: Length of each region
    
    Returns:
        Dictionary with information about phyloP score distribution
    """
    logging.info(f"Analyzing phyloP score distribution from {num_samples} random samples...")
    
    bw = pyBigWig.open(bigwig_file)
    
    # Verify the chromosomes exist in the bigwig file
    valid_chroms = []
    for chrom in chromosomes:
        if chrom in bw.chroms():
            valid_chroms.append(chrom)
        else:
            logging.warning(f"Chromosome {chrom} not found in bigwig file")

    print("Valid chromosomes found:", valid_chroms)
    
    if not valid_chroms:
        logging.error("No valid chromosomes found in bigwig file")
        raise ValueError("No valid chromosomes found in bigwig file")
    
    all_scores = []

    for _ in tqdm(range(num_samples), desc="Sampling phyloP scores"):
        chrom = random.choice(valid_chroms)
        try:
            chrom_length = bw.chroms()[chrom]
            if chrom_length <= region_length:
                continue
            
            # Define start and end positions
            start = random.randint(0, chrom_length - region_length)
            end = start + region_length
            
            # Initialize vals with zeros
            vals = np.zeros(end - start, dtype=np.float64)

            # Get the conservation scores from the bigwig file
            intervals = bw.intervals(chrom, start, end)
            # Check if intervals is None
            if intervals is None:
                print("Error: intervals is None")
                # skip this region
                continue
            else:
                for interval_start, interval_end, value in intervals:
                    vals[interval_start - start : interval_end - start] = value
                    # Get to 2 decimal places
                    vals = np.round(vals, 2)
            
            # Filter valid scores
            valid_scores = vals[~np.isnan(vals)]
            if len(valid_scores) > 0:
                all_scores.extend(valid_scores)
        except Exception as e:
            logging.debug(f"Error sampling scores from {chrom}: {e}")
            continue
    
    # Check if we have any valid scores
    if not all_scores:
        logging.error("No valid scores found during sampling")
        raise ValueError("No valid phyloP scores could be sampled from the provided chromosomes")
    
    all_scores = np.array(all_scores)
    
    # Calculate percentiles
    percentiles = {
        'min': np.min(all_scores),
        'p1': np.percentile(all_scores, 1),
        'p5': np.percentile(all_scores, 5),
        'p25': np.percentile(all_scores, 25),
        'median': np.median(all_scores),
        'p75': np.percentile(all_scores, 75),
        'p95': np.percentile(all_scores, 95),
        'p99': np.percentile(all_scores, 99),
        'max': np.max(all_scores)
    }
    
    # Add p45 and p55 for defining "neutral" range
    percentiles['p45'] = np.percentile(all_scores, 45)
    percentiles['p55'] = np.percentile(all_scores, 55)
    
    bw.close()
    
    logging.info(f"PhyloP score distribution: {percentiles}")
    
    return {
        'negative': (percentiles['min'], percentiles['p5']),
        'neutral': (percentiles['p45'], percentiles['p55']),
        'positive': (percentiles['p95'], percentiles['max']),
        'all_scores': all_scores,
        'percentiles': percentiles
    }

def write_bed(regions, output_dir, category, bw=None):
    """
    Write regions to BED files split by chromosome.

    Args:
        regions: List of region dictionaries.
        output_dir: Root output directory for BED files.
        category: Category name (e.g. 'coding_regions').
        bw: Optional pyBigWig object for filtering by phyloP if needed.
    """
    from pathlib import Path
    import os
    import logging

    os.makedirs(output_dir, exist_ok=True)
    output_path = Path(output_dir) / category
    output_path.mkdir(parents=True, exist_ok=True)

    by_chrom = {}
    for r in regions:
        chrom = r["chrom"]
        by_chrom.setdefault(chrom, []).append(r)

    for chrom, chrom_regions in by_chrom.items():
        out_file = output_path / f"{chrom}.bed"
        with open(out_file, "w") as f:
            for r in chrom_regions:
                fields = [
                    r['chrom'],
                    str(r['start']),
                    str(r['end']),
                    str(r.get('name') or category),
                    str(r.get('score', 0)),
                    r.get('strand', '.'),
                ]

                # optional: append pair_id
                if 'pair_id' in r:
                    fields.append(str(r['pair_id']))

                f.write("\t".join(fields) + "\n")
        logging.info(f"Saved {len(chrom_regions)} regions to {out_file}")

def load_bed_file(path, category):
    df = pd.read_csv(path, sep='\t', header=None, comment='#')

    # Debug print to check column count
    print(f"[DEBUG] BED file {path} columns: {df.shape[1]} - shape: {df.shape}")

    if df.shape[1] < 3:
        raise ValueError(f"BED file {path} has fewer than 3 columns. Must contain at least chrom, start, end.")

    # Assign column names based on standard BED field order (BED3 to BED12)
    all_cols = ['chrom', 'start', 'end', 'name', 'score', 'strand',
                'thickStart', 'thickEnd', 'itemRgb', 'blockCount',
                'blockSizes', 'blockStarts']
    df.columns = all_cols[:df.shape[1]]

    # Fill missing optional columns
    if 'name' not in df.columns:
        df['name'] = category
    if 'score' not in df.columns:
        df['score'] = 0.0
    if 'strand' not in df.columns:
        df['strand'] = '.'

    # Convert key fields
    df['start'] = df['start'].astype(int)
    df['end'] = df['end'].astype(int)

    return [
        {
            'chrom': row['chrom'],
            'start': row['start'],
            'end': row['end'],
            'name': row['name'],
            'score': float(row['score']),
            'strand': row['strand'],
            'category': category
        }
        for _, row in df.iterrows()
    ]
    

def load_vista_coordinates(tsv_path):
    """
    load vista enhancers from a TSV that looks like:

        ... vista_id ... coord ... coordinate_hg38 ... strand ...

    we’ll prefer `coordinate_hg38` if present, otherwise fall back to `coord`.
    name will be `vista_id` if available.
    """
    df = pd.read_csv(tsv_path, sep="\t")

    print(f"[VISTA] columns in {tsv_path}: {list(df.columns)}")

    # pick coord column
    if "coordinate_hg38" in df.columns:
        coord_col = "coordinate_hg38"
    elif "coord" in df.columns:
        coord_col = "coord"
    else:
        raise ValueError(
            f"could not find a coordinate column ('coordinate_hg38' or 'coord') "
            f"in {tsv_path}. got: {list(df.columns)}"
        )

    # pick id column
    if "vista_id" in df.columns:
        id_col = "vista_id"
    else:
        # fall back to first column if vista_id somehow missing
        id_col = df.columns[0]
        logging.warning(
            f"[VISTA] no 'vista_id' column; using '{id_col}' as id"
        )

    # optional strand column
    strand_col = "strand" if "strand" in df.columns else None

    vista_regions = []
    for _, row in df.iterrows():
        coords = row[coord_col]
        if pd.isna(coords):
            continue

        # coords like "chr1:3274017-3274864"
        coords = str(coords)
        chrom_part, pos_part = coords.split(":")
        chrom = chrom_part if chrom_part.startswith("chr") else "chr" + chrom_part
        start_str, end_str = pos_part.split("-")
        start = int(start_str)
        end = int(end_str)

        vista_regions.append({
            "chrom": chrom,
            "start": start,
            "end": end,
            "name": row[id_col],
            "score": 0.0,
            "strand": row[strand_col] if strand_col else ".",
        })

    logging.info(f"[vista_enhancer] loaded {len(vista_regions)} regions from {tsv_path}")
    return vista_regions



def sample_regions_by_feature(gtf_parser, feature_type, chromosomes, canonical=None):
    chrom_set = set(chromosomes)
    out = []
    for chrom in chromosomes:
        feats = gtf_parser.get_regions_by_type(feature_type, chrom=chrom)
        for f in feats:
            c = f['chrom']
            if canonical is not None:
                c = normalize_chrom(c, canonical)
            if c not in chrom_set:
                continue
            out.append({
                'chrom': c,
                'start': f['start'],
                'end': f['end'],
                'name': f.get('transcript_id', f.get('gene_id', feature_type)),
                'score': 0.0,
                'strand': f.get('strand', '.')
            })
    logging.info(f"[{feature_type}] kept {len(out)} rows")
    return out

def sample_regions_by_phylop(bigwig_file, genome_fasta, score_range, num_regions, max_length, chromosomes):
    genome = Fasta(genome_fasta)
    bw = pyBigWig.open(bigwig_file)

    valid_chroms = [c for c in chromosomes if c in genome.keys() and c in bw.chroms()]
    regions = []
    attempts = 0
    max_attempts = num_regions * 100
    min_score, max_score = score_range

    with tqdm(total=num_regions, desc=f"PhyloP {min_score:.2f}-{max_score:.2f}") as pbar:
        while len(regions) < num_regions and attempts < max_attempts:
            attempts += 1
            chrom = random.choice(valid_chroms)
            chrom_len = len(genome[chrom])
            if chrom_len <= max_length:
                continue
            start = random.randint(0, chrom_len - max_length)
            end = start + max_length
            try:
                scores = get_bigwig_values(bw, chrom, start, end)
                valid = scores[scores != 0]
                if len(valid) == 0:
                    continue
                mean_score = valid.mean()
                if min_score <= mean_score <= max_score:
                    regions.append({
                        'chrom': chrom,
                        'start': start,
                        'end': end,
                        'name': f"phylop_{mean_score:.2f}",
                        'score': float(mean_score),
                        'strand': '.'
                    })
                    pbar.update(1)
            except:
                continue
    bw.close()
    return regions


def build_canonical_set(genome_fasta, bigwig_file):
    # Prefer FASTA keys as canonical
    try:
        fa = Fasta(genome_fasta)
        return set(fa.keys())
    except Exception:
        pass
    bw = pyBigWig.open(bigwig_file)
    s = set(bw.chroms().keys())
    bw.close()
    return s

def normalize_chrom(chrom, canonical):
    """
    Map common variants to canonical names:
      '1' <-> 'chr1', 'X' <-> 'chrX', 'Y' <-> 'chrY', 'MT'/'M' <-> 'chrM'
    If no match, return original.
    """
    if chrom in canonical:
        return chrom

    # Try adding/removing 'chr'
    if chrom.startswith("chr"):
        nochr = chrom[3:]
        if nochr in canonical:
            return nochr
    else:
        withchr = "chr" + chrom
        if withchr in canonical:
            return withchr

    # Mito edge cases
    mito_aliases = {"M", "MT", "chrM", "chrMT"}
    if chrom in mito_aliases:
        for cand in ("chrM", "MT", "M"):
            if cand in canonical:
                return cand

    # No mapping found
    return chrom

def normalize_chrom_list(chroms, canonical):
    return [normalize_chrom(c, canonical) for c in chroms]

def limit_regions(regions, n, seed=None):
    if n is None or n <= 0 or len(regions) <= n:
        return regions
    rng = random.Random(seed)
    return rng.sample(regions, n)


def main():
    parser = argparse.ArgumentParser("sample regions for downstream analysis")
    parser.add_argument("--bigwig_file", default="/home/mica/scratch/gamba/data_processing/data/240-mammalian/241-mammalian-2020v2.bigWig")
    parser.add_argument("--genome_fasta", default="/home/mica/scratch/gamba/data_processing/data/240-mammalian/hg38.ml.fa")
    parser.add_argument("--vista_tsv", default="/home/mica/scratch/gamba/data_processing/data/for_sampling/VISTA_enhancers/subsets/vista_human_positive.tsv")
    parser.add_argument("--utr5_bed", default="/home/mica/scratch/gamba/data_processing/data/for_sampling/UCSC coordinates/UCSC_5UTR_exons.bed")
    parser.add_argument("--utr3_bed", default="/home/mica/scratch/gamba/data_processing/data/for_sampling/UCSC coordinates/UCSC_3UTR_exons.bed")
    parser.add_argument("--promoters_bed", default="/home/mica/scratch/gamba/data_processing/data/for_sampling/promoters/promoters.bed")
    parser.add_argument("--output_dir", default="/home/mica/scratch/gamba/data_processing/data/regions")
    parser.add_argument("--num_regions", type=int, default=10000)
    parser.add_argument("--region_length", type=int, default=2048)
    parser.add_argument("--gtf_dir", default="/home/mica/scratch/gamba/data_processing/data/for_sampling/gtfs/")
    parser.add_argument("--chromosomes", nargs='+', default=["auto"])
    parser.add_argument("--repeats_bed", default="/home/mica/scratch/gamba/data_processing/data/for_sampling/repeats_hg38.bed")
    parser.add_argument("--ucne_bed", default="/home/mica/scratch/gamba/data_processing/data/for_sampling/hg38_UCNE_coordinates.bed")
    parser.add_argument("--ucne_paralogues", default="/home/mica/scratch/gamba/data_processing/data/for_sampling/conserved_elements/ucne_paralogues.txt")
    parser.add_argument("--limit_per_category", type=int, default=10000,  # keep only N items per category
                    help="If set, randomly keep at most N regions per category (after building).")
    parser.add_argument(
        "--upstream_length",
        type=int,
        default=2000,
        help="Length of upstream window to sample for each region."
    )
    parser.add_argument("--phylop_num_samples", type=int, default=10000,
                        help="Number of random samples to estimate phyloP percentiles (use a small number for tests).")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()


    logging.basicConfig(level=logging.INFO)

    canonical = build_canonical_set(args.genome_fasta, args.bigwig_file)

    # build chrom length map for bounds checking
    fa = Fasta(args.genome_fasta)
    chrom_lengths = {}
    for c in canonical:
        try:
            chrom_lengths[c] = len(fa[c])
        except KeyError:
            # some canonical names may be from bigWig only
            continue

    # Discover chromosomes (auto) or normalize provided list
    if len(args.chromosomes) == 1 and args.chromosomes[0] == "auto":
        chromosomes = sorted(canonical)
    else:
        chromosomes = normalize_chrom_list(args.chromosomes, canonical)

    logging.info(f"Canonical chromosomes in use (sample): {chromosomes[:6]} … total={len(chromosomes)}")

    # 2) Expand GTFs from directory but only keep files for requested chromosomes
    gtf_files_all = list_gtf_files(args.gtf_dir)
    chrom_set = set(chromosomes)
    gtf_files = []

    for f in gtf_files_all:
        fname = os.path.basename(f)
        # Match chr name exactly (with or without .gtf/.gtf.gz extension)
        chrom_name = fname.split(".")[0]  # e.g., "chr22" from "chr22.gtf"
        if chrom_name in chrom_set:
            gtf_files.append(f)

    logging.info(f"Using {len(gtf_files)} GTF files for requested chromosomes: {chromosomes}")


    score_dist = get_phylop_score_ranges(args.bigwig_file, chromosomes, num_samples=args.phylop_num_samples)
    gtf_parser = GTFParser(gtf_files)
    # Build paralogue blocklist for UCNE filtering
    ucne_blocklist = load_name_blocklist(args.ucne_paralogues)

    vista = load_vista_coordinates(args.vista_tsv)
    if canonical is not None:
        for r in vista:
            r['chrom'] = normalize_chrom(r['chrom'], canonical)
    vista = [r for r in vista if r['chrom'] in set(chromosomes)]
    logging.info(f"[vista_enhancer] kept {len(vista)} rows after chrom filter")

    categories = {
        # Existing categories
        "promoters": load_bed_file_filtered(
            args.promoters_bed, "promoters", keep_chroms=chromosomes, canonical=canonical
        ),
        "vista_enhancer": vista,
        "UTR5": load_bed_file_filtered(
            args.utr5_bed, "UTR5", keep_chroms=chromosomes, canonical=canonical
        ),
        "UTR3": load_bed_file_filtered(
            args.utr3_bed, "UTR3", keep_chroms=chromosomes, canonical=canonical
        ),
        "coding_regions": sample_regions_by_feature(
            gtf_parser, "coding_regions", chromosomes,  canonical=canonical
        ),
        "noncoding_regions": sample_regions_by_feature(
            gtf_parser, "noncoding_regions", chromosomes, canonical=canonical
        ),
        "exons": sample_regions_by_feature(
            gtf_parser, "exons", chromosomes, canonical=canonical
        ),
        "introns": sample_regions_by_feature(
            gtf_parser, "introns", chromosomes, canonical=canonical
        ),
        "upstream_TSS": sample_regions_by_feature(
            gtf_parser, "upstream_TSS", chromosomes, canonical=canonical
        ),
        "start_codon": sample_regions_by_feature(
            gtf_parser, "start_codon", chromosomes, canonical=canonical
        ),
        "stop_codon": sample_regions_by_feature(
            gtf_parser, "stop_codon", chromosomes, canonical=canonical
        ),

        # NEW: Low complexity repeats
        "repeats": load_bed_file_filtered(
            args.repeats_bed, "repeats", keep_chroms=chromosomes, canonical=canonical
        ),

        # NEW: UCNEs (remove paralogues)
        "UCNE": load_bed_file_filtered(
            args.ucne_bed, "UCNE", keep_chroms=chromosomes, drop_names=ucne_blocklist, canonical=canonical
        ),

        # PhyloP categories
        "phyloP_positive": sample_regions_by_phylop(
            args.bigwig_file, args.genome_fasta,
            score_dist["positive"], args.num_regions, args.region_length, chromosomes
        ),
        "phyloP_neutral": sample_regions_by_phylop(
            args.bigwig_file, args.genome_fasta,
            score_dist["neutral"], args.num_regions, args.region_length, chromosomes
        ),
        "phyloP_negative": sample_regions_by_phylop(
            args.bigwig_file, args.genome_fasta,
            score_dist["negative"], args.num_regions, args.region_length, chromosomes
        ),
    }


    for k,v in categories.items():
        logging.info(f"[SUMMARY] {k}: {len(v)} regions")

    # priority: specific files first, then phyloP bins, then GTF-derived last
    priority_order = [
        "repeats", "UCNE", "vista_enhancer", "promoters", "UTR5", "UTR3",
        "phyloP_positive", "phyloP_neutral", "phyloP_negative",
        "coding_regions", "exons", "introns", "upstream_TSS", "noncoding_regions",
        "start_codon", "stop_codon"
    ]

    # optional per-class cap
    cap = args.limit_per_category

    # enforce half-open everywhere
    for cat, rs in categories.items():
        for r in rs:
            r['start'] = int(r['start'])
            r['end']   = int(r['end'])
            if r['end'] < r['start']:
                r['start'], r['end'] = r['end'], r['start']


    categories = ensure_nonoverlap(
        categories, priority_order,
        limit_per_category=cap,
        seed=args.seed
    )

    # build upstream regions and filter anchors to only those with valid upstreams
    categories, upstream_categories = build_upstream_regions(
        categories,
        chrom_lengths=chrom_lengths,
        upstream_len=args.upstream_length,
    )

    # merge anchor and upstream categories for writing
    all_categories = {}
    all_categories.update(categories)
    all_categories.update(upstream_categories)

    bw = pyBigWig.open(args.bigwig_file)
    for name, regions in all_categories.items():
        write_bed(regions, args.output_dir, name, bw=bw)
    bw.close()


if __name__ == "__main__":
    main()
