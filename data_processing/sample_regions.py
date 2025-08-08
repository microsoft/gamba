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
                    str(r.get('name') or category),  # Default to category if name missing
                    str(r.get('score', 0)),
                    r.get('strand', '.')
                ]
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
    df = pd.read_csv(tsv_path, sep='\t')
    vista_regions = []
    for _, row in df.iterrows():
        if pd.isna(row['Element Coordinates']):
            continue
        coords = row['Element Coordinates'].replace('chr', '').split(':')
        chrom = 'chr' + coords[0]
        start, end = map(int, coords[1].split('-'))
        vista_regions.append({
            'chrom': chrom,
            'start': start,
            'end': end,
            'name': row['Element ID'],
            'score': 0.0,
            'strand': '.'
        })
    return vista_regions

def sample_regions_by_feature(gtf_parser, feature_type, chromosomes):
    all_features = []
    for chrom in chromosomes:
        feats = gtf_parser.get_regions_by_type(feature_type, chrom=chrom)
        for f in feats:
            all_features.append({
                'chrom': f['chrom'],
                'start': f['start'],
                'end': f['end'],
                'name': f.get('transcript_id', f.get('gene_id', feature_type)),
                'score': 0.0,
                'strand': f.get('strand', '.')
            })
    return all_features

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

def main():
    parser = argparse.ArgumentParser("sample regions for downstream analysis")
    parser.add_argument("--bigwig_file", default="/home/mica/gamba/data_processing/data/240-mammalian/241-mammalian-2020v2.bigWig")
    parser.add_argument("--genome_fasta", default="/home/mica/gamba/data_processing/data/240-mammalian/hg38.ml.fa")
    parser.add_argument("--gtf_files", nargs='+', default=[
        '/home/mica/gamba/data_processing/data/240-mammalian/chr2.gtf',
        '/home/mica/gamba/data_processing/data/240-mammalian/chr19.gtf',
        '/home/mica/gamba/data_processing/data/240-mammalian/chr22.gtf']
    )
    parser.add_argument("--vista_tsv", default="/home/mica/gamba/data_processing/data/VISTA_enhancers/experiments.tsv")
    parser.add_argument("--utr5_bed", default="/home/mica/gamba/data_processing/data/UCSC coordinates/UCSC_5UTR_exons.bed")
    parser.add_argument("--utr3_bed", default="/home/mica/gamba/data_processing/data/UCSC coordinates/UCSC_3UTR_exons.bed")
    parser.add_argument("--promoters_bed", default="/home/mica/gamba/data_processing/data/promoters/promoters.bed")
    parser.add_argument("--output_dir", default="/home/mica/gamba/data_processing/data/regions")
    parser.add_argument("--num_regions", type=int, default=1000)
    parser.add_argument("--region_length", type=int, default=2048)
    parser.add_argument("--chromosomes", nargs='+', default=["chr2", "chr19", "chr22"])
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    # bw = pyBigWig.open(args.bigwig_file)
    # chrom = "chr2"
    # region_length = 2048
    # chrom_length = bw.chroms()[chrom]

    # for _ in range(10):
    #     start = random.randint(0, chrom_length - region_length)
    #     end = start + region_length
    #     vals = np.array(bw.values(chrom, start, end), dtype=np.float64)
        
    #     print(f"Sampled {chrom}:{start}-{end}")
    #     print("First 10 values:", vals[:10])
    #     print("Num NaNs:", np.isnan(vals).sum())
    #     print("Num valid:", (~np.isnan(vals)).sum())
    #     print()
    score_dist = get_phylop_score_ranges(args.bigwig_file, args.chromosomes, num_samples=10000)
    gtf_parser = GTFParser(args.gtf_files)
    categories = {
        "promoters": load_bed_file(args.promoters_bed, "promoters"),
        "vista_enhancer": load_vista_coordinates(args.vista_tsv),
        "UTR5": load_bed_file(args.utr5_bed, "UTR5"),
        "UTR3": load_bed_file(args.utr3_bed, "UTR3"),
        "coding_regions": sample_regions_by_feature(gtf_parser, "coding_regions", args.chromosomes),
        "noncoding_regions": sample_regions_by_feature(gtf_parser, "noncoding_regions", args.chromosomes),
        "exons": sample_regions_by_feature(gtf_parser, "exons", args.chromosomes),
        "introns": sample_regions_by_feature(gtf_parser, "introns", args.chromosomes),
        "upstream_TSS": sample_regions_by_feature(gtf_parser, "upstream_TSS", args.chromosomes),
        "start_codon": sample_regions_by_feature(gtf_parser, "start_codon", args.chromosomes),
        "stop_codon": sample_regions_by_feature(gtf_parser, "stop_codon", args.chromosomes),
        "phyloP_positive": sample_regions_by_phylop(args.bigwig_file, args.genome_fasta, score_dist["positive"], args.num_regions, args.region_length, args.chromosomes),
        "phyloP_neutral": sample_regions_by_phylop(args.bigwig_file, args.genome_fasta, score_dist["neutral"], args.num_regions, args.region_length, args.chromosomes),
        "phyloP_negative": sample_regions_by_phylop(args.bigwig_file, args.genome_fasta, score_dist["negative"], args.num_regions, args.region_length, args.chromosomes),
    }

    bw = pyBigWig.open(args.bigwig_file)
    for name, regions in categories.items():
        write_bed(regions, args.output_dir, name, bw=bw)
    bw.close()

if __name__ == "__main__":
    main()
