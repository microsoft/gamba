#!/bin/bash
# run_all_chromosomes.sh
# Processes all hg38 chromosomes in parallel and produces a
# merged PhyloP coverage summary when all jobs finish.

# ── configuration ────────────────────────────────────────────
SCRIPT="/home/mica/gamba/data_processing/generate_clean_phyloP.py"          # path to the Python script
DATA_DIR="/home/mica/gamba/data_processing/data/240-mammalian"
BIGWIG="$DATA_DIR/241-mammalian-2020v2.bigWig"
BED="$DATA_DIR/regions.bed"
GENOME="$DATA_DIR/hg38.ml.fa"
SPLITS="$DATA_DIR/splits.json"
STATS_FILE="$DATA_DIR/phylop_coverage_stats.csv"
LOG_DIR="$DATA_DIR/logs"
MAX_PARALLEL=12   # tune to your CPU/memory budget
# ─────────────────────────────────────────────────────────────

CHROMOSOMES=(
    chr1 chr2 chr3 chr4 chr5 chr6 chr7 chr8 chr9 chr10
    chr11 chr12 chr13 chr14 chr15 chr16 chr17 chr18 chr19
    chr20 chr21 chr22 chrX chrY
)

mkdir -p "$LOG_DIR"

# Remove old stats file so we start fresh (rows are appended by the script)
rm -f "$STATS_FILE"

echo "Launching chromosome jobs (max $MAX_PARALLEL in parallel)..."
echo "Logs → $LOG_DIR"
echo ""

running=0
pids=()
chroms_for_pid=()

for CHROM in "${CHROMOSOMES[@]}"; do
    LOG="$LOG_DIR/${CHROM}.log"

    python "$SCRIPT" \
        --chromosome     "$CHROM"  \
        --bigwig_file    "$BIGWIG" \
        --bed_file       "$BED"    \
        --file_path      "$DATA_DIR/" \
        --genome_fasta   "$GENOME" \
        --splits_file    "$SPLITS" \
        --stats_file     "$STATS_FILE" \
        > "$LOG" 2>&1 &

    pid=$!
    pids+=($pid)
    chroms_for_pid+=($CHROM)
    running=$((running + 1))
    echo "  Started $CHROM  (PID $pid)"

    # throttle to MAX_PARALLEL jobs at a time
    if [ "$running" -ge "$MAX_PARALLEL" ]; then
        wait "${pids[0]}"
        exit_code=$?
        if [ $exit_code -ne 0 ]; then
            echo "  ⚠️  ${chroms_for_pid[0]} failed (exit $exit_code) — check $LOG_DIR/${chroms_for_pid[0]}.log"
        fi
        pids=("${pids[@]:1}")
        chroms_for_pid=("${chroms_for_pid[@]:1}")
        running=$((running - 1))
    fi
done

# wait for any remaining jobs
echo ""
echo "Waiting for remaining jobs..."
for i in "${!pids[@]}"; do
    wait "${pids[$i]}"
    exit_code=$?
    if [ $exit_code -ne 0 ]; then
        echo "  ⚠️  ${chroms_for_pid[$i]} failed (exit $exit_code)"
    fi
done

echo ""
echo "All chromosomes processed."
echo ""

# ── pretty summary ────────────────────────────────────────────
if [ ! -f "$STATS_FILE" ]; then
    echo "No stats file found — something may have gone wrong."
    exit 1
fi

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║           PhyloP Coverage Summary (all chromosomes)         ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Use Python for the aggregation so we get totals and sorted output
python - "$STATS_FILE" <<'EOF'
import sys
import csv
from collections import defaultdict

stats_file = sys.argv[1]

rows = []
with open(stats_file) as f:
    for row in csv.DictReader(f):
        rows.append(row)

# sort by chromosome number/name
def chrom_key(r):
    c = r["chromosome"].replace("chr", "")
    return (0, int(c)) if c.isdigit() else (1, c)

rows.sort(key=chrom_key)

header = f"{'Chrom':<8} {'Split':<8} {'Total bp':>14} {'Covered bp':>14} {'Missing bp':>14} {'% Missing':>10}"
print(header)
print("─" * len(header))

total_bp = covered_bp = 0
for r in rows:
    t = int(r["total_bp"])
    c = int(r["covered_bp"])
    m = int(r["missing_bp"])
    pct = float(r["pct_missing"])
    total_bp += t
    covered_bp += c
    print(f"{r['chromosome']:<8} {r['split']:<8} {t:>14,} {c:>14,} {m:>14,} {pct:>9.2f}%")

missing_bp = total_bp - covered_bp
overall_pct = 100.0 * missing_bp / total_bp if total_bp else 0
print("─" * len(header))
print(f"{'TOTAL':<8} {'':<8} {total_bp:>14,} {covered_bp:>14,} {missing_bp:>14,} {overall_pct:>9.2f}%")
print()
print(f"Full stats saved to: {stats_file}")
EOF