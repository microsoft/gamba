#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

max_jobs=8
for c in chr{1..22} chrX; do
  echo ">> starting $c"
  python -u /home/mica/gamba/data_processing/sample_regions.py \
    --chromosomes "$c" \
    > "logs/$c.log" 2>&1 &
  # cap concurrency at 5
  while [ "$(jobs -rp | wc -l)" -ge "$max_jobs" ]; do sleep 1; done
done

wait
echo "all chromosomes done."
