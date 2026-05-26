# gamba

Genomic language models (gLMs) with evolutionary conservation supervision.

`gamba` contains genome language models trained on human DNA sequence together with Zoonomia 241-mammalian phyloP conservation scores. The released checkpoints include autoregressive ArGamba models and bidirectional BiGamba models trained with sequence-only, conservation-only, or dual sequence-plus-conservation objectives.

## Quick start: load released models from Hugging Face

The released checkpoints can be loaded directly with Hugging Face `transformers`.

```python
import torch
from transformers import AutoModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

REPO_ID = "micanonsens/bigamba-dual-step44000"

model = AutoModel.from_pretrained(
    REPO_ID,
    trust_remote_code=True,
).eval().to(DEVICE)

print(f"Loaded {REPO_ID} on {DEVICE}")
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

Example repositories:

| Checkpoint name | Architecture | Training task |
|---|---|---|
| [ArGamba-dual](https://huggingface.co/micanonsens/argamba-dual-step44000) | ArGamba (Jamba autoregressive) | NTP + CEP |
| [ArGamba-seq_only](https://huggingface.co/micanonsens/argamba-seq_only-step44000) | ArGamba (Jamba autoregressive) | NTP |
| [ArGamba-cons_only](https://huggingface.co/micanonsens/argamba-cons_only-step44000) | ArGamba (Jamba autoregressive) | CEP |
| [BiGamba-dual](https://huggingface.co/micanonsens/bigamba-dual-step44000) | BiGamba (Mamba bidirectional) | MLM + MEM |
| [BiGamba-seq_only](https://huggingface.co/micanonsens/bigamba-seq_only-step44000) | BiGamba (Mamba bidirectional) | MLM |
| [BiGamba-cons_only](https://huggingface.co/micanonsens/bigamba-cons_only-step44000) | BiGamba (Mamba bidirectional) | MEM |

## Colab notebook

A Colab notebook (`src/gamba_notebook.ipynb`) is provided for loading the released Hugging Face models, scoring genomic intervals, comparing predictions with phyloP from a bigWig file, and exporting predictions as bedGraph files.

Recommended notebook workflow:

1. Install the environment dependencies.
2. Restart the Colab runtime once after installation.
3. Load ArGamba and/or BiGamba from Hugging Face.
4. Upload or define BED regions.
5. Run tiled conservation prediction over those regions.
6. Optionally compare predictions to true phyloP values from a bigWig file.
7. Export predictions as bedGraph files.

For long regions, the notebook tiles windows differently for autoregressive and bidirectional models:

```text
ArGamba / causal:
  [upstream context | scored positions]

BiGamba / bidirectional:
  [left context | scored positions | right context]
```

Context-only positions are discarded to reduce edge effects.

## Installation

Clone the repository and navigate to the project directory:

```bash
git clone ...
cd gamba/
```

Install dependencies in your preferred Python environment. The exact CUDA/PyTorch/Mamba versions may depend on your system. The released Hugging Face models require `trust_remote_code=True`.

Core dependencies include:

```bash
pip install torch transformers safetensors huggingface_hub accelerate einops
pip install mamba-ssm causal-conv1d
pip install pyfaidx pyBigWig pandas numpy scipy scikit-learn matplotlib seaborn tqdm
pip install evodiff
```

On managed GPU systems, install PyTorch and CUDA-compatible packages according to the cluster or Colab runtime.

## Data preparation

To set up the main genome/phyloP data:

```bash
mkdir -p data_processing/data/240-mammalian/

# Download human chromosome sizes.
curl https://hgdownload.cse.ucsc.edu/goldenpath/hg38/bigZips/hg38.chrom.sizes \
  > data_processing/data/240-mammalian/hg38.chrom.sizes

python data_processing/generate_human_bed.py

# Download full human genome FASTA.
curl https://storage.googleapis.com/basenji_barnyard2/hg38.ml.fa.gz \
  > data_processing/data/240-mammalian/hg38.ml.fa.gz

gunzip data_processing/data/240-mammalian/hg38.ml.fa.gz

# Download centromere locations.
curl https://hgdownload.soe.ucsc.edu/goldenPath/hg38/database/centromeres.txt.gz \
  > data_processing/data/240-mammalian/centromeres.txt.gz

gunzip data_processing/data/240-mammalian/centromeres.txt.gz

# Download repeat locations from the UCSC Genome Browser RepeatMasker track.
# Save the file as data_processing/data/240-mammalian/repeats_hg38.bed.gz.
gunzip data_processing/data/240-mammalian/repeats_hg38.bed.gz

# Download Zoonomia 241-mammalian phyloP scores.
curl https://cgl.gi.ucsc.edu/data/cactus/241-mammalian-2020v2-hub/Homo_sapiens/241-mammalian-2020v2.bigWig \
  > data_processing/data/240-mammalian/241-mammalian-2020v2.bigWig
```

Create chromosome splits:

```bash
cat > data_processing/data/240-mammalian/splits.json <<'EOF'
{
  "train": [
    "1", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15",
    "17", "18", "19", "20", "21", "X"
  ],
  "valid": [
    "3", "16"
  ],
  "test": [
    "2", "22"
  ]
}
EOF
```

Generate exclusion regions and clean phyloP arrays:

```bash
python data_processing/exclusion_regions.py

for chrom in {1..22} X; do
  echo "Running chr${chrom}"
  python data_processing/generate_clean_phyloP.py --chromosome "chr${chrom}"
done
```

Optionally generate FASTA files from the same cleaned regions:

```bash
python data_processing/generate_same_data_fasta.py
```

Uncompress `.npz` files and verify chromosome sizes:

```bash
python data_processing/uncompress_npz.py --type "small"
python assert_chromosome_sizes.py --type "small"
```

Expected structure:

```text
data_processing/data/240-mammalian/
├── train/
│   ├── 1_conservation_small.npy
│   ├── 1_sequence_small.npy
│   ├── 1.npz
│   └── ...
├── valid/
│   ├── 3_conservation_small.npy
│   ├── 3_sequence_small.npy
│   ├── 3.npz
│   ├── 16_conservation_small.npy
│   ├── 16_sequence_small.npy
│   └── 16.npz
└── test/
    ├── 2_conservation_small.npy
    ├── 2_sequence_small.npy
    ├── 2.npz
    ├── 22_conservation_small.npy
    ├── 22_sequence_small.npy
    └── 22.npz
```

Run a basic data sanity check:

```bash
python src/test_sequence.py
```

## Training

Training scripts load a JSON experiment config, construct the corresponding model/task wrapper, build `ConservationDataset` dataloaders, and save checkpoint directories during training.

All released Gamba checkpoints were trained with scripts in `src/` using:

```bash
configs/jamba-small-240mammalian.json
```

The training scripts support a shared command-line interface:

```text
out_fpath           Output/checkpoint directory. Optional positional argument.
data_root           Root directory containing prepared data. Optional positional argument.
--config_fpath      Experiment config JSON.
--mini_run          Run on a small subset for debugging.
--checkpoint_freq   Save/validate every N steps.
--random_seed       Random seed.
--run_type          train or test.
--dtype             float32, float16, or bfloat16.
--verbose           Verbose logging.
--no_wandb          Disable Weights & Biases logging.
--last_step         Resume from latest checkpoint (-1) or a specific checkpoint step.
```

The default config path is:

```bash
configs/jamba-small-240mammalian.json
```

In the examples below, `data_processing/data/` is used as the data root. The training scripts append the dataset name from the config, e.g. `240-mammalian`.

Note: several BiGamba training scripts retain historical `caduceus_train` filenames.

### Training script mapping

| Model family | Task | Script | Config |
|---|---|---|---|
| ArGamba | dual: NTP + CEP | `src/test_train.py` | `configs/jamba-small-240mammalian.json` |
| ArGamba | sequence-only: NTP | `src/test_train_noPHYLOP.py` | `configs/jamba-small-240mammalian.json` |
| ArGamba | conservation-only: CEP | `src/test_train_noALM.py` | `configs/jamba-small-240mammalian.json` |
| BiGamba | dual: MLM + MEM | `src/caduceus_train.py` | `configs/jamba-small-240mammalian.json` |
| BiGamba | sequence-only: MLM | `src/caduceus_train_noPHYLOP.py` | `configs/jamba-small-240mammalian.json` |
| BiGamba | conservation-only: MEM | `src/caduceus_train_noMLM.py` | `configs/jamba-small-240mammalian.json` |

### Mini run

Use `--mini_run` to verify that the environment, data paths, model construction, and checkpoint writing work before launching a full run.

Example for ArGamba dual:

```bash
python src/test_train.py \
  checkpoints/argamba-dual \
  data_processing/data \
  --config_fpath configs/jamba-small-240mammalian.json \
  --mini_run \
  --checkpoint_freq 100 \
  --dtype bfloat16 \
  --no_wandb
```

Example for BiGamba dual:

```bash
python src/caduceus_train.py \
  checkpoints/bigamba-dual \
  data_processing/data \
  --config_fpath configs/jamba-small-240mammalian.json \
  --mini_run \
  --checkpoint_freq 100 \
  --dtype bfloat16 \
  --no_wandb
```

### Full training examples

ArGamba dual:

```bash
python src/test_train.py \
  checkpoints/argamba-dual \
  data_processing/data \
  --config_fpath configs/jamba-small-240mammalian.json \
  --checkpoint_freq 2000 \
  --dtype bfloat16
```

ArGamba sequence-only:

```bash
python src/test_train_noPHYLOP.py \
  checkpoints/argamba-seq-only \
  data_processing/data \
  --config_fpath configs/jamba-small-240mammalian.json \
  --checkpoint_freq 2000 \
  --dtype bfloat16
```

ArGamba conservation-only:

```bash
python src/test_train_noALM.py \
  checkpoints/argamba-cons-only \
  data_processing/data \
  --config_fpath configs/jamba-small-240mammalian.json \
  --checkpoint_freq 2000 \
  --dtype bfloat16
```

BiGamba dual:

```bash
python src/caduceus_train.py \
  checkpoints/bigamba-dual \
  data_processing/data \
  --config_fpath configs/jamba-small-240mammalian.json \
  --checkpoint_freq 2000 \
  --dtype bfloat16
```

BiGamba sequence-only:

```bash
python src/caduceus_train_noPHYLOP.py \
  checkpoints/bigamba-seq-only \
  data_processing/data \
  --config_fpath configs/jamba-small-240mammalian.json \
  --checkpoint_freq 2000 \
  --dtype bfloat16
```

BiGamba conservation-only:

```bash
python src/caduceus_train_noMLM.py \
  checkpoints/bigamba-cons-only \
  data_processing/data \
  --config_fpath configs/jamba-small-240mammalian.json \
  --checkpoint_freq 2000 \
  --dtype bfloat16
```

### Resume training

Resume from the latest checkpoint in the output directory:

```bash
python src/test_train.py \
  checkpoints/argamba-dual \
  data_processing/data \
  --config_fpath configs/jamba-small-240mammalian.json \
  --last_step -1
```

Resume from a specific checkpoint step:

```bash
python src/test_train.py \
  checkpoints/argamba-dual \
  data_processing/data \
  --config_fpath configs/jamba-small-240mammalian.json \
  --last_step 44000
```


### Checkpoint format

Training writes checkpoint directories into `out_fpath`. Most scripts save checkpoints as `dcp_<step>/`; some task-specific scripts may use a different prefix.

Checkpoints contain model weights, optimizer state, scheduler state, current epoch, step count, token count, and sequence count.


## Downstream evaluation data and benchmarks

After preparing the main genome/phyloP training data, additional scripts can be used to generate downstream evaluation regions and run representation-level evaluations.

### 1. Generate genomic region BED files for testing

The script `data_processing/sample_regions.py` creates BED files for biologically defined genomic categories used in downstream evaluation.

Some small region annotation files are included in:

```text
data_processing/region_info/
```

Currently included:

```text
data_processing/region_info/
├── experiments.tsv
├── hg38_UCNE_coordinates.bed
├── promoters.bed
└── ucne_paralogues.txt
```

These files are small enough to keep in the repository. See the paper for details on how these annotation files were derived.

Large annotation files are **not included** in the repository and should be downloaded or generated locally:

```text
data_processing/region_info/
├── repeats_hg38.bed
├── UCSC_3UTR_exons.bed
└── UCSC_5UTR_exons.bed
```

The following inputs are required or optional depending on which categories you want to generate:

| Category | Source |
|---|---|
| `coding_regions` | GTF |
| `noncoding_regions` | inferred from GTF-derived annotated regions |
| `exons` | GTF |
| `introns` | inferred from GTF exon structure |
| `upstream_TSS` | inferred from GTF transcript boundaries |
| `start_codon` | GTF |
| `stop_codon` | GTF |
| `promoters` | `data_processing/region_info/promoters.bed` |
| `UTR5` | UCSC 5′ UTR BED export |
| `UTR3` | UCSC 3′ UTR BED export |
| `repeats` | UCSC RepeatMasker BED export |
| `UCNE` | `data_processing/region_info/hg38_UCNE_coordinates.bed` |
| `vista_enhancer` | `data_processing/region_info/experiments.tsv` |
| `phyloP_positive` | sampled from phyloP bigWig |
| `phyloP_neutral` | sampled from phyloP bigWig |
| `phyloP_negative` | sampled from phyloP bigWig |

Prepare per-chromosome GTF files in:

```text
data_processing/data/gtfs/
```

Expected structure:

```text
data_processing/data/gtfs/
├── chr1.gtf
├── chr2.gtf
├── ...
└── chrX.gtf
```

These are derived from GENCODE.

Repeat annotations can be exported from the UCSC Genome Browser RepeatMasker track and saved locally as:

```text
data_processing/region_info/repeats_hg38.bed
```

UTR annotations can be exported from UCSC as BED files and saved locally as:

```text
data_processing/region_info/UCSC_5UTR_exons.bed
data_processing/region_info/UCSC_3UTR_exons.bed
```

Promoter annotations can be downloaded from EPD:

```text
https://epd.expasy.org/ftp/epdnew/human/current/
```

By default, generated region BED files are written to:

```text
data_processing/data/regions/
```

with one subdirectory per category and one BED file per chromosome:

```text
data_processing/data/regions/
├── coding_regions/
│   ├── chr1.bed
│   ├── chr2.bed
│   └── ...
├── UCNE/
├── repeats/
├── UTR3/
├── UTR5/
└── vista_enhancer/
```

Example command:

```bash
python data_processing/sample_regions.py \
  --bigwig_file data_processing/data/240-mammalian/241-mammalian-2020v2.bigWig \
  --genome_fasta data_processing/data/240-mammalian/hg38.ml.fa \
  --gtf_dir data_processing/data/gtfs/ \
  --vista_tsv data_processing/region_info/experiments.tsv \
  --promoters_bed data_processing/region_info/promoters.bed \
  --utr5_bed data_processing/region_info/UCSC_5UTR_exons.bed \
  --utr3_bed data_processing/region_info/UCSC_3UTR_exons.bed \
  --repeats_bed data_processing/region_info/repeats_hg38.bed \
  --ucne_bed data_processing/region_info/hg38_UCNE_coordinates.bed \
  --ucne_paralogues data_processing/region_info/ucne_paralogues.txt \
  --output_dir data_processing/data/regions \
  --chromosomes auto \
  --num_regions 10000 \
  --region_length 2048 \
  --limit_per_category 10000 \
  --phylop_num_samples 10000 \
  --seed 42
```

Small chr22 test:

```bash
python data_processing/sample_regions.py \
  --chromosomes chr22 \
  --num_regions 100 \
  --phylop_num_samples 1000 \
  --limit_per_category 100 \
  --output_dir data_processing/data/regions_test
```

The script enforces non-overlap between categories using a priority order, so higher-priority feature classes are retained first.

### 2. Generate ATG data

The script `data_processing/make_ATG_data.py` generates a 5-way ATG benchmark. For each transcript with a valid ATG start codon, it identifies:

1. the true start codon ATG
2. a nearby noncoding ATG, 2–5 kb away by default
3. a far noncoding ATG, at least 100 kb away by default
4. an in-frame internal methionine from the same coding sequence
5. an out-of-frame ATG motif from the same coding sequence

Default output directory:

```text
data_processing/data/ATGs_simplified/
```

Each chromosome produces a TSV:

```text
data_processing/data/ATGs_simplified/
├── chr1_atg_5way_labels.tsv
├── chr2_atg_5way_labels.tsv
└── ...
```

Example for one chromosome:

```bash
python data_processing/make_ATG_data.py \
  --chrom chr22 \
  --gtf_dir data_processing/data/gtfs \
  --genome data_processing/data/240-mammalian/hg38.ml.fa \
  --out data_processing/data/ATGs_simplified \
  --n_sample 10000 \
  --random_seed 42
```

Loop over autosomes:

```bash
for i in $(seq 1 22); do
  chrom="chr${i}"
  echo "[RUN] ${chrom}"
  python data_processing/make_ATG_data.py \
    --chrom "$chrom" \
    --gtf_dir data_processing/data/gtfs \
    --genome data_processing/data/240-mammalian/hg38.ml.fa \
    --out data_processing/data/ATGs_simplified \
    --n_sample 10000 \
    --random_seed 42
done
```

Concatenate chromosome-level TSVs:

```bash
mkdir -p data_processing/data/ATGs

head -n 1 data_processing/data/ATGs_simplified/chr1_atg_5way_labels.tsv \
  > data_processing/data/ATGs/all_chr_atg_5way.tsv

for f in data_processing/data/ATGs_simplified/chr*_atg_5way_labels.tsv; do
  tail -n +2 "$f" >> data_processing/data/ATGs/all_chr_atg_5way.tsv
done
```

### 3. Run ATG representation evaluations

The script `src/evaluation/ATG_reps.py` loads the ATG 5-way TSV, extracts sequence contexts around each ATG, embeds them with Gamba/Caduceus or baseline features, and evaluates whether representations distinguish the five ATG classes using leave-one-out 1-nearest-neighbor classification.

Default ATG input:

```text
data_processing/data/ATGs/all_chr_atg_5way.tsv
```

Default output directory:

```text
data_processing/data/240-mammalian/ATG_reps_5way/
```

Example Gamba dual-task evaluation:

```bash
python src/evaluation/ATG_reps.py \
  --atg_tsv_path data_processing/data/ATGs/all_chr_atg_5way.tsv \
  --bigwig_file data_processing/data/240-mammalian/241-mammalian-2020v2.bigWig \
  --genome_fasta data_processing/data/240-mammalian/hg38.ml.fa \
  --checkpoint_dir /home/mica/gamba/ \
  --config_fpath configs/jamba-small-240mammalian.json \
  --output_dir data_processing/data/240-mammalian/ATG_reps_5way \
  --model_type gamba \
  --training_task dual \
  --last_step 44000 \
  --batch_size 32 \
  --n_examples 2000 \
  --seed 42
```

Other trained model variants:

```bash
python src/evaluation/ATG_reps.py \
  --model_type gamba \
  --training_task seq_only \
  --last_step 44000 \
  --n_examples 2000

python src/evaluation/ATG_reps.py \
  --model_type gamba \
  --training_task cons_only \
  --last_step 44000 \
  --n_examples 2000
```

Random-initialized comparison:

```bash
python src/evaluation/ATG_reps.py \
  --model_type gamba \
  --training_task dual \
  --last_step 0 \
  --n_examples 2000
```

Baselines:

```bash
python src/evaluation/ATG_reps.py \
  --baseline kmer6 \
  --n_examples 2000

python src/evaluation/ATG_reps.py \
  --baseline kmer6_flanked \
  --n_examples 2000

python src/evaluation/ATG_reps.py \
  --baseline phylop \
  --n_examples 2000
```

Use a 6-nt ROI around each ATG instead of the default 3-nt ROI:

```bash
python src/evaluation/ATG_reps.py \
  --model_type gamba \
  --training_task dual \
  --last_step 44000 \
  --use_6mer_roi
```

This evaluation saves files such as:

```text
reps_<model_tag>_ATG_5way_all_labels.npz
reps_<model_tag>_ATG_5way_all_labels_meta.parquet
reps_<model_tag>_ATG_5way_all_labels_full.npz
reps_<model_tag>_ATG_5way_all_labels_full_meta.parquet
knn_heatmap_<model_tag>_ATG5way_all_labels.png
knn_heatmap_<model_tag>_ATG1_vs_2.png
knn_heatmap_<model_tag>_ATG1_vs_3.png
knn_heatmap_<model_tag>_ATG1_vs_4.png
knn_heatmap_<model_tag>_ATG1_vs_5.png
balanced_accuracy_<model_tag>_ATG5way.json
sampled_examples_atg5.tsv
sampled_examples_atg5.meta.json
```

### 4. Run general downstream evaluations

General representation evaluations can be run with:

```bash
python src/evaluation/run_eval.py \
  --checkpoint_dir /home/mica/gamba/ \
  --config_fpath configs/jamba-small-240mammalian.json \
  --regions_dir data_processing/data/regions \
  --bigwig_file data_processing/data/240-mammalian/241-mammalian-2020v2.bigWig \
  --genome_fasta data_processing/data/240-mammalian/hg38.ml.fa \
  --model_type gamba \
  --training_task dual \
  --last_step 44000
```

Adjust `--training_task` for different model variants:

```bash
--training_task dual
--training_task seq_only
--training_task cons_only
```

Use `--last_step 0` for random-initialized comparisons where supported.

## Outputs

Training checkpoints are saved under the requested checkpoint/output directory. Evaluation outputs are written under the specified evaluation output directory and typically include representation arrays, metadata tables, plots, and metrics JSON files.

Common output types:

```text
*.npz              compressed embeddings and labels
*.parquet          per-example metadata
*.png              heatmaps and plots
*.json             evaluation metrics
*.bed              genomic regions
*.bedGraph         genome-browser-compatible prediction tracks
```

## Notes

- BED files are expected to use 0-based, half-open coordinates.
- GTF files are 1-based and closed by convention; scripts that derive BED-style intervals from GTFs should be checked carefully if exact base-level boundaries matter.
- The released Hugging Face models use custom model code, so `trust_remote_code=True` is required.
- For public Hugging Face repositories, no token is required.
