# gamba

Genomic language models (glm) with conservation scores

## Abstract
gamba is a glm based on the Jamba model trained to predict both nucleotide tokens and phyloP conservation scores.


## Installation
To set up gamba, clone the repository and navigate to the project directory:

```bash
git clone ...
cd gamba/
```

## Data Preparation
To set up the data for gamba:

```bash
mkdir -p data_processing/data
curl https://storage.googleapis.com/basenji_barnyard2/hg38.ml.fa.gz > data_processing/data/hg38.ml.fa.gz 
curl https://storage.googleapis.com/basenji_barnyard2/sequences_human.bed > data_processing/data/human_sequences.bed
curl https://cgl.gi.ucsc.edu/data/cactus/241-mammalian-2020v2-hub/Homo_sapiens/241-mammalian-2020v2.bigWig > data_processing/data/241-mammalian-2020v2.bigWig
gunzip data_processing/data/hg38.ml.fa.gz
python data_processing/generate_per_nucleotide_conservation.py
```
