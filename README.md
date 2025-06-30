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
mkdir -p data_processing/data/240-mammalian/
#download the human chromosome sizes to make the bed file
curl https://hgdownload.cse.ucsc.edu/goldenpath/hg38/bigZips/hg38.chrom.sizes > data_processing/data/240-mammalian/hg38.chrom.sizes
python data_processing/generate_human_bed.py
#download full human genome fasta
curl https://storage.googleapis.com/basenji_barnyard2/hg38.ml.fa.gz > data_processing/data/240-mammalian/hg38.ml.fa.gz 
gunzip data_processing/data/240-mammalian/hg38.ml.fa.gz
# download centromere locations
curl https://hgdownload.soe.ucsc.edu/goldenPath/hg38/database/centromeres.txt.gz > data_processing/data/240-mammalian/centromeres.txt.gz
gunzip data_processing/data/240-mammalian/centromeres.txt.gz
#download the repeat locations from the UCSC genome browser at RepeatMasker track, whole genome, save file as repeats_hg38.bed.gzip & put in  data_processing/data/
gunzip data_processing/data/repeats_hg38.bed.gz
#download the phyloP scores
curl https://cgl.gi.ucsc.edu/data/cactus/241-mammalian-2020v2-hub/Homo_sapiens/241-mammalian-2020v2.bigWig > data_processing/data/240-mammalian/241-mammalian-2020v2.bigWig
#add splits.json to data_processing/data/240-mammalian/
touch data_processing/data/240-mammalian/splits.json
# copy this in and uncomment:
# {
#   "train": [
#     "1", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", 
#     "17", "18", "19", "20", "21", "X"
#   ],
#   "valid": [
#     "3", "16"
#   ],
#   "test": [
#     "2", "22"
#   ]
# }
#make exclusions and generate clean data
python data_processing/exclusion_regions.py
python data_processing/generate_clean_phyloP.py #this needs to be run for EACH chromosome something like:
# #!/bin/bash
# # loop through chromosomes 1 to 22 and X
# for chrom in {1..22} X
# do
#     echo "running for chromosome chr$chrom"
#     python data_processing/generate_clean_phyloP.py --chromosome "chr$chrom"
# done
#to make a fasta of the same data run data_processing/generate_same_data_fasta.py
# now need to uncompress the data before running 
python data_procesing/uncompress_npz.py --type "small"
python assert_chromosome_sizes.py --type "small" #check this returns correct
#at the end you should have this file structure in data_processing/data/240-mammalian
# test /
#      2_conservation_small.npy
#      2_sequence_small.npy
#      2.npz
#      22_conservation_small.npy
#      22_sequence_small.npy
#      22.npz
# train/
#    (with similar but for chr1,4,5,6,7,8,9,10,11,12,13,14,15,17,18,19,20,21,X )
# valid/
#    (with similar but for chr3 and 16)
python /src/test_sequence.py #(do not do mini-run)
# promoters from https://epd.expasy.org/ftp/epdnew/human/current/
```
