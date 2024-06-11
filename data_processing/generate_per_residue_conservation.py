import pyBigWig
import pandas as pd
import os 
import argparse

#generates a dataframe with per nucleotide conservation scores
def make_scores_df(bigwig_file, bed, file_path):
    #open the bigwig file
    bw = pyBigWig.open(bigwig_file)

    #empty list to store conservation scores
    conservation_scores = []

    #iterate over the BED file
    for index, row in bed.iterrows():
        #get the conservation scores
        chrom = row["chrom"]
        start = row["start"]
        end = row["end"]

        #get scores for each position in the range
        vals = bw.values(chrom, start, end, numpy = True)

        #check if the returned vals are valid
        if vals is not None:
            #replace nans with 0s
            vals = [0 if pd.isna else val for val in vals]
            conservation_scores.append(vals)
        else:
            #if the returned vals are invalid,append 0s
            conservation_scores.append([0] * (end - start))
        
    #close the bigwig file
    bw.close()

    #flatten the scores and create new df for the conservation scores
    flattened_scores = []
    for index, score_list in enumerate(conservation_scores):
        for position, score in enumerate(score_list):
            flattened_scores.append([bed.loc[index, "chrom"], bed.loc[index, "start"] + position, bed.loc[index, "start"] + position +1, score])

    flattened_scores_df = pd.DataFrame(flattened_scores, columns=["chrom", "start", "end", "conservation_score"])

    #save df to new BED
    flattened_scores_df.to_csv(f'{file_path}conservation_scores.bed', sep="\t", index=False, header=False)

def main():
    #process command line arguments
    parser = argparse.ArgumentParser(description='Generate a dataframe with per nucleotide conservation scores')
    parser.add_argument('--bigwig_file', type=str, default='/home/t-mconsens/micaconsenshot/241-mammalian-2020v2.bigwig', help='Path to the bigwig file with phyloP scores')
    parser.add_argument('--bed_file', type=str, default ='/home/t-mconsens/gamba/data_processing/hg38.bed', help='File name of the bed file')
    parser.add_argument('--file_path', type=str, default ='/home/t-mconsens/gamba/data_processing/', help='Directory to save the new dataframe')
    args = parser.parse_args()

    #load the BED file to pandas df
    bed = pd.read_csv(args.bed_file, sep="\t", header=None, names=["chrom", "start", "end"])

    make_scores_df(args.bigwig_file, bed, args.file_path)
    print(f"Conservation scores BED file created: {args.file_path}conservation_scores.bed")



if __name__ == "__main__":
    main()