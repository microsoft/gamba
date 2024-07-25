import pyBigWig
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import numpy as np


def read_positions_and_entropy(npz_file_path):
    data = np.load(npz_file_path)
    print(data)
    entropy = data["entropy"]
    counts = data["counts"]
    print(entropy)
    print(counts)
    print(f"Loaded {len(entropy)} values.")
    return entropy


def read_positions(file_path):
    positions = []
    # read in the outfile as a pandas df, first column is header
    df = pd.read_csv(file_path, delim_whitespace=True)
    print(df.columns)

    positions = df["start"].tolist()
    position_ends = df["end"].tolist()
    position_pvals = df["pval"].tolist()
    print(positions[:5], position_ends[:5], position_pvals[:5])
    return positions, position_ends, position_pvals


def compare_phyloP(bigwig_file, position_info, entropy):
    # lets make a list from 0 to 3685883
    positions = list(range(0, 3685883))
    print(len(positions))
    # positions, positions_ends, pvals = position_info
    scores = []  # List to collect scores

    with pyBigWig.open(bigwig_file) as bw:
        # for position, position_end, pval in zip(positions, positions_ends, pvals):
        for position in positions:
            position_end = position + 1
            score = bw.values("chr1", position, position_end)[
                0
            ]  # pyBigWig uses 0-based, half-open intervals
            scores.append(score)
            # print(f"At Position: {position}, PhyloP run: {pval}, PhyloP Score: {score}")

    # Calculate Pearson correlation
    # correlation, _ = pearsonr(scores, pvals)

    # turn all infs in entropy to 0
    entropy[np.isnan(entropy)] = 0
    entropy[np.isinf(entropy)] = 0
    # check if nan in scores
    scores = np.array(scores)
    scores[np.isnan(scores)] = 0
    scores[np.isinf(scores)] = 0

    correlation_entrop, _ = pearsonr(scores, entropy)
    print(f"Pearson Correlation: {correlation_entrop}")

    # range of positions for binning
    min_pos, max_pos = min(positions), max(positions)
    bins = np.arange(min_pos, max_pos, 100)  # 100 bp bins
    bin_centers = 0.5 * (bins[1:] + bins[:-1])

    # scores and entropy into bins
    score_bins = np.digitize(positions, bins)
    entropy_bins = np.digitize(positions, bins)

    # average score and entropy for each bin
    avg_scores = [np.mean(scores[score_bins == i]) for i in range(1, len(bins))]
    avg_entropy = [np.mean(entropy[entropy_bins == i]) for i in range(1, len(bins))]

    # plot phyloP
    plt.figure(figsize=(10, 6))
    plt.scatter(
        bin_centers, avg_scores, alpha=0.5, color="blue", label="Avg PhyloP Scores"
    )
    plt.title("Positions vs Avg PhyloP Scores in 100bp Bins")
    plt.xlabel("Position")
    plt.ylabel("Average PhyloP Score")
    plt.legend()
    plt.savefig(
        "/home/t-mconsens/gamba/data_processing/data/240-mammalian/phyloP_100bp_bins.png"
    )
    plt.show()

    # plot entropy
    plt.figure(figsize=(10, 6))
    plt.scatter(bin_centers, avg_entropy, alpha=0.5, color="green", label="Avg Entropy")
    plt.title("Positions vs Avg Entropy in 100bp Bins")
    plt.xlabel("Position")
    plt.ylabel("Average Entropy")
    plt.legend()
    plt.annotate(
        f"Pearson Correlation: {correlation_entrop:.2f}",
        xy=(0.05, 0.95),
        xycoords="axes fraction",
    )
    plt.savefig(
        "/home/t-mconsens/gamba/data_processing/data/240-mammalian/entropy_100bp_bins.png"
    )
    plt.show()
    # plt.figure(figsize=(10, 6))
    # plt.scatter(positions, scores, alpha=0.5, color="blue", label="Scores")
    # plt.scatter(positions, pvals, alpha=0.5, color="red", label="P-values")
    # plt.title("Positions vs PhyloP Score and P-value")
    # plt.xlabel("Position")
    # plt.ylabel("Value")
    # plt.legend()
    # plt.grid(True)
    # # Annotate the plot with the correlation coefficient
    # plt.annotate(
    #     f"Pearson Correlation: {correlation:.2f}",
    #     xy=(0.05, 0.95),
    #     xycoords="axes fraction",
    # )
    # plt.savefig(
    #     "/home/t-mconsens/gamba/data_processing/data/240-mammalian/score_vs_pval.png"
    # )  # Save the plot as an image

    # clear figure
    # plt.clf()

    # # Plotting
    # plt.figure(figsize=(10, 6))
    # plt.scatter(positions, scores, alpha=0.5, color="blue", label="PhyloP Scores")
    # plt.scatter(positions, entropy, alpha=0.5, color="green", label="Entropy")
    # plt.title("Positions vs PhyloP Scores and Entropy")
    # plt.xlabel("Position")
    # plt.ylabel("Value")
    # plt.legend()
    # plt.annotate(
    #     f"Pearson Correlation: {correlation_entrop:.2f}",
    #     xy=(0.05, 0.95),
    #     xycoords="axes fraction",
    # )
    # plt.grid(True)
    # plt.savefig(
    #     "/home/t-mconsens/gamba/data_processing/data/240-mammalian/phyloP_vs_entropy.png"
    # )
    # plt.show()


def main():
    parser = argparse.ArgumentParser(description="Compare p-values with phyloP scores")
    parser.add_argument(
        "--bigwig_file",
        type=str,
        default="/home/t-mconsens/gamba/data_processing/data/241-mammalian-2020v2.bigWig",
        help="Path to the bigwig file with phyloP scores",
    )
    parser.add_argument(
        "--positions_file",
        type=str,
        default="/home/t-mconsens/gamba/data_processing/data/240-mammalian/chr1test_features.out",
        help="File with positions to compare",
    )
    parser.add_argument(
        "--entropy_file",
        type=str,
        default="/home/t-mconsens/gamba/data_processing/data/240-mammalian/1_entropy_and_species.npz",
        help="NPZ file with entropy",
    )

    args = parser.parse_args()

    print("entropy file: ", args.entropy_file)

    position_info = read_positions(args.positions_file)
    entropy = read_positions_and_entropy(args.entropy_file)
    compare_phyloP(args.bigwig_file, position_info, entropy)


if __name__ == "__main__":
    main()
