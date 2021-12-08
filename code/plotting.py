from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


OUTPUT = "results"
INPUT = "inputs/interim/dataset-ds000288_atlas-schaefer7networks_nroi-400_desc-qcfc_data.tsv"


def main():
    output = Path(__file__).parents[1] / OUTPUT
    input = Path(__file__).parents[1] / INPUT

    data = pd.read_csv(input, sep='\t', index_col=0)
    sig_per_edge = data.filter(regex='pval')
    metric_per_edge = data.filter(regex='corr')

    # separate p-value and correlation stats
    # plotting test
    ax = sns.barplot(data=(sig_per_edge<0.05), ci=None)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.set(ylabel="Proportion of \nedge significantly correlated \nwith mean FD",
            xlabel="confound removal strategy")
    plt.tight_layout()
    plt.savefig(output / "dataset-ds000288_qcfc_percentage_sig_edge.png")
    plt.close()

    ax = sns.stripplot(data=metric_per_edge, dodge=True, alpha=.01, zorder=1)
    sns.pointplot(data=metric_per_edge, dodge=.8 - .8 / 3,
                  join=False, palette="dark",
                  estimator=np.median,
                  markers="d", scale=.75, ci=None)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.set(ylabel="Pearson\'s correlation: \nmean FD and\nconnectome edges",
           xlabel="confound removal strategy")
    plt.tight_layout()
    plt.savefig(output / "dataset-ds000288_qcfc_dsitribution_edge.png")


if __name__ == "__main__":
    main()
