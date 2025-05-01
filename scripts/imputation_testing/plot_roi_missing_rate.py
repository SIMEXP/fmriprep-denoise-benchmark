import pandas as pd
import matplotlib.pyplot as plt
import argparse

def plot_missing_data(tsv_path):
    # Read the TSV file
    df = pd.read_csv(tsv_path, sep='\t')

    # Sort the dataframe by avg_missing_pct
    df_sorted = df.sort_values(by='avg_missing_pct', ascending=True).reset_index(drop=True)

    # Rename ROIs to "ROI 1", "ROI 2", ...
    df_sorted['roi'] = [f'ROI {i+1}' for i in range(len(df_sorted))]

    # Set up the plot
    # Set up the plot
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(df_sorted)), df_sorted['avg_missing_pct'])

    # Show generic ROI x-tick labels
    plt.xticks(ticks=range(len(df_sorted)), labels=df_sorted['roi'], rotation=90, fontsize=6)

    # Add vertical dashed lines for percentiles
    total = len(df_sorted)
    for pct in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
        index = int(total * pct)
        plt.axvline(x=index, color='red', linestyle='--', linewidth=1)
        plt.text(index, plt.ylim()[1]*0.95, f'{int(pct*100)}%', rotation=90, color='red', ha='right', fontsize=8)

    # Labels and title
    plt.ylabel('Average Missing Percentage')
    plt.xlabel('Sorted Generic ROIs')
    plt.title('Average Missing Percentage per ROI with Percentile Markers')
    plt.tight_layout()
    plt.savefig('/home/seann/scratch/denoise/fmriprep-denoise-benchmark/scripts/imputation_testing/plot_roi_missing_rate.png', bbox_inches="tight")
    # Show the plot
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Plot average missing percentage per ROI from a TSV file.")
    parser.add_argument("tsv_path", type=str, help="Path to the TSV file")
    args = parser.parse_args()
    plot_missing_data(args.tsv_path)

if __name__ == "__main__":
    main()

    #python plot_roi_missing_rate.py /home/seann/scratch/denoise/fmriprep-denoise-benchmark/outputs/denoise-metrics/ds000228_04.03.25/brain_visualization/ds000228/fmriprep-20.2.7/version_B_all_subjects_avg_missing_per_roi.tsv

