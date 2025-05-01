import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt
import logging
from pathlib import Path

from fmriprep_denoise.visualization import utils  # or hardcode strategy list

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

strategies = list(utils.GRID_LOCATION.values())  # or hardcode: ["baseline", "simple", "simple+gsr", ...]

def analyze_connectome_similarity(path_root, datasets, fmriprep_version):
    for dataset in datasets:
        logger.info(f"--- Dataset: {dataset} ---")
        connectome_files = Path(path_root).glob(f"{dataset}/{fmriprep_version}/*connectome.tsv")

        for file_path in connectome_files:
            logger.info(f"Loading file: {file_path}")
            df = pd.read_csv(file_path, sep="\t", index_col=0)
            strategies_present = [s for s in strategies if s in df.columns]

            if not strategies_present:
                logger.warning(f"No expected strategies found in {file_path}")
                continue

            df_strats = df[strategies_present]

            print(f"\n📂 File: {file_path.name}")
            print(f"Strategies found: {strategies_present}")

            # 1. Correlation matrix
            corr_matrix = df_strats.corr()
            print("\n✅ Pairwise Correlation Matrix:")
            print(corr_matrix.round(4))

            # 2. Mean & Max Absolute Differences
            print("\n📉 Mean & Max Absolute Differences:")
            for s1, s2 in itertools.combinations(strategies_present, 2):
                abs_diff = (df_strats[s1] - df_strats[s2]).abs()
                print(f"{s1:17s} vs {s2:17s} | mean = {abs_diff.mean():.6f}, max = {abs_diff.max():.6f}")

            # 3. Per-edge variance across strategies
            print("\n🎯 Per-edge Variance Across Strategies:")
            per_edge_var = df_strats.var(axis=1)
            print(per_edge_var.describe().round(6))

            # 4. Histogram of differences between most and least similar strategies
            s_most, s_least = None, None
            min_corr, max_corr = 1.0, 0.0
            for s1, s2 in itertools.combinations(strategies_present, 2):
                corr_val = corr_matrix.loc[s1, s2]
                if corr_val < min_corr:
                    min_corr, s_least = corr_val, (s1, s2)
                if corr_val > max_corr and s1 != s2:
                    max_corr, s_most = corr_val, (s1, s2)

            if s_least:
                diff = (df_strats[s_least[0]] - df_strats[s_least[1]]).abs()
                plt.figure()
                plt.hist(diff, bins=50, alpha=0.7)
                plt.title(f"Histogram of |{s_least[0]} - {s_least[1]}|\n(corr = {min_corr:.4f})")
                plt.xlabel("Absolute Difference (Edge Strength)")
                plt.ylabel("Number of Edges")
                plt.grid(True)
                plt.tight_layout()
                plt.savefig("diff_histogram.png")

            print("\n------------------------------------------------------------")

# Example usage
if __name__ == "__main__":
    analyze_connectome_similarity(
        path_root="/home/seann/scratch/denoise/fmriprep-denoise-benchmark/outputs/denoise-metrics-atlas.5-4.27.25",
        datasets=["ds000228"],
        fmriprep_version="fmriprep-25.0.0"
    )