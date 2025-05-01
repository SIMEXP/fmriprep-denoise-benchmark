import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt


def simulate_missing_data(data, missing_rate=0.1, seed=42):
    np.random.seed(seed)
    mask = np.random.rand(*data.shape) < missing_rate
    data_missing = data.copy()
    data_missing[mask] = np.nan
    return data_missing, mask


def simulate_missing_from_real_distribution(data, roi_names, missing_tsv_path, seed=42):
    np.random.seed(seed)
    df_missing = pd.read_csv(missing_tsv_path, sep="\t")
    missing_map = dict(zip(df_missing["roi"], df_missing["avg_missing_pct"]))
    roi_probs = np.array([missing_map.get(roi, 0.0) for roi in roi_names])
    n_timepoints, n_rois = data.shape
    mask = np.random.rand(n_timepoints, n_rois) < roi_probs
    data_missing = data.copy()
    data_missing[mask] = np.nan
    print(f"[DEBUG] Missing profile (first 10): {roi_probs[:10]}")
    print(f"[DEBUG] Simulated missing mask stats — True count: {np.sum(mask)}, False count: {np.size(mask) - np.sum(mask)}")
    print(f"[DEBUG] Data shape: {data.shape}")
    print(f"[DEBUG] NaNs in simulated data: {np.isnan(data_missing).sum()}")
    return data_missing, mask


def impute_data(data_missing, strategy="mean"):
    if strategy == "mean":
        imputer = SimpleImputer(strategy="mean")
        return imputer.fit_transform(data_missing)
    elif strategy == "knn":
        imputer = KNNImputer(n_neighbors=5, weights="distance")
        return imputer.fit_transform(data_missing.T).T
    elif strategy == "mice":
        imputer = IterativeImputer(max_iter=10, random_state=0)
        return imputer.fit_transform(data_missing)
    else:
        raise ValueError(f"Unsupported imputation strategy: {strategy}")


def compute_metrics(true, imputed, mask):
    true_vals = true[mask]
    imputed_vals = imputed[mask]

    print(f"[DEBUG] true_vals shape: {true_vals.shape}")
    print(f"[DEBUG] imputed_vals shape: {imputed_vals.shape}")
    print(f"[DEBUG] Number of masked entries: {np.sum(mask)}")
    print(f"[DEBUG] Total NaNs in original: {np.isnan(true).sum()}, After mask: {np.isnan(true_vals).sum()}")
    print(f"[DEBUG] Total NaNs in imputed: {np.isnan(imputed).sum()}, After mask: {np.isnan(imputed_vals).sum()}")
    print(f"[DEBUG] Sample of true_vals: {true_vals[:10]}")
    print(f"[DEBUG] Sample of imputed_vals: {imputed_vals[:10]}")

    if true_vals.size == 0 or imputed_vals.size == 0:
        print("[DEBUG] Empty values detected — likely bad mask or all NaNs.")
        return {"error": "No valid data after masking — possible all NaNs or bad mask."}

    return {
        "rmse": np.sqrt(mean_squared_error(true_vals, imputed_vals)),
        "mae": mean_absolute_error(true_vals, imputed_vals),
        "r2": r2_score(true_vals, imputed_vals),
    }


def save_imputation_visualizations(subject_id, original, mask, missing_data, imputations_dict, out_dir):
    num_imputations = len(imputations_dict)
    fig, axs = plt.subplots(1, 4 + num_imputations, figsize=(5 * (4 + num_imputations), 5))

    # Normalize color scale (shared across continuous data panels)
    vmin, vmax = np.nanmin(original), np.nanmax(original)

    axs[0].imshow(original, aspect="auto", interpolation="none", vmin=vmin, vmax=vmax, cmap="viridis")
    axs[0].set_title("Original")

    im_mask = axs[1].imshow(mask, aspect="auto", interpolation="none", cmap="gray")
    axs[1].set_title("Missing Mask")
    fig.colorbar(im_mask, ax=axs[1], fraction=0.046, pad=0.04)

    axs[2].imshow(missing_data, aspect="auto", interpolation="none", vmin=vmin, vmax=vmax, cmap="viridis")
    axs[2].set_title("With Missing")

    for i, (method, imputed) in enumerate(imputations_dict.items()):
        axs[3 + i].imshow(imputed, aspect="auto", interpolation="none", vmin=vmin, vmax=vmax, cmap="viridis")
        axs[3 + i].set_title(f"Imputed ({method})")

    for ax in axs:
        ax.axis("off")

    plt.tight_layout()
    out_path = Path(out_dir) / f"{subject_id}_imputation_viz.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def process_subject(subject_path, missing_rate=0.1, missing_profile=None, visualize=False, output_dir="."):
    ts_files = list(subject_path.glob("*_timeseries.tsv"))
    if not ts_files:
        print(f"[DEBUG] No timeseries file found in: {subject_path}")
        return {}

    file = ts_files[0]
    print(f"[DEBUG] Found timeseries file for {subject_path.name}: {file}")

    if file.stat().st_size == 0:
        print(f"[DEBUG] Skipping {file.name}: file is empty.")
        return {}

    try:
        df = pd.read_csv(file, sep="\t", header=None)
    except pd.errors.EmptyDataError:
        print(f"[DEBUG] Skipping {file.name}: unable to parse (empty or invalid format).")
        return {}
    except Exception as e:
        print(f"[DEBUG] Skipping {file.name}: unexpected error: {e}")
        return {}

    if df.empty:
        print(f"[DEBUG] Skipping {file.name}: parsed DataFrame is empty.")
        return {}

    data = df.values.astype(float)
    print(f"[DEBUG] Loaded data shape for {subject_path.name}: {data.shape}")
    if data.size == 0:
        print(f"[DEBUG] Skipping {file.name}: no numerical data.")
        return {}

    roi_names = df.columns.tolist() if df.columns.size == data.shape[1] else [f"ROI_{i}" for i in range(data.shape[1])]

    if missing_profile is not None:
        data_missing, mask = simulate_missing_from_real_distribution(data, roi_names, missing_profile)
        print(f"[DEBUG] Simulated real NaNs for {subject_path.name} — NaNs total: {np.isnan(data_missing).sum()}")
        # Check if the mask is actually doing its job
        if not np.any(mask):
            print(f"[DEBUG] Mask contains no True values for {subject_path.name} — check missing_profile values.")
        else:
            print(f"[DEBUG] Mask seems valid with {np.sum(mask)} masked elements for {subject_path.name}")
    else:
        data_missing, mask = simulate_missing_data(data, missing_rate=missing_rate)
        print(f"[DEBUG] Simulated random NaNs for {subject_path.name} — NaNs total: {np.isnan(data_missing).sum()}")

    metrics = {}
    imputations = {}
    for method in ["mean", "knn", "mice"]:
        try:
            print(f"[DEBUG] Running {method} imputation for {subject_path.name}...")
            imputed = impute_data(data_missing, strategy=method)
            print(f"[DEBUG] Imputed data shape: {imputed.shape}")
            # Print how many values were imputed
            n_missing = np.isnan(data_missing).sum()
            n_restored = np.sum(~np.isnan(imputed)) - np.sum(~np.isnan(data_missing))
            print(f"[DEBUG] {method} imputed {n_restored} values out of {n_missing} missing.")
            metrics[method] = compute_metrics(data, imputed, mask)
            imputations[method] = imputed
        except Exception as e:
            print(f"[DEBUG] Error during {method} for {subject_path.name}: {e}")
            metrics[method] = {"error": str(e)}

    if visualize:
        save_imputation_visualizations(subject_path.name, data, mask, data_missing, imputations, output_dir)

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate imputation methods across subjects.")
    parser.add_argument("--base_folder", type=str, required=True, help="Base folder containing subject subfolders")
    parser.add_argument("--missing_rate", type=float, default=0.1, help="Rate of simulated missing data")
    parser.add_argument("--missing_profile_tsv", type=str, default=None, help="TSV file with avg_missing_pct per ROI")
    parser.add_argument("--output_file", type=str, default="imputation_comparison_results.csv", help="Output CSV filename")
    parser.add_argument("--visualize_output", type=str, default="imputation_viz", help="Folder to save visualizations")
    args = parser.parse_args()

    base = Path(args.base_folder)
    subjects = [p for p in base.iterdir() if p.is_dir() and "sub-" in p.name]

    Path(args.visualize_output).mkdir(parents=True, exist_ok=True)

    all_results = []
    for i, subj_path in enumerate(subjects):
        subj_id = subj_path.name
        print(f"Processing {subj_id}...")
        result = process_subject(
            subj_path,
            missing_rate=args.missing_rate,
            missing_profile=args.missing_profile_tsv,
            visualize=(i < 2),
            output_dir=args.visualize_output
        )
        for method, metrics in result.items():
            row = {"subject": subj_id, "method": method}
            row.update(metrics)
            all_results.append(row)

    df_results = pd.DataFrame(all_results)
    df_results.to_csv(args.output_file, index=False)
    print(f"Saved results to {args.output_file}")


if __name__ == "__main__":
    main()