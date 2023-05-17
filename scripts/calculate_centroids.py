import os
from pathlib import Path


def calculate_difumo_centroids():
    from fmriprep_denoise.features.distance_dependency import get_difumo_centroids

    for d in [64, 128, 256, 512, 1024]:
        get_difumo_centroids(d)


def main():
    tf_dir = Path(__file__).parents[1] / "inputs" / \
        "fmriprep-denoise-benchmark" / "custome_templateflow"
    os.environ["TEMPLATEFLOW_HOME"] = str(tf_dir.resolve())
    calculate_difumo_centroids()


if __name__ == "__main__":
    main()
