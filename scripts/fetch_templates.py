"""
Set up templateflow in customised directory.
Download atlases that are relevant.
"""
import os
from pathlib import Path


def fetch_schaefer():
    """Download schaefer2018."""
    import templateflow.api as tf
    schaefer_path = tf.get("MNI152NLin2009cAsym", atlas="Schaefer2018")
    if isinstance(schaefer_path, list) and len(schaefer_path) > 0:
        print("Schaefer atlas exists.")


def verify_gordon():
    """Check gordon 333 exists, or raise warning for sanity."""
    import templateflow.api as tf
    gordon_path = tf.get("MNI152NLin6Asym", atlas="gordon")
    if isinstance(gordon_path, list) and len(gordon_path) == 2:
        print("Gordon 333 atlas exists.")
    else:
        raise FileNotFoundError(
            "Gordon 333 atlas not found. "
            "Please find the file in project remote repository."
        )


def calculate_difumo_centroids():
    from fmriprep_denoise.features.distance_dependency import get_difumo_centroids
    for d in [64, 128, 256, 512, 1024]:
        get_difumo_centroids(d)


def main():
    tf_dir = Path(__file__).parents[1] / "inputs" / "custome_templateflow"
    os.environ['TEMPLATEFLOW_HOME'] = str(tf_dir.resolve())

    fetch_schaefer()
    verify_gordon()
    calculate_difumo_centroids()


if __name__ == "__main__":
    main()
