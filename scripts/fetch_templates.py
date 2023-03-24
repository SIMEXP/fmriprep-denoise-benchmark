"""
Set up templateflow in customised directory.
Download atlases that are relevant.

This is no longer need for curious curator. We uploaded all the atlas on osf.
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


def download_difumo():
    """Download to nilearn_data"""
    import nilearn.datasets

    difumo_path = Path(__file__).parent / "difumo_segmentation" / "data" / "raw"
    for d in [64, 128, 256, 512, 1024]:
        for r in [2, 3]:
            nilearn.datasets.fetch_atlas_difumo(
                dimension=d, resolution_mm=r, data_dir=str(difumo_path)
            )


def main():
    tf_dir = Path(__file__).parents[1] / "inputs" / "custome_templateflow"
    os.environ["TEMPLATEFLOW_HOME"] = str(tf_dir.resolve())

    fetch_schaefer()
    verify_gordon()


if __name__ == "__main__":
    main()
