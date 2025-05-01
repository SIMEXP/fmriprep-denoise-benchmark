import os

from pathlib import Path

import pandas as pd

from nilearn.maskers import NiftiLabelsMasker, NiftiMapsMasker

from sklearn.utils import Bunch


ATLAS_METADATA = {
    "schaefer7networks": {
        "atlas": "schaefer7networks",
        "template": "MNI152NLin2009cAsym",
        "resolution": 2,
        "dimensions": [100, 200, 300, 400, 500, 600, 800],
    },
    "mist": {
        "atlas": "MIST",
        "template": "MNI152NLin2009bSym",
        "resolution": 3,
        "dimensions": [7, 12, 20, 36, 64, 122, 197, 325, 444, "ROI"],
    },
    "difumo": {
        "atlas": "DiFuMo",
        "template": "MNI152NLin2009cAsym",
        "resolution": 2,
        "dimensions": [64, 128, 256, 512, 1024],
    },
    "gordon333": {
        "atlas": "gordon",
        "template": "MNI152NLin6Asym",
        "resolution": 3,
        "dimensions": [333],
    },
    "Schaefer2018": {
        "atlas": "Schaefer2018Combined",
        "template": "MNI152NLin6Asym",
        "resolution": 2,
        "dimensions": [434],
    }
}

TEMPLATEFLOW_DIR = (
    Path(__file__).parents[2] / "data" / "fmriprep-denoise-benchmark" / "custome_templateflow"
    )

# Include retrieval of these data in README


def fetch_atlas_path(atlas_name, dimension, tf_dir=TEMPLATEFLOW_DIR):
    """
    Retrieve the path to the atlas files, either from TemplateFlow or a custom location.

    Parameters
    ----------
    atlas_name : str
        Atlas name. Must be a key in ATLAS_METADATA or a custom atlas.

    dimension : str or int
        Atlas dimension.

    tf_dir : pathlib.Path or str
        Custom TemplateFlow directory (if needed).

    Returns
    -------
    sklearn.utils.Bunch
        Contains:
            maps : str
                Path to atlas map (NIfTI file).
            labels : pandas.DataFrame
                Corresponding atlas labels.
            type : str
                'dseg' for NiftiLabelsMasker or 'probseg' for NiftiMapsMasker.
    """

    # Paths to custom atlases
    custom_atlas_paths = {
        "Schaefer2018Combined": {
            "nii": "/home/seann/projects/def-cmoreau/All_user_common_folder/atlas/atlas_enigma/atlas-Schaefer2018Combined_dseg.nii.gz",
            "tsv": "/home/seann/projects/def-cmoreau/All_user_common_folder/atlas/atlas_enigma/atlas-Schaefer2018Combined_dseg.tsv",
        }
    }

    print(f"Fetching atlas: {atlas_name} with dimension: {dimension}")
    
    # Check if the atlas is a custom atlas
    if atlas_name in custom_atlas_paths:
        print(f"Using custom atlas: {atlas_name}")
        img_path = custom_atlas_paths[atlas_name]["nii"]
        label_path = custom_atlas_paths[atlas_name]["tsv"]
        labels = pd.read_csv(label_path, delimiter="\t")
        atlas_type = "dseg"  # Likely a segmentation atlas (adjust if needed)

        return Bunch(maps=img_path, labels=labels, type=atlas_type)

    # If not a custom atlas, fall back to TemplateFlow
    import templateflow
    cur_atlas_meta = ATLAS_METADATA[atlas_name].copy()

    parameters = {
        "atlas": cur_atlas_meta["atlas"],
        "resolution": f"{cur_atlas_meta['resolution']:02d}",
        "extension": ".nii.gz",
    }
    if atlas_name == "schaefer7networks":
        parameters["desc"] = f"{dimension}Parcels7Networks"
    elif atlas_name == "difumo":
        parameters["desc"] = f"{dimension}dimensionsSegmented"
    else:
        parameters["desc"] = str(dimension)

    img_path = templateflow.api.get(
        cur_atlas_meta["template"], raise_empty=True, **parameters
    )
    img_path = str(img_path)

    parameters["extension"] = ".tsv"
    label_path = templateflow.api.get(
        cur_atlas_meta["template"], raise_empty=True, **parameters
    )
    labels = pd.read_csv(label_path, delimiter="\t")
    atlas_type = img_path.split("_")[-1].split(".nii.gz")[0]

    return Bunch(maps=img_path, labels=labels, type=atlas_type)


def create_atlas_masker(
    atlas_name,
    dimension,
    subject_mask,
    detrend=True,
    standardize=False,
    nilearn_cache="",
):
    """
    Create masker given metadata.

    Parameters
    ----------
    atlas_name : str
        Atlas name. Must be a key in ATLAS_METADATA.

    dimension : str or int
        Atlas dimension.

    subject_mask : pathlib.Path
        The corresponding brain mask to the subject's processed functional
        data.

    detrend : bool, default True
        Pass to the NiftiLabelsMasker / NiftiMapsMasker parameter detrend.

    standardize : bool, default False
        Pass to the NiftiLabelsMasker / NiftiMapsMasker parameter standardize.

    nilearn_cache : str
        Path to nilearn cache. Pass to the NiftiLabelsMasker / NiftiMapsMasker.

    Returns
    -------
    list
        Atlas labels
    """
    atlas = fetch_atlas_path(atlas_name, dimension)
    labels = list(range(1, atlas.labels.shape[0] + 1))

    if atlas.type == "dseg":
        masker = NiftiLabelsMasker(
            atlas.maps,
            labels=labels,
            mask_img=subject_mask,
            detrend=detrend,
            standardize=standardize,
        )
    elif atlas.type == "probseg":
        masker = NiftiMapsMasker(
            atlas.maps,
            mask_img=subject_mask,
            detrend=detrend,
            standardize=standardize,
        )
    if nilearn_cache:
        masker = masker.set_params(memory=nilearn_cache, memory_level=1)

    return masker, labels


def get_atlas_dimensions(atlas_name):
    """As function name."""
    return ATLAS_METADATA[atlas_name]["dimensions"]
