from pathlib import Path

import pandas as pd

from nilearn.maskers import NiftiLabelsMasker, NiftiMapsMasker
from nilearn.plotting import find_probabilistic_atlas_cut_coords

from sklearn.utils import Bunch

import templateflow


ATLAS_METADATA = {
    'schaefer7networks': {
        'atlas': 'Schaefer2018',
        'template': "MNI152NLin2009cAsym",
        'resolution': 2,
        'dimensions': [100, 200, 300, 400, 500, 600, 800, 1000],
    },
    'mist':{
        'atlas': 'MIST',
        'template': "MNI152NLin2009bSym",
        'resolution': 3,
        'dimensions' : [7, 12, 20, 36, 64, 122, 197, 325, 444, "ROI"],
    },
    'difumo': {
        'atlas': 'DiFuMo',
        'template': "MNI152NLin2009cAsym",
        'resolution': 2,
        'dimensions': [64, 128, 256, 512, 1024],
    },
    'gordon333': {
        'atlas': 'gordon',
        'template': "MNI152NLin6Asym",
        'resolution': 3,
        'dimensions': [333],
    }
}

# Include retreival of these data in README


def fetch_atlas_path(atlas_name, dimension):
    """
    Generate a dictionary containing parameters for TemplateFlow quiery.

    Parameters
    ----------

    atlas_name : str
        Atlas name. Must be a key in ATLAS_METADATA.

    dimension : str or int
        Atlas dimension.

    description_keywords : dict
        Keys and values to fill in description_pattern.
        For valid keys check relevant ATLAS_METADATA[atlas_name]['description_pattern'].

    Return
    ------
    sklearn.utils.Bunch
        Containing the following fields:
        maps : str
            Path to atlas map.
        labels : pandas.DataFrame
            The corresponding pandas dataframe of the atlas
        type : str
            'dseg' (for NiftiLabelsMasker) or 'probseg' (for NiftiMapsMasker)
    """
    templateflow.conf.TF_HOME = Path(__file__).parents[2] / "inputs" / "custome_templateflow"
    cur_atlas_meta = ATLAS_METADATA[atlas_name].copy()

    parameters = {
        'atlas': cur_atlas_meta['atlas'],
        'resolution': f"{cur_atlas_meta['resolution']:02d}",
        'extension': ".nii.gz"
    }
    if atlas_name == 'schaefer7networks':
        parameters['desc'] = f"{dimension}Parcels7Networks"
    elif atlas_name == 'difumo':
        parameters['desc'] = f"{dimension}dimensionsSegmented"
    else:
        parameters['desc'] = str(dimension)
    img_path = templateflow.api.get(cur_atlas_meta['template'],
                                    raise_empty=True, **parameters)
    img_path = str(img_path)
    if atlas_name == 'schaefer7networks':
        parameters.pop('resolution')
    parameters['extension'] = ".tsv"
    label_path = templateflow.api.get(cur_atlas_meta['template'],
                                      raise_empty=True, **parameters)
    labels = pd.read_csv(label_path, delimiter="\t")
    atlas_type = img_path.split('_')[-1].split('.nii.gz')[0]

    return Bunch(maps=img_path, labels=labels, type=atlas_type)


def create_atlas_masker(atlas_name, dimension, subject_mask, detrend=True,
                        nilearn_cache=""):
    """Create masker given metadata.
    Parameters
    ----------
    atlas_name : str
        Atlas name. Must be a key in ATLAS_METADATA.
    """
    atlas = fetch_atlas_path(atlas_name, dimension)
    labels = list(range(1, atlas.labels.shape[0] + 1))

    if atlas.type == 'dseg':
        masker = NiftiLabelsMasker(atlas.maps, labels=labels,
                                   mask_img=subject_mask, detrend=detrend)
    elif atlas.type == 'probseg':
        masker = NiftiMapsMasker(atlas.maps,
                                 mask_img=subject_mask, detrend=detrend)
    if nilearn_cache:
        masker = masker.set_params(memory=nilearn_cache, memory_level=1)

    return masker, labels


def get_centroid(atlas_name, dimension):
    """Load parcel centroid for each atlas."""
    if atlas_name not in ATLAS_METADATA:
        raise NotImplementedError("Selected atlas is not supported.")

    if atlas_name == 'schaefer7networks':
        url = f"https://raw.githubusercontent.com/ThomasYeoLab/CBIG/master/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/MNI/Centroid_coordinates/Schaefer2018_{dimension}Parcels_7Networks_order_FSLMNI152_2mm.Centroid_RAS.csv"
        return pd.read_csv(url).loc[:, ['R', 'S', 'A']].values
    if atlas_name == 'gordon333':
        file_dist = "atlas-gordon333_nroi-333_desc-distance.tsv"
        return pd.read_csv(Path(__file__).parent / "data" / file_dist, sep='\t')

    current_atlas = fetch_atlas_path(atlas_name, dimension)
    if atlas_name == 'mist':
        return current_atlas.labels.loc[:, ['x', 'y', 'z']].values
    if atlas_name == 'difumo':
        return find_probabilistic_atlas_cut_coords(current_atlas.maps)


def get_atlas_dimensions(atlas_name):
    return ATLAS_METADATA[atlas_name]['dimensions']
