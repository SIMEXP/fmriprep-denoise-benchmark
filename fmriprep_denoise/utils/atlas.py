from pathlib import Path
from re import A

import pandas as pd

from fmriprep_denoise.metrics import compute_pairwise_distance
from nilearn.maskers import NiftiLabelsMasker, NiftiMapsMasker
from sklearn.utils import Bunch

import templateflow


ATLAS_METADATA = {
    'schaefer7networks': {
        'atlas': 'Schaefer2018',
        'template': "MNI152NLin2009cAsym",
        'resolution': 2,
        'dimensions': [100, 200, 300, 400, 500, 600, 800, 1000],
        'source': "templateflow"
    },
    'mist':{
        'atlas': 'MIST',
        'template': "MNI152NLin2009bSym",
        'resolution': 3,
        'dimensions' : [7, 12, 20, 36, 64, 122, 197, 325, 444, "ROI"],
        'source': "custome_templateflow"
    },
    'difumo': {
        'atlas': 'DiFuMo',
        'template': "MNI152NLin2009cAsym",
        'resolution': 2,
        'dimensions': [64, 128, 256, 512, 1024],
        'source': "custome_templateflow"
    },
    'gordon333': {
        'atlas': 'gordon',
        'template': "MNI152NLin6Asym",
        'resolution': 3,
        'dimensions': [333],
        'source': "custome_templateflow"
    }
}

custome_templateflow = Path(__file__).parent / "data" / "custome_templateflow"

def update_templateflow_path(atlas_name):
    """Update local templateflow path, if needed."""

    atlas_source = ATLAS_METADATA[atlas_name]['source']

    # by default, it uses `~/.cache/templateflow/`
    if atlas_source == "templateflow":
        templateflow.conf.TF_HOME = templateflow.conf.TF_DEFAULT_HOME
        templateflow.conf.update(overwrite=False, silent=True)
    # otherwise use customised map
    elif atlas_source == "custome_templateflow":
        templateflow.conf.TF_HOME = custome_templateflow
        templateflow.conf.update(local=True)
    templateflow.conf.init_layout()


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
    update_templateflow_path(atlas_name)
    cur_atlas_meta = ATLAS_METADATA[atlas_name].copy()

    parameters = {
        'atlas': cur_atlas_meta['atlas'],
        'resolution': cur_atlas_meta['resolution'],
        'extension': ".nii.gz"
    }
    if atlas_name == 'schaefer7networks':
        parameters['desc'] = f"{dimension}Parcels7Networks"
    elif atlas_name == 'difumo':
        parameters['desc'] = f"{dimension}dimensionsSegmented"
    else:
        parameters['desc'] = str(dimension)
    print(cur_atlas_meta['template'])
    print(parameters)
    img_path = templateflow.api.get(cur_atlas_meta['template'], raise_empty=True, **parameters)
    img_path = str(img_path)
    if atlas_name == 'schaefer7networks':
        parameters.popitem('resolution')
    parameters['extension'] = ".tsv"
    label_path = templateflow.api.get(cur_atlas_meta['template'], raise_empty=True, **parameters)
    labels = pd.read_csv(label_path, delimiter="\t")
    atlas_type = img_path.split('_')[-1].split('.nii.gz')[0]

    return Bunch(maps=img_path, labels=labels, type=atlas_type)


def create_atlas_masker(atlas_name, dimension, nilearn_cache=""):
    """Create masker given metadata.
    Parameters
    ----------
    atlas_name : str
        Atlas name. Must be a key in ATLAS_METADATA.
    """
    atlas = fetch_atlas_path(atlas_name, dimension)

    if atlas.type == 'dseg':
        masker = NiftiLabelsMasker(atlas.maps, detrend=True)
    elif atlas.type == 'probseg':
        masker = NiftiMapsMasker(atlas.maps, detrend=True)
    if nilearn_cache:
        masker = masker.set_params(memory=nilearn_cache, memory_level=1)
    labels = list(range(1, atlas.labels.shape[0] + 1))
    return masker, labels


def get_atlas_dimensions(atlas_name):
    return ATLAS_METADATA[atlas_name]['dimensions']
