import nilearn
from nilearn.input_data import NiftiLabelsMasker, NiftiMapsMasker


# not availible in nilearn: Gordon, FIND, brainnetome
# not clear: ICA?
# possibly not applicable: Glasser
ATLAS_METADATA = {
    'schaefer7networks': {
        'type': 'static',
        'resolutions': [100, 400, 1000],
        'fetcher': "nilearn.datasets.fetch_atlas_schaefer_2018(n_rois={resolution}, yeo_networks=7, resolution_mm=2)"},
    'basc':{
        'type': 'static',
        'resolutions' : [122, 197, 325, 444],
        'fetcher': "nilearn.datasets.fetch_atlas_basc_multiscale_2015(version='asym')"
        },
    'craddock':{
        'type': 'static',
        'resolutions' : [100, 500, 1000],
        'fetcher': "nilearn.datasets.fetch_atlas_craddock_2012()"
        },
    'difumo': {
        'type': 'dynamic',
        'resolutions': [128, 512, 1024],
        'label_idx': 1,
        'fetcher': "nilearn.datasets.fetch_atlas_difumo(dimension={resolution}, resolution_mm=2)"}
}


def create_atlas_masker(atlas_name, nilearn_cache=""):
    """Create masker of all resolutions given metadata."""
    if atlas_name not in ATLAS_METADATA.keys():
        raise ValueError("{} not defined!".format(atlas_name))
    curr_atlas = ATLAS_METADATA[atlas_name]
    curr_atlas['name'] = atlas_name

    for resolution in curr_atlas['resolutions']:

        atlas, atlas_map = _get_atlas_maps(atlas_name, curr_atlas['fetcher'], resolution)

        if curr_atlas['type'] == "static":
            masker = NiftiLabelsMasker(
                atlas_map, detrend=True, standardize=True)
        elif curr_atlas['type'] == "dynamic":
            masker = NiftiMapsMasker(
                atlas_map, detrend=True, standardize=True)
        if nilearn_cache:
            masker = masker.set_params(memory=nilearn_cache, memory_level=1)
        # fill atlas info
        curr_atlas[resolution] = {'masker': masker}

        if 'labels' in atlas:
            curr_atlas[resolution]['labels'] = _clean_atlas_labels(curr_atlas, atlas.labels)
        else:
            curr_atlas[resolution]['labels'] = [
                f"region-{label}" for label in range(1, resolution + 1)]
    return curr_atlas


def _clean_atlas_labels(curr_atlas, atlas_labels):
    """Clean atlas in numpy array."""
    if isinstance(atlas_labels[0], tuple) | isinstance(atlas_labels[0], list):
        if isinstance(atlas_labels[0][curr_atlas['label_idx']], bytes):
            return [label[curr_atlas['label_idx']].decode() for label in atlas_labels]
        else:
            return [label[curr_atlas['label_idx']] for label in atlas_labels]
    elif isinstance(atlas_labels[0], bytes):
            return [label.decode() for label in atlas_labels]
    else:
        return [label for label in atlas_labels]


def _get_atlas_maps(atlas_name, atlas_fetcher, resolution):
    """Get atlas map path or nifti object."""
    if "resolution" in atlas_fetcher:
        atlas = eval(atlas_fetcher.format(resolution=resolution))
        atlas_map = atlas.maps
    elif atlas_name == "basc":
        atlas = eval(atlas_fetcher)
        atlas_map = atlas.get(f'scale{resolution:03d}', False)
        if not atlas_map:
            raise ValueError("The selected resolution is not avalible with BASC.")
    elif atlas_name == "craddock":
        raise NotImplementedError
    else:
        raise NotImplementedError
    return atlas,atlas_map