import nilearn
from nilearn.input_data import NiftiLabelsMasker, NiftiMapsMasker

ATLAS_METADATA = {
    'schaefer7networks': {
        'type': 'static',
        'resolutions': [100, 400, 1000],
        'fetcher': "nilearn.datasets.fetch_atlas_schaefer_2018(n_rois={resolution}, resolution_mm=2)"},
    'mist':{
        'type': 'static',
        'resolutions' : [122, 197, 325, 444],
        'fetcher': "nilearn.datasets.fetch_atlas_basc_multiscale_2015(version='asym')"
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

        if "resolution" in curr_atlas['fetcher']:
            atlas = eval(curr_atlas['fetcher'].format(resolution=resolution))
            atlas_path = atlas.maps
        elif atlas_name == "mist":
            atlas = eval(curr_atlas['fetcher'])
            atlas_path = atlas.get(f'scale{resolution:03d}', False)
            if not atlas_path:
                raise ValueError("The selected resolution is not avalible with mist")
        else:
            raise NotImplementedError

        if curr_atlas['type'] == "static":
            masker = NiftiLabelsMasker(
                atlas_path, detrend=True, standardize=True)
        elif curr_atlas['type'] == "dynamic":
            masker = NiftiMapsMasker(
                atlas_path, detrend=True, standardize=True)
        if nilearn_cache:
            masker = masker.set_params(memory=nilearn_cache, memory_level=1)
        # fill atlas info
        curr_atlas[resolution] = {'masker': masker}

        # fix label
        if 'labels' in atlas:
            if isinstance(atlas.labels[0], tuple) | isinstance(atlas.labels[0], list):
                if isinstance(atlas.labels[0][curr_atlas['label_idx']], bytes):
                    curr_atlas[resolution]['labels'] = [
                        label[curr_atlas['label_idx']].decode() for label in atlas.labels]
                else:
                    curr_atlas[resolution]['labels'] = [
                        label[curr_atlas['label_idx']] for label in atlas.labels]
            elif isinstance(atlas.labels[0], bytes):
                    curr_atlas[resolution]['labels'] = [
                        label.decode() for label in atlas.labels]
            else:
                curr_atlas[resolution]['labels'] = [
                    label for label in atlas.labels]
        else:
            curr_atlas[resolution]['labels'] = [
                f"region-{label}" for label in range(1, resolution + 1)]
    return curr_atlas