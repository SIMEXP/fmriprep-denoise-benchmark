import pandas as pd

from nilearn.signal import clean
from nilearn.interfaces.fmriprep import load_confounds_strategy, load_confounds

from fmriprep_denoise.dataset.atlas import (
    create_atlas_masker,
    get_atlas_dimensions,
)


def generate_timeseries_per_dimension(
    atlas_name, output, benchmark_strategies, data_aroma, data
):
    """
    Get confounds and sample mask.

    Parameters
    ----------

    atlas_name : str
        Name of the atlas.

    output : pathlib.Path
        Output directory for the generated timeseries.

    benchmark_strategies : dict
        Denoising strategy collection.

    data_aroma : sklearn.utils.Bunch
        fMRIPRep output collection for functional data corresponding to
        ICA-AROMA outputs.

    data : sklearn.utils.Bunch
        fMRIPRep output collection for functional data outputs.
    """
    dimensions = get_atlas_dimensions(atlas_name)
    for dimension in dimensions:
        print(f'-- {atlas_name}: dimension {dimension} --')
        print('raw time series')
        atlas_info = {'atlas_name': atlas_name, 'dimension': dimension}
        subject_timeseries = _generate_raw_timeseries(output, data, atlas_info)

        for strategy_name, parameters in benchmark_strategies.items():
            print(f'Denoising: {strategy_name}')
            print(parameters)
            if 'aroma' in strategy_name:
                _clean_timeserise_aroma(
                    atlas_name,
                    dimension,
                    strategy_name,
                    parameters,
                    output,
                    data_aroma,
                )
            else:
                _clean_timeserise_normal(
                    subject_timeseries,
                    atlas_name,
                    dimension,
                    strategy_name,
                    parameters,
                    output,
                    data,
                )


def get_confounds(strategy_name, parameters, img):
    """
    Get confounds and sample mask.

    Parameters
    ----------

    strategy_name : str
        Denoise strategy name.

    parameters : dict
        Denoise parameter passed to load_confounds or load_confounds_strategy.

    img : str
        Path of the processed functional image to be denoised.

    Returns
    -------
    See docs of load_confounds

    """
    if strategy_name == 'baseline':
        reduced_confounds, sample_mask = load_confounds(img, **parameters)
    else:
        reduced_confounds, sample_mask = load_confounds_strategy(
            img, **parameters
        )
    return reduced_confounds, sample_mask


def _clean_timeserise_normal(
    subject_timeseries,
    atlas_name,
    dimension,
    strategy_name,
    parameters,
    output,
    data,
):
    """Denoise timeseries of regular functional processed output."""
    atlas_spec = f'atlas-{atlas_name}_nroi-{dimension}'
    _, img, ts_path = _get_output_info(strategy_name, output, data, atlas_spec)
    reduced_confounds, sample_mask = get_confounds(
        strategy_name, parameters, img
    )
    if _check_exclusion(reduced_confounds, sample_mask):
        clean_timeseries = []
    else:
        clean_timeseries = clean(
            subject_timeseries,
            detrend=True,
            standardize=True,
            sample_mask=sample_mask,
            confounds=reduced_confounds,
        )
    clean_timeseries = pd.DataFrame(clean_timeseries)
    clean_timeseries.to_csv(ts_path, sep='\t', index=False)


def _clean_timeserise_aroma(
    atlas_name, dimension, strategy_name, parameters, output, data_aroma
):
    """Denoise timeseries of ICA-AROMA processed output."""
    atlas_spec = f'atlas-{atlas_name}_nroi-{dimension}'
    subject_mask, img, ts_path = _get_output_info(
        strategy_name, output, data_aroma, atlas_spec
    )
    reduced_confounds, sample_mask = get_confounds(
        strategy_name, parameters, img
    )
    aroma_masker, _ = create_atlas_masker(
        atlas_name, dimension, subject_mask, standardize=True, nilearn_cache=''
    )
    clean_timeseries = aroma_masker.fit_transform(
        img, confounds=reduced_confounds, sample_mask=sample_mask
    )
    clean_timeseries = pd.DataFrame(clean_timeseries)
    clean_timeseries.to_csv(ts_path, sep='\t', index=False)


def _generate_raw_timeseries(output, data, atlas_info):
    """Generate detrended raw time series for a given atlas map."""
    subject_spec, subject_output, subject_mask = _get_subject_info(
        output, data
    )
    rawts_path = subject_output / (
        f"{subject_spec}_atlas-{atlas_info['atlas_name']}_"
        f"nroi-{atlas_info['dimension']}_desc-raw_timeseries.tsv"
    )
    raw_masker, atlas_labels = create_atlas_masker(
        atlas_info['atlas_name'],
        atlas_info['dimension'],
        subject_mask,
        detrend=False,
        nilearn_cache='',
    )
    timeseries_labels = pd.DataFrame(columns=atlas_labels)
    if not rawts_path.is_file():
        subject_timeseries = raw_masker.fit_transform(data.func[0])
        fitted_labels = [int(i) for i in raw_masker.labels_]
        df = pd.DataFrame(subject_timeseries, columns=fitted_labels)
        # make sure missing label were put pack
        df = pd.concat([timeseries_labels, df])
        df.to_csv(rawts_path, sep='\t', index=False)
    else:
        df = pd.read_csv(rawts_path, header=0, sep='\t')
        subject_timeseries = df.values
    del raw_masker

    return subject_timeseries


def _get_output_info(strategy_name, output, data, atlas_spec):
    """Generate output path."""
    subject_spec, subject_output, subject_mask = _get_subject_info(
        output, data
    )
    img = data.func[0]
    ts_path = (
        subject_output
        / f'{subject_spec}_{atlas_spec}_desc-{strategy_name}_timeseries.tsv'
    )
    return subject_mask, img, ts_path


def _check_exclusion(reduced_confounds, sample_mask):
    """For scrubbing based strategy, check if regression can be performed."""
    if sample_mask is not None:
        kept_vol = len(sample_mask)
    else:
        kept_vol = reduced_confounds.shape[0]
    # more noise regressors than volume
    remove = kept_vol < reduced_confounds.shape[1]
    return remove


def _get_subject_info(output, data):
    """Generate subject directory and get specifier and EPI mask."""
    img = data.func[0]

    subject_spec = data.func[0].split('/')[-1].split('_desc-')[0]

    subject_root = img.split(subject_spec)[0]
    subject_id = subject_spec.split('_')[0]

    subject_output = output / subject_id
    subject_output.mkdir(exist_ok=True)

    subject_mask = f'{subject_root}/{subject_spec}_desc-brain_mask.nii.gz'
    return subject_spec, subject_output, subject_mask
