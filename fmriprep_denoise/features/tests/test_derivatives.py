"""Test some private functions."""
from pathlib import Path
from fmriprep_denoise.features import derivatives
import pandas as pd
import numpy as np
import pytest


fake_timeseries_template = (
    'task-test_space-MNI152NLin6Asym_'
    'atlas-test_nroi-100_desc-test_timeseries.tsv'
)


def _make_fake_timeseries_collection(extracted_path, empty_file):
    """Make test data."""
    for i in range(10):
        fake_subject_dir = extracted_path / 'atlas-test' / f'sub-{i+1:03d}'
        fake_subject = f'sub-{i+1:03d}_{fake_timeseries_template}'
        fake_subject_dir.mkdir(parents=True, exist_ok=True)

        if empty_file and i == 7:  # one invalid file
            (fake_subject_dir / fake_subject).touch(exist_ok=True)
        else:
            df = pd.DataFrame(np.random.uniform(0, 1, size=(200, 100)))
            df.to_csv(fake_subject_dir / fake_subject, sep='\t', index=False)
    subjects = [f'sub-{i+1:03d}' for i in range(10)]
    return extracted_path, subjects


def test_load_valid_timeseries(tmp_path):
    """Test time series data loader."""
    extracted_path, subjects = _make_fake_timeseries_collection(
        tmp_path, empty_file=True
    )
    valid_ids, valid_ts = derivatives._load_valid_timeseries(
        atlas='test',
        extracted_path=extracted_path,
        participant_id=subjects,
        file_pattern='desc-test',
    )
    assert 'sub-008' not in valid_ids
    assert len(valid_ts) == len(valid_ids)
    assert valid_ts[0].shape == (200, 100)


@pytest.mark.parametrize(
    'strategy_name,fd_thresh',
    [
        ('minimal', None),
        ('stringent', 0.2),
        (None, None),
    ],
)
def test_get_qc_criteria(strategy_name, fd_thresh):
    motion_qc = derivatives.get_qc_criteria(strategy_name)
    assert type(motion_qc) is dict
    assert motion_qc['fd_thresh'] == fd_thresh
