import os
import contextlib
from pathlib import Path

from fmriprep_denoise.visualization import utils


@contextlib.contextmanager
def working_directory(path):
    """Changes working directory and returns to previous on exit."""
    prev_cwd = Path.cwd()
    os.chdir(path)
    try:
        path_root = utils.get_data_root() / "denoise-metrics"
        print(path_root)
    finally:
        os.chdir(prev_cwd)


if __name__ == "__main__":
    path = Path(__file__).parents[1] / "content" / "docs"
    working_directory(path)
