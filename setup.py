# from setuptools import setup

# setup(
#     name="fmriprep-denoise-benchmark",
#     version="0.0.1",
#     description="Benchmarking fmriprep denoising strategies on functional connectomes.",
#     url="https://github.com/SIMEXP/fmriprep-denoise-benchmark",
#     author="Hao-Ting Wang",
#     license="MIT",
#     packages=["fmriprep_denoise"],
#     entry_points={
#         "console_scripts": [
#             "make_timeseries = fmriprep_denoise.dataset.make_timeseries:main",
#             "calculate_degrees_of_freedom = fmriprep_denoise.features.calculate_degrees_of_freedom:main",
#             "build_features = fmriprep_denoise.features.build_features:main",
#             "summarise_metadata = fmriprep_denoise.visualization.summarise_metadata:main"
#         ],
#     },
#     zip_safe=False,
# )
from setuptools import setup, find_packages

setup(
    name="fmriprep-denoise-benchmark",
    version="0.0.1",
    description="Benchmarking fmriprep denoising strategies on functional connectomes.",
    url="https://github.com/SIMEXP/fmriprep-denoise-benchmark",
    author="Hao-Ting Wang",
    license="MIT",
    packages=find_packages(),  # originally was ["fmriprep_denoise"],
    entry_points={
        "console_scripts": [
            "make_timeseries = fmriprep_denoise.dataset.make_timeseries:main",
            "calculate_degrees_of_freedom = fmriprep_denoise.features.calculate_degrees_of_freedom:main",
            "build_features = fmriprep_denoise.features.build_features:main",
            "summarise_metadata = fmriprep_denoise.visualization.summarise_metadata:main",
            "calculate_degrees_of_freedom_test = fmriprep_denoise.features.calculate_degrees_of_freedom_test:main",
            "build_features_test = fmriprep_denoise.features.build_features_test:main",
            "evaluate_NaN = fmriprep_denoise.features.evaluate_NaN:main"
        ],
    },
    zip_safe=False,
)