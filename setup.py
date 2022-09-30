from setuptools import setup

setup(name='fmriprep-denoise-benchmark',
      version='0.0.1',
      description='Benchmarking fmriprep denoising strategies on functional connectomes.',
      url='https://github.com/SIMEXP/fmriprep-denoise-benchmark',
      author='Hao-Ting Wang',
      license='MIT',
      packages=['fmriprep_denoise'],
      entry_points = {
            'console_scripts': [
                  "make_timeseries = fmriprep_denoise.dataset.make_timeseries:main",
                  "calculate_degrees_of_freedom = fmriprep_denoise.features.calculate_degrees_of_freedom:main",
                  "build_features = fmriprep_denoise.features.build_features:main",
            ],
      },
      zip_safe=False)
