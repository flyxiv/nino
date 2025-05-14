from setuptools import find_packages, setup

_install_requires = [
    'pyyaml',
    'pyqt5',
    'tqdm',
    'jupyter',
    'scikit-learn',
    'diffusers',
    'transformers',
    'pandas',
    'numpy',
    'matplotlib',
    'scipy',
    'selenium',
    'matplotlib',
    'lxml',
    'label-studio',
    'pycocotools',
    'kagglehub',
    'datasets',
    'ultralytics',
    'google-cloud-storage',
    'optuna',
    'plotly',
]

_package_excludes = [
    '*.tests'
]

setup(
    name='nino',
    version='0.0.1',
    packages=find_packages(exclude=_package_excludes),
    install_requires=_install_requires,
    python_requires='<=3.11'
)
