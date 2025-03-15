from setuptools import find_packages, setup

_install_requires = [
    'pyyaml',
    'pyqt5',
    'tqdm',
    'jupyter',
    'scikit-learn',
    'pandas',
    'numpy',
    'matplotlib',
    'scipy',
    'selenium',
    'torch',
    'torchvision',
    'matplotlib',
    'lxml',
    'label-studio',
    'kagglehub'
]

_package_excludes = [
    '*.tests'
]

setup(
    name='nino',
    version='0.0.1',
    packages=find_packages(exclude=_package_excludes),
    install_requires=_install_requires,
    python_requires='>=3.11'
)
