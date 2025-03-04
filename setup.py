from setuptools import find_packages, setup

_install_requires = [
    'fflogsapi',
    'pyyaml',
    'tqdm',
    'jupyter',
	'scikit-learn',
	'pandas',
	'numpy',
	'opencv-python',
	'matplotlib',
	'scipy',
	'selenium',
	'torch',
	'torchvision',
	'matplotlib',
]

_package_excludes = [
    '*.tests'
]

setup (
    name='nina',
    version='0.0.1',
    packages=find_packages(exclude=_package_excludes),
    install_requires=_install_requires,
    python_requires='>=3.11'
)