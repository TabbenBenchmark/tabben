"""
This is a setup.py file used *only* for allowing `pip install -e .` to work as a dev
environment. Everything else should use `setup.cfg` (the files need to be manually
kept in sync).
"""

import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='tabben',
    version='0.0.8-dev',
    description='A package for working with datasets from an open benchmark for tabular data',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/TabbenBenchmark/tabben',
    project_urls={
        'Bug Tracker': 'https://github.com/TabbenBenchmark/tabben/issues',
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    package_dir={'': 'src'},
    packages=setuptools.find_packages(where='src'),
    python_requires=">=3.8",
    include_package_data=True,
    zip_safe=False,
    install_requires=[
        'numpy',
        'requests',
        'scikit-metrics',
        'simplejson',
        'toml',
        'tqdm',
    ]
)