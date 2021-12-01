import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='tabben',
    version='0.0.2-dev',
    description='A package for working with datasets from the open benchmark for tabular data',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/umd-otb/OpenTabularDataBenchmark',
    project_urls={
        'Bug Tracker': 'https://github.com/umd-otb/OpenTabularDataBenchmark/issues',
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    package_dir={'': 'src'},
    packages=setuptools.find_packages(where='src'),
    python_requires=">=3.8",
    include_package_data=True,
    zip_safe=False,
    install_requires=[
        'numpy',
        'requests',
        'requests-file',
        'scikit-metrics',
        'toml',
        'torch',
        'tqdm',
    ]
)