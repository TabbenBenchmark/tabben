from pathlib import Path
import setuptools

# load local README as long description text
long_description = Path('README.md').read_text(encoding='utf-8')

# package dependencies and required versions
dependencies = [
    'numpy',
    'torch',
    'tqdm',
    'requests',
    'toml',
]

# package setup
setuptools.setup(
    name='otb',
    version='0.1.0',
    description='A package for working with datasets from the open benchmark for tabular data',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/nurostream/cmsc473',
    project_urls={
        'Bug Tracker': 'https://github.com/nurostream/cmsc473/issues',
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    package_dir={
        '': 'src'
    },
    packages=setuptools.find_packages(where='src'),
    python_requires='>=3.8',
    install_requires=dependencies,
    include_package_data=True,
)

