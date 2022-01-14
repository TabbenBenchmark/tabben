"""
A useful script for running multiple dataset processing scripts together. This
script assumes that the data directory contains a directory for each dataset (by
name).

The 'gh' cli can be used afterwards to upload all the assets in the output folder
to the correct GitHub release as follows:
```shell
gh auth login
gh release upload <version-tag> <files>... --clobber
```
"""

import argparse
import os
import subprocess
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description='Automatically run data processing scripts for multiple datasets',
    )
    
    parser.add_argument(
        '--data-dir', '-d', dest='data_directory',
        default=None,
        help='Directory for cached data outputs',
    )
    parser.add_argument(
        '--output-directory', '-o', dest='output_directory',
        default=None,
        help='Output directory to save processed outputs'
    )
    parser.add_argument(
        'names', nargs='*',
        help='Names of the datasets to process (if none, all scripts are run)',
    )
    parser.add_argument(
        '--exclude', '-e', nargs='+', default=[],
        help='Names of datasets to exclude from processing',
    )
    parser.add_argument(
        '--no-profile', '-np', nargs='*', default=None,
        help='Don\'t run profiles for all or a select set of datasets',
    )
    
    parser.add_argument(
        '--upload', action='store_true',
        help='Whether to try to use the `gh` cli to upload assets in the output directory',
    )
    
    parser.add_argument(
        '--python', default='python3',
        help='Name of the python executable',
    )
    
    return parser.parse_args()


def collect_dataset_scripts(config):
    scripts = {}
    
    # verify paths/directories
    scripts_directory = Path(os.path.dirname(os.path.abspath(__file__)))
    assert scripts_directory.exists()
    
    if config.output_directory is None:
        if config.data_directory is not None:
            config.output_directory = Path(config.data_directory) / 'output'
        else:
            config.output_directory = Path.cwd() / 'output'
    else:
        config.output_directory = Path(config.output_directory)
    
    if not config.output_directory.exists():
        config.output_directory.mkdir(parents=True, exist_ok=True)
    
    # when no dataset names are given, run all scripts
    if len(config.names) == 0:
        filenames = [file.name.partition('.')[0] for file in scripts_directory.glob('*.py')]
        config.names = [name for name in filenames
                        if not name.startswith('utils') and not name.startswith('all')]
    
    # collect the arguments for running the script for each dataset
    for name in config.names:
        if name not in config.exclude:
            script_file = scripts_directory / f'{name}.py'
            assert script_file.exists()
            
            scripts[name] = [config.python, script_file, config.output_directory]
            if config.data_directory is not None:
                scripts[name].extend(['-s', Path(config.data_directory) / name])
            
            if config.no_profile is not None:
                if len(config.no_profile) == 0 or name in config.no_profile:
                    scripts[name].append('--no-profile')
    
    return scripts


def run_scripts(scripts):
    successes = set()
    
    for index, (name, script_args) in enumerate(scripts.items()):
        print(f'\u001b[34mRunning the script for the {name} dataset...\u001b[0m')
        script_result = subprocess.run(script_args)
        if script_result.returncode == 0:
            successes.add(name)
        else:
            print(f'\u001b[31mRan into an issue for the {name} dataset (see above).\u001b[0m')
    
    print(f'Successfully processed {len(successes)} out of {len(scripts)} datasets.')
    if len(successes) != len(scripts):
        print(f'Failed datasets: {", ".join(set(scripts.keys()) - successes)}')


def upload_assets(config):
    asset_files = [str(file) for file in config.output_directory.glob('*')]
    command = [
        'gh', 'release', 'upload', 'v0.0.7-pre',
        *asset_files,
        '--clobber',
    ]
    subprocess.run(command)


if __name__ == '__main__':
    config = parse_args()
    scripts = collect_dataset_scripts(config)
    run_scripts(scripts)
    
    if config.upload:
        upload_assets(config)
