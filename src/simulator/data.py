'''
simulator/data.py
MAT Deployment data parser.

Created: 2026-02-22
 Author: Maxence Morel Dierckx
'''
import sys
import numpy as np
import scipy.io as sio
from pathlib import Path

HERE = Path(__file__).parent.resolve()
ROOT = HERE.parent.parent

class Data:
    '''The top-level list of deployments to simulate against.'''

    def __init__(self, path_str: str) -> None:
        '''Load the data path as a file or directory and populate batches.'''
        path = Path(path_str)
        if not path.is_absolute():
            path = Path.cwd() / path_str

        self.batches = []
        
        if path.is_file():
            if path.suffix == '.mat':
                mat_file = self.load_mat_file(path)
                if not mat_file:
                    print(f'Data file {path}: unable to load.', file=sys.stderr)
                    exit(1)

                errors = self.validate(mat_file)
                if not errors:
                    self.batches = [mat_file]
                else:
                    errors_str = '\n'.join(errors)
                    print(f'Data file {path} has errors:\n{errors_str}', file=sys.stderr)
                    exit(1)
            else:
                print(f'Data file {path} is not a .mat file.', file=sys.stderr)
                exit(1)

        elif path.is_dir():
            files = [file_path for file_path in path.rglob('*.mat')]
            if len(files) == 0:
                print(f'Data directory {path} does not contain any .mat files.', file=sys.stderr)
                exit(1)
            else:
                for file in files:
                    mat_file = self.load_mat_file(file)
                    if not mat_file:
                        print(f'Data file {file}: unable to load.')
                        ignore = input('Ignore this file and continue? [Y/n] ')
                        if ''.join(ignore.lower().split()) in ['n', 'no']: # split-join removes all whitespace
                            exit(1)

                    errors = self.validate(mat_file)
                    if not errors:
                        self.batches.append(mat_file)
                    else:
                        errors_str = '\n'.join(errors)
                        print(f'Data file {file} has errors:\n{errors_str}')
                        ignore = input('Ignore this file and continue? [Y/n] ')
                        if ''.join(ignore.lower().split()) in ['n', 'no']:
                            exit(1)

        else:
            print(f'Data path {path} not found.', file=sys.stderr)
            exit(1)

    
    def __iter__(self) -> list:
        '''Return batches.'''
        return iter(self.batches)
    
    
    def load_mat_file(self, path: Path) -> dict[str, np.ndarray] | None:
        '''Load a single mat file into a dict of variable to numpy arrays.'''
        try:
            data = sio.loadmat(path, squeeze_me=True)
            data = {k: v for k, v in data.items() if not k.startswith('__')}
            return data
        except ValueError as e:
            print(f'Data file {path} is not a valid MATLAB file: {e}', file=sys.stderr)
        except Exception as e:
            print(f'Unexpected error in load_mat_file: {e}', file=sys.stderr)
        return None
    
    def validate(self, data: dict[str, np.ndarray]) -> list[str]:
        '''Validate that a given mat file contains the necessary keys and shapes.'''
        errors: list[str] = []
        data_lengths: dict[str, int] = {}
        
        for key in ['A', 'M', 'Aw', 'Mw', 'p']:

            if key not in data:
                errors.append(f'Missing variable {key}')

            else:
                shape = data[key].shape
                data_lengths[key] = shape[0]
                mode_data_length = max(set(data_lengths.values()), key=list(data_lengths.values()).count)

                if key == 'p':
                    if len(shape) != 1:
                        errors.append(f'Wrong shape {shape} for key p: expected ({mode_data_length},)')

                else: # A, M, Aw, Mw
                    if len(shape) != 2 or shape[-1] != 3:
                        errors.append(f'Wrong shape {shape} for key {key}: expected ({mode_data_length}, 3)')
        
        unique_data_lengths = set(data_lengths.values())
        if len(unique_data_lengths) != 1:
            errors.append(f'Inconsistent data lengths {unique_data_lengths}')
        
        return errors
