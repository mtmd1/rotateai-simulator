'''
simulator/config.py
Parser for configuration of the microcontroller constants.

Created: 2026-02-21
 Author: Maxence Morel Dierckx
'''
import sys
try:
    import tomllib
except ImportError:
    import tomli as tomllib
from pathlib import Path


HERE = Path(__file__).parent.resolve()
ROOT = HERE.parent.parent

class Config:
    '''A configuration object.'''

    def __init__(self, config_path_str: str) -> None:
        '''Load the configuration from the given path and validate it.'''
        config_path = Path(config_path_str)
        if not config_path.is_absolute():
            config_path = Path.cwd() / config_path_str
        try:
            with config_path.open('rb') as f:
                data = tomllib.load(f)
        except FileNotFoundError as e:
            print(f'Config file {config_path} not found.', file=sys.stderr)
            exit(1)

        errors = self.validate(data)

        if not errors:
            values = self.extract_values(data)
            (self.sample_rate, self.voltage, self.DMIPS_per_MHz, self.uA_per_MHz, self.safety_margin) = values
        
        else:
            errors_str = '\n'.join(errors)
            print(f'Config file {config_path} has errors:\n{errors_str}', file=sys.stderr)
            exit(1)


    def validate(self, config: dict) -> list[str]:
        '''Validate that a given config contains the necessary keys with valid values.'''
        errors: list[str] = []
        
        for key in ['sample_rate', 'voltage', 'DMIPS_per_MHz', 'uA_per_MHz', 'safety_margin']:

            value = self.get_nested_value(config, key)
            
            if value is None:
                errors.append(f'Missing variable {key}')
            
            else:
                try:
                    if key == 'safety_margin':
                        if value < 1:
                            errors.append('safety_margin must be >= 1.0')
                    else:
                        if value <= 0:
                            errors.append(f'{key} must be positive')

                except TypeError:
                    errors.append(f'{key} must be a number')

        return errors

    
    def extract_values(self, config: dict) -> tuple[float | int]:
        '''Extract the expected keys recursively'''
        values: list[float | int] = []
        
        for key in ['sample_rate', 'voltage', 'DMIPS_per_MHz', 'uA_per_MHz', 'safety_margin']:
            values.append(self.get_nested_value(config, key))
        
        return tuple(values)


    def get_nested_value(self, d: dict, key: str):
        '''Extract the value of a key anywhere in a nested dict. Returns the first occurence.'''
        for k in d:
            if k == key:
                return d[k]
            elif isinstance(d[k], dict):
                result = self.get_nested_value(d[k], key)
                if result is not None:
                    return result

    
    def to_dict(self) -> dict[str, float | int]:
        '''Return the config values as a flat dict.'''
        return {
            'sample_rate': self.sample_rate,
            'voltage': self.voltage,
            'DMIPS_per_MHz': self.DMIPS_per_MHz,
            'uA_per_MHz': self.uA_per_MHz,
            'safety_margin': self.safety_margin,
        }
