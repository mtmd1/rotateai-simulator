'''
simulator/config.py
Parser for configuration of the microcontroller constants.

Created: 2026-02-21
 Author: Maxence Morel Dierckx
'''
import tomllib
from pathlib import Path


HERE = Path(__file__).parent.resolve()
ROOT = HERE.parent.parent

class Config:
    '''A configuration object.'''

    def __init__(self, file_path_str: str) -> Config:
        file_path = ROOT / file_path_str
        data = tomllib.load(file_path)
        
        # TODO: load params from config dict
