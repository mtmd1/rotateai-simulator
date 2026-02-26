'''
tests/conftest.py
Shared fixtures for the test suite.

Created: 2026-02-26
 Author: Maxence Morel Dierckx
'''
import pytest
import subprocess
from pathlib import Path
from unittest.mock import MagicMock

from simulator.config import Config


FIXTURES = Path(__file__).parent / 'fixtures'
REPEATER_SRC = FIXTURES / 'repeater.c'
TINY_MAT = FIXTURES / 'mn11_157aprh_tiny.mat'


@pytest.fixture(scope='session')
def repeater(tmp_path_factory) -> Path:
    '''Compile the repeater binary from tests/fixtures/repeater.c.'''
    if not REPEATER_SRC.is_file():
        pytest.skip('tests/fixtures/repeater.c not found')

    out = tmp_path_factory.mktemp('bin') / 'repeater'
    result = subprocess.run(
        ['gcc', str(REPEATER_SRC), '-o', str(out)],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        pytest.skip(f'Failed to compile repeater: {result.stderr}')

    return out


@pytest.fixture(scope='session')
def config(tmp_path_factory) -> Config:
    '''Create a real Config from a temporary TOML file.'''
    path = tmp_path_factory.mktemp('config') / 'test.toml'
    path.write_text('''\
[config]
sample_rate = 5
voltage = 1.8
DMIPS_per_MHz = 1.5
uA_per_MHz = 51.6
max_frequency = 160
''')
    return Config(str(path))


def mock_benchmarker():
    '''Return a mock Benchmarker with realistic attributes.'''
    mock = MagicMock()
    mock.file_size = 1024
    mock.peak_memory = 4096
    mock.total_instructions = 1_000_000
    mock.total_flops = 50_000
    mock.collect = MagicMock()
    return mock
