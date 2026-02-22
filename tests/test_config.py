'''
tests/test_config.py
Unit tests for simulator/config.py.

Created: 2026-02-22
 Author: Maxence Morel Dierckx
'''
import pytest
try:
    import tomllib
except ImportError:
    import tomli as tomllib

from simulator.config import Config


# MARK: fixtures

VALID_TOML = '''\
[config]
sample_rate = 5
voltage = 1.8
DMIPS_per_MHz = 1.5
safety_margin = 1.5
'''

NESTED_TOML = '''\
[sensor]
sample_rate = 10

[hardware]
voltage = 3.3
DMIPS_per_MHz = 2.0

[runtime]
safety_margin = 1.2
'''

@pytest.fixture
def valid_config(tmp_path) -> str:
    '''Generate a valid config TOML and return its path.'''
    path = tmp_path / 'valid.toml'
    path.write_text(VALID_TOML)
    return str(path)


@pytest.fixture
def nested_config(tmp_path) -> str:
    '''Generate a valid config with keys spread across sections.'''
    path = tmp_path / 'nested.toml'
    path.write_text(NESTED_TOML)
    return str(path)


@pytest.fixture
def missing_keys_config(tmp_path) -> str:
    '''Generate a config missing required keys.'''
    path = tmp_path / 'missing.toml'
    path.write_text('[config]\nsample_rate = 5\n')
    return str(path)


@pytest.fixture
def bad_values_config(tmp_path) -> str:
    '''Generate a config with invalid values.'''
    path = tmp_path / 'bad_values.toml'
    path.write_text('''\
[config]
sample_rate = -1
voltage = 0
DMIPS_per_MHz = 1.5
safety_margin = 0.5
''')
    return str(path)


# MARK: get_nested_value

class TestGetNestedValue:
    '''Tests for the get_nested_value helper.'''

    def _config(self):
        '''Return a Config instance without running __init__.'''
        return object.__new__(Config)

    def test_finds_top_level_key(self):
        d = {'sample_rate': 5, 'voltage': 1.8}
        assert self._config().get_nested_value(d, 'sample_rate') == 5

    def test_finds_nested_key(self):
        d = {'config': {'sample_rate': 5}}
        assert self._config().get_nested_value(d, 'sample_rate') == 5

    def test_finds_deeply_nested_key(self):
        d = {'a': {'b': {'voltage': 3.3}}}
        assert self._config().get_nested_value(d, 'voltage') == 3.3

    def test_returns_none_for_missing_key(self):
        d = {'config': {'sample_rate': 5}}
        assert self._config().get_nested_value(d, 'voltage') is None

    def test_returns_first_occurrence(self):
        d = {'sample_rate': 1, 'config': {'sample_rate': 2}}
        assert self._config().get_nested_value(d, 'sample_rate') == 1


# MARK: validate

class TestValidate:
    '''Tests for the validate method (called on parsed TOML dicts).'''

    def _config(self):
        return object.__new__(Config)

    def _make_config(self, **overrides):
        '''Build a valid flat config dict, then apply overrides.'''
        base = {'sample_rate': 5, 'voltage': 1.8, 'DMIPS_per_MHz': 1.5, 'safety_margin': 1.5}
        base.update(overrides)
        return base

    def test_valid_config_no_errors(self):
        errors = self._config().validate(self._make_config())
        assert errors == []

    def test_missing_single_key(self):
        cfg = self._make_config()
        del cfg['voltage']
        errors = self._config().validate(cfg)
        assert any('Missing variable voltage' in e for e in errors)

    def test_missing_multiple_keys(self):
        errors = self._config().validate({})
        for key in ['sample_rate', 'voltage', 'DMIPS_per_MHz', 'safety_margin']:
            assert any(f'Missing variable {key}' in e for e in errors)

    def test_negative_value_rejected(self):
        errors = self._config().validate(self._make_config(sample_rate=-1))
        assert any('sample_rate must be positive' in e for e in errors)

    def test_zero_value_rejected(self):
        errors = self._config().validate(self._make_config(voltage=0))
        assert any('voltage must be positive' in e for e in errors)

    def test_safety_margin_below_one_rejected(self):
        errors = self._config().validate(self._make_config(safety_margin=0.5))
        assert any('safety_margin must be >= 1.0' in e for e in errors)

    def test_safety_margin_exactly_one_valid(self):
        errors = self._config().validate(self._make_config(safety_margin=1))
        assert not any('safety_margin' in e for e in errors)

    def test_non_numeric_value_rejected(self):
        errors = self._config().validate(self._make_config(voltage='high'))
        assert any('voltage must be a number' in e for e in errors)

    def test_validates_nested_config(self):
        cfg = tomllib.loads(NESTED_TOML)
        errors = self._config().validate(cfg)
        assert errors == []


# MARK: extract_values

class TestExtractValues:
    '''Tests for the extract_values method.'''

    def _config(self):
        return object.__new__(Config)

    def test_extracts_flat_config(self):
        cfg = {'sample_rate': 5, 'voltage': 1.8, 'DMIPS_per_MHz': 1.5, 'safety_margin': 1.5}
        assert self._config().extract_values(cfg) == (5, 1.8, 1.5, 1.5)

    def test_extracts_nested_config(self):
        cfg = tomllib.loads(NESTED_TOML)
        assert self._config().extract_values(cfg) == (10, 3.3, 2.0, 1.2)

    def test_returns_tuple(self):
        cfg = {'sample_rate': 5, 'voltage': 1.8, 'DMIPS_per_MHz': 1.5, 'safety_margin': 1.5}
        result = self._config().extract_values(cfg)
        assert isinstance(result, tuple)
        assert len(result) == 4


# MARK: init

class TestInit:
    '''Tests for constructing Config from a TOML file.'''

    def test_loads_valid_config(self, valid_config):
        c = Config(valid_config)
        assert c.sample_rate == 5
        assert c.voltage == 1.8
        assert c.DMIPS_per_MHz == 1.5
        assert c.safety_margin == 1.5

    def test_loads_nested_config(self, nested_config):
        c = Config(nested_config)
        assert c.sample_rate == 10
        assert c.voltage == 3.3
        assert c.DMIPS_per_MHz == 2.0
        assert c.safety_margin == 1.2

    def test_nonexistent_file_exits(self, tmp_path):
        with pytest.raises(SystemExit):
            Config(str(tmp_path / 'nope.toml'))

    def test_validation_failure_exits(self, missing_keys_config):
        with pytest.raises(SystemExit):
            Config(missing_keys_config)

    def test_bad_values_exit(self, bad_values_config):
        with pytest.raises(SystemExit):
            Config(bad_values_config)
