'''
tests/test_data.py
Unit tests for simulator/data.py

Created: 2026-02-22
 Author: Maxence Morel Dierckx
'''
import pytest
import numpy as np
import scipy.io as sio
from simulator.data import Data


# MARK: fixtures

@pytest.fixture
def valid_mat(tmp_path) -> str:
    '''Generate a temporary valid .mat file and return its path'''
    data = {
        'A': np.random.rand(100, 3),
        'M': np.random.rand(100, 3),
        'Aw': np.random.rand(100, 3),
        'Mw': np.random.rand(100, 3),
        'p': np.random.rand(100),
    }
    path = tmp_path / 'valid.mat'
    sio.savemat(path, data)
    return str(path)


@pytest.fixture
def bad_lengths_mat(tmp_path) -> str:
    '''Generate a temporary .mat file with inconsistent lengths and return its path.'''
    data = {
        'A': np.random.rand(100, 3),
        'M': np.random.rand(90, 3),
        'Aw': np.random.rand(100, 3),
        'Mw': np.random.rand(110, 3),
        'p': np.random.rand(100),
    }
    path = tmp_path / 'bad_lengths.mat'
    sio.savemat(path, data)
    return str(path)


@pytest.fixture
def bad_shapes_mat(tmp_path) -> str:
    '''Generate a temporary .mat file with wrong shapes and return its path.'''
    data = {
        'A': np.random.rand(100, 3),
        'M': np.random.rand(100, 2, 3),
        'Aw': np.random.rand(100, 3),
        'Mw': np.random.rand(100, 2),
        'p': np.random.rand(100),
    }
    path = tmp_path / 'bad_shapes.mat'
    sio.savemat(path, data)
    return str(path)


@pytest.fixture
def missing_keys_mat(tmp_path) -> str:
    '''Generate a .mat file missing required keys.'''
    data = {
        'A': np.random.rand(100, 3),
        'p': np.random.rand(100),
    }
    path = tmp_path / 'missing_keys.mat'
    sio.savemat(path, data)
    return str(path)


@pytest.fixture
def valid_mat_dir(tmp_path) -> str:
    '''Generate a directory with multiple valid .mat files.'''
    for i in range(3):
        n = 50 + i * 10
        data = {
            'A': np.random.rand(n, 3),
            'M': np.random.rand(n, 3),
            'Aw': np.random.rand(n, 3),
            'Mw': np.random.rand(n, 3),
            'p': np.random.rand(n),
        }
        sio.savemat(tmp_path / f'file_{i}.mat', data)
    return str(tmp_path)


# MARK: validate

class TestValidate:
    '''Tests for the validate method (called on already-loaded dicts).'''

    def _make_data(self, **overrides):
        '''Build a valid data dict, then apply overrides.'''
        base = {
            'A': np.random.rand(100, 3),
            'M': np.random.rand(100, 3),
            'Aw': np.random.rand(100, 3),
            'Mw': np.random.rand(100, 3),
            'p': np.random.rand(100),
        }
        base.update(overrides)
        return base

    def _validator(self):
        '''Return a Data instance without running __init__.'''
        obj = object.__new__(Data)
        return obj

    def test_valid_data_no_errors(self):
        errors = self._validator().validate(self._make_data())
        assert errors == []

    def test_missing_single_key(self):
        data = self._make_data()
        del data['Mw']
        errors = self._validator().validate(data)
        assert any('Missing variable Mw' in e for e in errors)

    def test_missing_multiple_keys(self):
        data = self._make_data()
        del data['A']
        del data['M']
        del data['p']
        errors = self._validator().validate(data)
        assert any('Missing variable A' in e for e in errors)
        assert any('Missing variable M' in e for e in errors)
        assert any('Missing variable p' in e for e in errors)

    def test_wrong_shape_matrix_not_nx3(self):
        data = self._make_data(M=np.random.rand(100, 4))
        errors = self._validator().validate(data)
        assert any('Wrong shape' in e and 'M' in e for e in errors)

    def test_wrong_shape_matrix_3d(self):
        data = self._make_data(Aw=np.random.rand(100, 2, 3))
        errors = self._validator().validate(data)
        assert any('Wrong shape' in e and 'Aw' in e for e in errors)

    def test_wrong_shape_p_not_1d(self):
        data = self._make_data(p=np.random.rand(100, 1))
        errors = self._validator().validate(data)
        assert any('Wrong shape' in e and 'p' in e for e in errors)

    def test_inconsistent_lengths(self):
        data = self._make_data(M=np.random.rand(50, 3))
        errors = self._validator().validate(data)
        assert any('Inconsistent data lengths' in e for e in errors)

    def test_extra_keys_ignored(self):
        data = self._make_data(fs=5, roll=np.random.rand(100))
        errors = self._validator().validate(data)
        assert errors == []


# MARK: load

class TestLoadMatFile:
    '''Tests for load_mat_file.'''

    def _loader(self):
        return object.__new__(Data)

    def test_loads_valid_file(self, valid_mat):
        result = self._loader().load_mat_file(valid_mat)
        assert result is not None
        assert 'A' in result
        assert 'p' in result

    def test_strips_dunder_keys(self, valid_mat):
        result = self._loader().load_mat_file(valid_mat)
        assert all(not k.startswith('__') for k in result)

    def test_returns_none_for_nonexistent_file(self, tmp_path):
        result = self._loader().load_mat_file(tmp_path / 'no_such.mat')
        assert result is None

    def test_returns_none_for_corrupt_file(self, tmp_path):
        bad = tmp_path / 'corrupt.mat'
        bad.write_text('this is not matlab')
        result = self._loader().load_mat_file(bad)
        assert result is None


# MARK: init file

class TestInitSingleFile:
    '''Tests for constructing Data from a single .mat file.'''

    def test_loads_valid_file(self, valid_mat):
        d = Data(valid_mat)
        assert len(d.batches) == 1
        assert 'A' in d.batches[0]

    def test_source_key_set(self, valid_mat):
        d = Data(valid_mat)
        assert '_source' in d.batches[0]
        assert d.batches[0]['_source'] == 'valid.mat'

    def test_nonexistent_path_exits(self, tmp_path):
        with pytest.raises(SystemExit):
            Data(str(tmp_path / 'does_not_exist.mat'))

    def test_non_mat_extension_exits(self, tmp_path):
        csv = tmp_path / 'data.csv'
        csv.write_text('a,b,c')
        with pytest.raises(SystemExit):
            Data(str(csv))

    def test_corrupt_file_exits(self, tmp_path):
        bad = tmp_path / 'bad.mat'
        bad.write_text('not matlab')
        with pytest.raises(SystemExit):
            Data(str(bad))

    def test_validation_failure_single_file_exits(self, missing_keys_mat):
        with pytest.raises((SystemExit)):
            Data(missing_keys_mat)


# MARK: init dir

class TestInitDirectory:
    '''Tests for constructing Data from a directory of .mat files.'''

    def test_loads_all_files(self, valid_mat_dir):
        d = Data(valid_mat_dir)
        assert len(d.batches) == 3

    def test_source_keys_set_from_filenames(self, valid_mat_dir):
        d = Data(valid_mat_dir)
        sources = sorted(b['_source'] for b in d.batches)
        assert sources == ['file_0.mat', 'file_1.mat', 'file_2.mat']

    def test_empty_directory_exits(self, tmp_path):
        empty = tmp_path / 'empty'
        empty.mkdir()
        with pytest.raises(SystemExit):
            Data(str(empty))

    def test_directory_no_mat_files_exits(self, tmp_path):
        (tmp_path / 'readme.txt').write_text('hello')
        with pytest.raises(SystemExit):
            Data(str(tmp_path))


# MARK: iter

class TestIter:
    def test_iterates_over_batches(self, valid_mat):
        d = Data(valid_mat)
        batches = list(d)
        assert len(batches) == 1
        assert 'A' in batches[0]

    def test_multiple_batches(self, valid_mat_dir):
        d = Data(valid_mat_dir)
        batches = list(d)
        assert len(batches) == 3
