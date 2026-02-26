'''
tests/test_integration.py
Integration tests for the data-simulation-report pipeline.

Created: 2026-02-26
 Author: Maxence Morel Dierckx
'''
import json
import pytest
import numpy as np
from unittest.mock import patch

from simulator.data import Data
from simulator.runner import Simulator, SimResult
from simulator.report import save_report, calculate_errors
from conftest import TINY_MAT, mock_benchmarker


# MARK: data loading

class TestDataLoading:
    '''Test that Data loads the real tiny mat file correctly.'''

    def test_loads_tiny_mat(self):
        if not TINY_MAT.is_file():
            pytest.skip('tiny mat fixture not found')
        d = Data(str(TINY_MAT))
        assert len(d) == 1

    def test_batch_has_required_keys(self):
        if not TINY_MAT.is_file():
            pytest.skip('tiny mat fixture not found')
        d = Data(str(TINY_MAT))
        batch = d.batches[0]
        for key in ['A', 'M', 'Aw', 'Mw', 'p', '_source']:
            assert key in batch

    def test_batch_shapes_consistent(self):
        if not TINY_MAT.is_file():
            pytest.skip('tiny mat fixture not found')
        d = Data(str(TINY_MAT))
        batch = d.batches[0]
        n = len(batch['p'])
        assert batch['A'].shape == (n, 3)
        assert batch['M'].shape == (n, 3)
        assert batch['Aw'].shape == (n, 3)
        assert batch['Mw'].shape == (n, 3)

    def test_source_is_filename(self):
        if not TINY_MAT.is_file():
            pytest.skip('tiny mat fixture not found')
        d = Data(str(TINY_MAT))
        assert d.batches[0]['_source'] == 'mn11_157aprh_tiny.mat'


# MARK: simulate

class TestSimulateWithRepeater:
    '''Run the repeater binary against the real tiny mat file.'''

    def test_run_completes(self, repeater):
        if not TINY_MAT.is_file():
            pytest.skip('tiny mat fixture not found')
        d = Data(str(TINY_MAT))
        batch = d.batches[0]
        sim = object.__new__(Simulator)
        sim.binary = repeater

        with patch('simulator.runner.Benchmarker', return_value=mock_benchmarker()):
            result = sim.run(batch)

        assert isinstance(result, SimResult)
        assert result.sample_index == len(batch['p'])

    def test_output_shapes_match_input(self, repeater):
        if not TINY_MAT.is_file():
            pytest.skip('tiny mat fixture not found')
        d = Data(str(TINY_MAT))
        batch = d.batches[0]
        n = len(batch['p'])
        sim = object.__new__(Simulator)
        sim.binary = repeater

        with patch('simulator.runner.Benchmarker', return_value=mock_benchmarker()):
            result = sim.run(batch)

        assert result.Mw.shape == (n, 3)
        assert result.Aw.shape == (n, 3)

    def test_repeater_echoes_values(self, repeater):
        '''Repeater writes back the first 6 of 7 input floats.
        Input:  p mx my mz ax ay az
        Output: p mx my mz ax ay
        So Mw = (p, mx, my) and Aw = (mz, ax, ay).'''
        if not TINY_MAT.is_file():
            pytest.skip('tiny mat fixture not found')
        d = Data(str(TINY_MAT))
        batch = d.batches[0]
        sim = object.__new__(Simulator)
        sim.binary = repeater

        with patch('simulator.runner.Benchmarker', return_value=mock_benchmarker()):
            result = sim.run(batch)

        for i in range(len(batch['p'])):
            expected_mw = np.float32([batch['p'][i], batch['M'][i][0], batch['M'][i][1]])
            expected_aw = np.float32([batch['M'][i][2], batch['A'][i][0], batch['A'][i][1]])
            np.testing.assert_allclose(result.Mw[i], expected_mw, atol=1e-7)
            np.testing.assert_allclose(result.Aw[i], expected_aw, atol=1e-7)


# MARK: error calculation

class TestErrorWithRepeater:
    '''Test error calculations using repeater output against ground truth.'''

    def test_errors_are_nonzero(self, repeater):
        '''Since repeater echoes raw inputs, not real predictions,
        the error against ground truth Mw/Aw should be nonzero.'''
        if not TINY_MAT.is_file():
            pytest.skip('tiny mat fixture not found')
        d = Data(str(TINY_MAT))
        batch = d.batches[0]
        sim = object.__new__(Simulator)
        sim.binary = repeater

        with patch('simulator.runner.Benchmarker', return_value=mock_benchmarker()):
            result = sim.run(batch)

        mae_mw, mae_aw, rmse_mw, rmse_aw = calculate_errors(batch, result)
        assert np.all(mae_mw > 0)
        assert np.all(mae_aw > 0)
        assert np.all(rmse_mw >= mae_mw)
        assert np.all(rmse_aw >= mae_aw)

    def test_errors_are_per_axis(self, repeater):
        if not TINY_MAT.is_file():
            pytest.skip('tiny mat fixture not found')
        d = Data(str(TINY_MAT))
        batch = d.batches[0]
        sim = object.__new__(Simulator)
        sim.binary = repeater

        with patch('simulator.runner.Benchmarker', return_value=mock_benchmarker()):
            result = sim.run(batch)

        mae_mw, mae_aw, rmse_mw, rmse_aw = calculate_errors(batch, result)
        assert mae_mw.shape == (3,)
        assert mae_aw.shape == (3,)
        assert rmse_mw.shape == (3,)
        assert rmse_aw.shape == (3,)


# MARK: report

class TestReportGeneration:
    '''Test save_report produces a valid JSON file from real pipeline output.'''

    def test_produces_valid_json(self, repeater, config, tmp_path):
        if not TINY_MAT.is_file():
            pytest.skip('tiny mat fixture not found')
        d = Data(str(TINY_MAT))
        batch = d.batches[0]
        sim = object.__new__(Simulator)
        sim.binary = repeater

        with patch('simulator.runner.Benchmarker', return_value=mock_benchmarker()):
            result = sim.run(batch)

        save_report('integration_report', config, batch, result, tmp_path)

        with open(tmp_path / 'integration_report.json') as f:
            report = json.load(f)

        assert report['name'] == 'integration_report'
        assert report['data_file'] == 'mn11_157aprh_tiny.mat'

    def test_report_has_all_sections(self, repeater, config, tmp_path):
        if not TINY_MAT.is_file():
            pytest.skip('tiny mat fixture not found')
        d = Data(str(TINY_MAT))
        batch = d.batches[0]
        sim = object.__new__(Simulator)
        sim.binary = repeater

        with patch('simulator.runner.Benchmarker', return_value=mock_benchmarker()):
            result = sim.run(batch)

        save_report('integration_report', config, batch, result, tmp_path)

        with open(tmp_path / 'integration_report.json') as f:
            report = json.load(f)

        for section in ['name', 'data_file', 'config', 'benchmark', 'derived', 'error']:
            assert section in report

    def test_report_error_values_are_3_element_lists(self, repeater, config, tmp_path):
        if not TINY_MAT.is_file():
            pytest.skip('tiny mat fixture not found')
        d = Data(str(TINY_MAT))
        batch = d.batches[0]
        sim = object.__new__(Simulator)
        sim.binary = repeater

        with patch('simulator.runner.Benchmarker', return_value=mock_benchmarker()):
            result = sim.run(batch)

        save_report('integration_report', config, batch, result, tmp_path)

        with open(tmp_path / 'integration_report.json') as f:
            report = json.load(f)

        for key in ['MAE_Mw_uT', 'RMSE_Mw_uT', 'MAE_Aw_g', 'RMSE_Aw_g']:
            assert isinstance(report['error'][key], list)
            assert len(report['error'][key]) == 3

    def test_report_derived_values_positive(self, repeater, config, tmp_path):
        if not TINY_MAT.is_file():
            pytest.skip('tiny mat fixture not found')
        d = Data(str(TINY_MAT))
        batch = d.batches[0]
        sim = object.__new__(Simulator)
        sim.binary = repeater

        with patch('simulator.runner.Benchmarker', return_value=mock_benchmarker()):
            result = sim.run(batch)

        save_report('integration_report', config, batch, result, tmp_path)

        with open(tmp_path / 'integration_report.json') as f:
            report = json.load(f)

        derived = report['derived']
        assert derived['minimum_operating_frequency_MHz'] > 0
        assert derived['energy_per_inference_mJ'] > 0
        assert derived['duty_cycle'] > 0
        assert derived['power_consumption_mW'] > 0

    def test_report_config_matches(self, repeater, config, tmp_path):
        if not TINY_MAT.is_file():
            pytest.skip('tiny mat fixture not found')
        d = Data(str(TINY_MAT))
        batch = d.batches[0]
        sim = object.__new__(Simulator)
        sim.binary = repeater

        with patch('simulator.runner.Benchmarker', return_value=mock_benchmarker()):
            result = sim.run(batch)

        save_report('integration_report', config, batch, result, tmp_path)

        with open(tmp_path / 'integration_report.json') as f:
            report = json.load(f)

        assert report['config']['sample_rate'] == 5
        assert report['config']['voltage'] == 1.8
        assert report['config']['max_frequency'] == 160
