'''
tests/test_e2e.py
End-to-end tests for the full simulation pipeline.
No mocking except Benchmarker (platform-dependent perf).

Created: 2026-02-26
 Author: Maxence Morel Dierckx
'''
import json
import pytest
from argparse import Namespace
from unittest.mock import patch

from simulator.__main__ import simulate
from conftest import FIXTURES, TINY_MAT, mock_benchmarker


TESTCONFIG = FIXTURES / 'testconfig.toml'


def _skip_if_missing():
    if not TINY_MAT.is_file():
        pytest.skip('tiny mat fixture not found')
    if not TESTCONFIG.is_file():
        pytest.skip('testconfig.toml fixture not found')


@pytest.fixture()
def report(repeater, tmp_path):
    '''Run the full pipeline once and return the parsed JSON report.'''
    _skip_if_missing()

    args = Namespace(
        config=str(TESTCONFIG),
        binary=str(repeater),
        data=str(TINY_MAT),
        output=str(tmp_path),
    )
    with patch('simulator.runner.Benchmarker', return_value=mock_benchmarker()):
        with patch('simulator.__main__.tqdm', lambda it, **kw: it):
            simulate(args)

    reports = list(tmp_path.glob('simreport_repeater_*.json'))
    assert len(reports) == 1, f'Expected 1 report, found {len(reports)}'

    with open(reports[0]) as f:
        return json.load(f)


# MARK: pipeline output

class TestFullPipeline:
    '''Run simulate() end-to-end with real Config, Data, Simulator, and report.'''

    def test_report_file_created(self, report):
        assert report is not None

    def test_report_has_all_sections(self, report):
        for section in ['name', 'data_file', 'config', 'benchmark', 'derived', 'error']:
            assert section in report

    def test_report_name_contains_binary(self, report):
        assert 'repeater' in report['name']

    def test_data_file_matches_fixture(self, report):
        assert report['data_file'] == 'mn11_157aprh_tiny.mat'


# MARK: config

class TestE2EConfig:
    '''Verify the report reflects testconfig.toml values.'''

    def test_sample_rate(self, report):
        assert report['config']['sample_rate'] == 5

    def test_voltage(self, report):
        assert report['config']['voltage'] == 1

    def test_max_frequency(self, report):
        assert report['config']['max_frequency'] == 160

    def test_DMIPS_per_MHz(self, report):
        assert report['config']['DMIPS_per_MHz'] == 2

    def test_uA_per_MHz(self, report):
        assert report['config']['uA_per_MHz'] == 50


# MARK: benchmark

class TestE2EBenchmark:
    '''Verify benchmark section is populated.'''

    def test_has_file_size(self, report):
        assert report['benchmark']['file_size_KB'] > 0

    def test_has_peak_memory(self, report):
        assert report['benchmark']['peak_memory_KB'] is not None

    def test_has_instructions(self, report):
        assert report['benchmark']['instructions_per_inference'] > 0

    def test_has_flops(self, report):
        assert isinstance(report['benchmark']['FLOPS_per_inference'], int)


# MARK: derived

class TestE2EDerived:
    '''Verify derived metrics are positive and consistent.'''

    def test_minimum_frequency_positive(self, report):
        assert report['derived']['minimum_operating_frequency_MHz'] > 0

    def test_energy_per_inference_positive(self, report):
        assert report['derived']['energy_per_inference_mJ'] > 0

    def test_duty_cycle_in_range(self, report):
        dc = report['derived']['duty_cycle']
        assert 0 < dc <= 1

    def test_power_consumption_positive(self, report):
        assert report['derived']['power_consumption_mW'] > 0


# MARK: error

class TestE2EError:
    '''Verify error metrics structure and values.'''

    def test_error_keys_present(self, report):
        for key in ['MAE_Mw_uT', 'RMSE_Mw_uT', 'MAE_Aw_g', 'RMSE_Aw_g']:
            assert key in report['error']

    def test_error_values_are_3_element_lists(self, report):
        for key in ['MAE_Mw_uT', 'RMSE_Mw_uT', 'MAE_Aw_g', 'RMSE_Aw_g']:
            assert isinstance(report['error'][key], list)
            assert len(report['error'][key]) == 3

    def test_errors_are_nonzero(self, report):
        '''Repeater echoes raw inputs, not real predictions, so error > 0.'''
        for key in ['MAE_Mw_uT', 'MAE_Aw_g']:
            assert all(v > 0 for v in report['error'][key])

    def test_rmse_geq_mae(self, report):
        for metric in ['Mw_uT', 'Aw_g']:
            mae = report['error'][f'MAE_{metric}']
            rmse = report['error'][f'RMSE_{metric}']
            for m, r in zip(mae, rmse):
                assert r >= m


# MARK: multi-batch

class TestMultiBatch:
    '''Verify the pipeline handles directory input (multiple batches).'''

    def test_directory_produces_one_report_per_batch(self, repeater, tmp_path):
        _skip_if_missing()

        args = Namespace(
            config=str(TESTCONFIG),
            binary=str(repeater),
            data=str(TINY_MAT.parent),
            output=str(tmp_path),
        )
        with patch('simulator.runner.Benchmarker', return_value=mock_benchmarker()):
            with patch('simulator.__main__.tqdm', lambda it, **kw: it):
                simulate(args)

        reports = sorted(tmp_path.glob('simreport_repeater_*.json'))
        assert len(reports) >= 1

        for report_path in reports:
            with open(report_path) as f:
                report = json.load(f)
            assert 'error' in report
            assert 'derived' in report
