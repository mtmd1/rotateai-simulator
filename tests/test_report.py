'''
tests/test_report.py
Unit tests for simulator/report.py

Created: 2026-02-26
 Author: Maxence Morel Dierckx
'''
import json
import pytest
import numpy as np
from unittest.mock import MagicMock

from simulator.report import NumpyEncoder, derive_metrics, calculate_errors, save_report
from simulator.runner import SimResult


# MARK: fixtures

def _make_config(**overrides):
    '''Create a mock Config with default values.'''
    cfg = MagicMock()
    cfg.sample_rate = overrides.get('sample_rate', 5)
    cfg.voltage = overrides.get('voltage', 1.8)
    cfg.DMIPS_per_MHz = overrides.get('DMIPS_per_MHz', 1.5)
    cfg.uA_per_MHz = overrides.get('uA_per_MHz', 51.6)
    cfg.safety_margin = overrides.get('safety_margin', 1.5)
    cfg.to_dict.return_value = {
        'sample_rate': cfg.sample_rate,
        'voltage': cfg.voltage,
        'DMIPS_per_MHz': cfg.DMIPS_per_MHz,
        'uA_per_MHz': cfg.uA_per_MHz,
        'safety_margin': cfg.safety_margin,
    }
    return cfg


def _make_benchmark(**overrides):
    '''Create a mock Benchmarker with default values.'''
    bench = MagicMock()
    bench.file_size = overrides.get('file_size', 1024)
    bench.peak_memory = overrides.get('peak_memory', 4096)
    bench.total_instructions = overrides.get('total_instructions', 1_000_000)
    bench.total_flops = overrides.get('total_flops', 50_000)
    bench.cpu_time = overrides.get('cpu_time', 0.5)
    return bench


def _make_result(N: int, benchmark=None, Mw=None, Aw=None) -> SimResult:
    '''Create a SimResult with controlled data.'''
    result = SimResult(N)
    result.benchmark = benchmark or _make_benchmark()
    if Mw is not None:
        result.Mw = Mw
    if Aw is not None:
        result.Aw = Aw
    return result


# MARK: NumpyEncoder

class TestNumpyEncoder:
    '''Tests for the custom JSON encoder.'''

    def test_encodes_ndarray(self):
        arr = np.array([1.0, 2.0, 3.0])
        result = json.loads(json.dumps(arr, cls=NumpyEncoder))
        assert result == [1.0, 2.0, 3.0]

    def test_encodes_2d_ndarray(self):
        arr = np.array([[1, 2], [3, 4]])
        result = json.loads(json.dumps(arr, cls=NumpyEncoder))
        assert result == [[1, 2], [3, 4]]

    def test_encodes_np_integer(self):
        val = np.int64(42)
        result = json.loads(json.dumps(val, cls=NumpyEncoder))
        assert result == 42

    def test_encodes_np_floating(self):
        val = np.float64(3.14)
        result = json.loads(json.dumps(val, cls=NumpyEncoder))
        assert result == pytest.approx(3.14)

    def test_falls_through_for_regular_types(self):
        data = {'a': 1, 'b': 'hello', 'c': [1, 2]}
        result = json.loads(json.dumps(data, cls=NumpyEncoder))
        assert result == data


# MARK: calculate_errors

class TestCalculateErrors:
    '''Tests for MAE and RMSE error calculations.'''

    def test_zero_error_when_identical(self):
        ground = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        data = {'Mw': ground, 'Aw': ground}
        result = _make_result(2, Mw=ground.copy(), Aw=ground.copy())
        mae_mw, mae_aw, rmse_mw, rmse_aw = calculate_errors(data, result)
        np.testing.assert_array_equal(mae_mw, [0.0, 0.0, 0.0])
        np.testing.assert_array_equal(mae_aw, [0.0, 0.0, 0.0])
        np.testing.assert_array_equal(rmse_mw, [0.0, 0.0, 0.0])
        np.testing.assert_array_equal(rmse_aw, [0.0, 0.0, 0.0])

    def test_known_mae(self):
        ground = np.array([[10.0, 20.0, 30.0], [10.0, 20.0, 30.0]])
        predicted = np.array([[12.0, 18.0, 30.0], [8.0, 22.0, 30.0]])
        data = {'Mw': ground, 'Aw': np.zeros((2, 3))}
        result = _make_result(2, Mw=predicted, Aw=np.zeros((2, 3)))
        mae_mw, _, _, _ = calculate_errors(data, result)
        # MAE per axis: mean([|2|, |2|]) = 2, mean([|2|, |2|]) = 2, mean([0, 0]) = 0
        np.testing.assert_array_almost_equal(mae_mw, [2.0, 2.0, 0.0])

    def test_known_rmse(self):
        ground = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        predicted = np.array([[3.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
        data = {'Mw': np.zeros((2, 3)), 'Aw': ground}
        result = _make_result(2, Mw=np.zeros((2, 3)), Aw=predicted)
        _, _, _, rmse_aw = calculate_errors(data, result)
        # RMSE axis 0: sqrt(mean([9, 16])) = sqrt(12.5) = 3.5355...
        np.testing.assert_array_almost_equal(rmse_aw, [np.sqrt(12.5), 0.0, 0.0])

    def test_returns_per_axis_values(self):
        n = 100
        ground = np.random.rand(n, 3)
        predicted = np.random.rand(n, 3)
        data = {'Mw': ground, 'Aw': ground}
        result = _make_result(n, Mw=predicted, Aw=predicted)
        mae_mw, mae_aw, rmse_mw, rmse_aw = calculate_errors(data, result)
        assert mae_mw.shape == (3,)
        assert mae_aw.shape == (3,)
        assert rmse_mw.shape == (3,)
        assert rmse_aw.shape == (3,)

    def test_rmse_geq_mae(self):
        '''RMSE is always >= MAE for any distribution.'''
        n = 100
        ground = np.random.rand(n, 3)
        predicted = ground + np.random.randn(n, 3) * 0.1
        data = {'Mw': ground, 'Aw': ground}
        result = _make_result(n, Mw=predicted, Aw=predicted)
        mae_mw, _, rmse_mw, _ = calculate_errors(data, result)
        assert np.all(rmse_mw >= mae_mw)


# MARK: derive_metrics

class TestDeriveMetrics:
    '''Tests for the derived metric calculations.'''

    def test_known_values(self):
        '''Hand-calculated with:
        config: sample_rate=5, voltage=1.8, DMIPS_per_MHz=1.5, uA_per_MHz=51.6, safety_margin=1.5
        benchmark: total_instructions=1_000_000, cpu_time=0.5, N=100

        instructions_per_inference = 1_000_000 / 100 = 10_000

        minimum_frequency = (10_000 * 5 * 1.5) / (1.5 * 1e6) = 75_000 / 1_500_000 = 0.05 MHz

        charge_per_inference = 51.6 * 10_000 / (1.5 * 1e6) = 516_000 / 1_500_000 = 0.344 uC
        energy_per_inference = 1.8 * 0.344 = 0.6192 uJ

        time_per_inference = 0.5 / 100 = 0.005 s
        sample_period = 1 / 5 = 0.2 s
        duty_cycle = 0.005 / 0.2 = 0.025

        power_consumption = 0.6192 * 5 = 3.096 uW
        '''
        config = _make_config()
        bench = _make_benchmark(total_instructions=1_000_000, cpu_time=0.5)
        result = _make_result(100, benchmark=bench)

        min_freq, energy, duty, power = derive_metrics(config, result)
        assert min_freq == pytest.approx(0.05)
        assert energy == pytest.approx(0.6192)
        assert duty == pytest.approx(0.025)
        assert power == pytest.approx(3.096)

    def test_higher_safety_margin_increases_frequency(self):
        config_low = _make_config(safety_margin=1.0)
        config_high = _make_config(safety_margin=2.0)
        bench = _make_benchmark()
        result = _make_result(100, benchmark=bench)

        freq_low, _, _, _ = derive_metrics(config_low, result)
        freq_high, _, _, _ = derive_metrics(config_high, result)
        assert freq_high > freq_low

    def test_higher_sample_rate_increases_power(self):
        config_slow = _make_config(sample_rate=1)
        config_fast = _make_config(sample_rate=10)
        bench = _make_benchmark()
        result = _make_result(100, benchmark=bench)

        _, _, _, power_slow = derive_metrics(config_slow, result)
        _, _, _, power_fast = derive_metrics(config_fast, result)
        assert power_fast > power_slow

    def test_zero_cpu_time_gives_zero_duty_cycle(self):
        config = _make_config()
        bench = _make_benchmark(cpu_time=0.0)
        result = _make_result(100, benchmark=bench)

        _, _, duty, _ = derive_metrics(config, result)
        assert duty == 0.0


# MARK: save_report

class TestSaveReport:
    '''Tests for the full report generation and JSON output.'''

    def test_creates_json_file(self, tmp_path):
        config = _make_config()
        n = 10
        ground = np.random.rand(n, 3)
        data = {'Mw': ground, 'Aw': ground, 'p': np.random.rand(n),
                'M': ground, 'A': ground}
        result = _make_result(n, Mw=ground.copy(), Aw=ground.copy())

        report_path = str(tmp_path / 'test_report')
        save_report(report_path, config, data, result)
        assert (tmp_path / 'test_report.json').is_file()

    def test_json_structure(self, tmp_path):
        config = _make_config()
        n = 10
        ground = np.random.rand(n, 3)
        data = {'Mw': ground, 'Aw': ground}
        result = _make_result(n, Mw=ground.copy(), Aw=ground.copy())

        report_path = str(tmp_path / 'test_report')
        save_report(report_path, config, data, result)
        with open(tmp_path / 'test_report.json') as f:
            report = json.load(f)

        assert 'name' in report
        assert 'config' in report
        assert 'benchmark' in report
        assert 'derived' in report
        assert 'error' in report

    def test_benchmark_keys(self, tmp_path):
        config = _make_config()
        n = 10
        ground = np.random.rand(n, 3)
        data = {'Mw': ground, 'Aw': ground}
        result = _make_result(n, Mw=ground, Aw=ground)

        report_path = str(tmp_path / 'test_report')
        save_report(report_path, config, data, result)
        with open(tmp_path / 'test_report.json') as f:
            report = json.load(f)

        bench = report['benchmark']
        assert 'file_size_bytes' in bench
        assert 'peak_memory_KB' in bench
        assert 'instructions_per_inference' in bench
        assert 'FLOPS_per_inference' in bench

    def test_derived_keys(self, tmp_path):
        config = _make_config()
        n = 10
        ground = np.random.rand(n, 3)
        data = {'Mw': ground, 'Aw': ground}
        result = _make_result(n, Mw=ground, Aw=ground)

        report_path = str(tmp_path / 'test_report')
        save_report(report_path, config, data, result)
        with open(tmp_path / 'test_report.json') as f:
            report = json.load(f)

        derived = report['derived']
        assert 'minimum_operating_frequency_MHz' in derived
        assert 'energy_per_inference_uJ' in derived
        assert 'duty_cycle_ratio' in derived
        assert 'power_consumption_uW' in derived

    def test_error_keys(self, tmp_path):
        config = _make_config()
        n = 10
        ground = np.random.rand(n, 3)
        data = {'Mw': ground, 'Aw': ground}
        result = _make_result(n, Mw=ground, Aw=ground)

        report_path = str(tmp_path / 'test_report')
        save_report(report_path, config, data, result)
        with open(tmp_path / 'test_report.json') as f:
            report = json.load(f)

        error = report['error']
        assert 'MAE_Mw' in error
        assert 'RMSE_Mw' in error
        assert 'MAE_Aw' in error
        assert 'RMSE_Aw' in error

    def test_error_values_are_3_element_lists(self, tmp_path):
        config = _make_config()
        n = 10
        ground = np.random.rand(n, 3)
        data = {'Mw': ground, 'Aw': ground}
        result = _make_result(n, Mw=np.random.rand(n, 3), Aw=np.random.rand(n, 3))

        report_path = str(tmp_path / 'test_report')
        save_report(report_path, config, data, result)
        with open(tmp_path / 'test_report.json') as f:
            report = json.load(f)

        for key in ['MAE_Mw', 'RMSE_Mw', 'MAE_Aw', 'RMSE_Aw']:
            assert isinstance(report['error'][key], list)
            assert len(report['error'][key]) == 3
