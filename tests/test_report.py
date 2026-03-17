'''
tests/test_report.py
Unit tests for simulator/report.py

Created: 2026-02-26
 Author: Maxence Morel Dierckx
'''
import json
import re
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
    cfg.max_frequency = overrides.get('max_frequency', 160)
    cfg.sleep_current_uA = overrides.get('sleep_current_uA', 10)
    cfg.to_dict.return_value = {
        'sample_rate': cfg.sample_rate,
        'voltage': cfg.voltage,
        'DMIPS_per_MHz': cfg.DMIPS_per_MHz,
        'uA_per_MHz': cfg.uA_per_MHz,
        'max_frequency': cfg.max_frequency,
        'sleep_current_uA': cfg.sleep_current_uA,
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


def _make_result(N: int, benchmark=None, Mw=None, Aw=None, output_indices=None) -> SimResult:
    '''Create a SimResult with controlled data.'''
    result = SimResult(N)
    result.benchmark = benchmark or _make_benchmark()
    if Mw is not None:
        result.Mw = Mw
    if Aw is not None:
        result.Aw = Aw
    result.output_indices = output_indices if output_indices is not None else list(range(N))
    result.sample_index = len(result.output_indices)
    result.wall_time = 0.1
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

    def test_interpolation_identity_when_all_output(self):
        '''When all samples have output, interpolation is a no-op.'''
        ground = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        predicted = np.array([[1.5, 2.0, 3.0], [4.0, 5.5, 6.0]])
        data = {'Mw': ground, 'Aw': ground}
        result = _make_result(2, Mw=predicted, Aw=predicted)
        mae_mw, _, _, _ = calculate_errors(data, result)
        expected = np.mean(np.abs(ground - predicted), axis=0)
        np.testing.assert_array_almost_equal(mae_mw, expected)

    def test_interpolated_error_differs_from_sparse(self):
        '''With sparse output, interpolated error includes skipped samples.'''
        # 4 ground truth samples, output only at indices 0 and 3
        ground_Mw = np.array([[0.0, 0.0, 0.0],
                              [10.0, 10.0, 10.0],
                              [10.0, 10.0, 10.0],
                              [0.0, 0.0, 0.0]])
        # Predictions match ground truth at output indices
        predicted_Mw = np.array([[0.0, 0.0, 0.0],
                                 [0.0, 0.0, 0.0]])
        data = {'Mw': ground_Mw, 'Aw': ground_Mw}
        result = _make_result(4, Mw=predicted_Mw, Aw=predicted_Mw, output_indices=[0, 3])
        mae_mw, _, _, _ = calculate_errors(data, result)
        # Interpolated predictions: [0, 0, 0, 0] — linear interp between 0 and 0
        # Ground truth at indices 1,2 is 10 — so error is nonzero
        assert np.all(mae_mw > 0)


# MARK: derive_metrics

class TestDeriveMetrics:
    '''Tests for the derived metric calculations.'''

    def test_known_values(self):
        '''Hand-calculated with:
        config: sample_rate=5, voltage=1.8, DMIPS_per_MHz=1.5, uA_per_MHz=51.6,
                max_frequency=160, sleep_current_uA=10
        benchmark: total_instructions=1_000_000, N=100, output_ratio=1.0

        instructions_per_inference = 1_000_000 / 100 = 10_000

        minimum_frequency = (10_000 * 5) / (1.5 * 1e6) = 1/30 MHz

        charge_per_inference = 51.6 * 10_000 / (1.5 * 1e6) = 0.344 uC
        energy_per_inference = 1.8 * 0.344 / 1e3 = 0.0006192 mJ

        duty_cycle = (1/30) / 160 = 1/4800

        power = E * f_s + V * I_sleep * 1e-3 * (1 - duty)
              = 0.0006192 * 5 + 1.8 * 10 * 1e-3 * (1 - 1/4800)
              = 0.003096 + 0.018 * (4799/4800)
        '''
        config = _make_config()
        bench = _make_benchmark(total_instructions=1_000_000, cpu_time=0.5)
        result = _make_result(100, benchmark=bench)

        min_freq, energy, duty, power = derive_metrics(config, result)
        assert min_freq == pytest.approx(1 / 30)
        assert energy == pytest.approx(0.0006192)
        assert duty == pytest.approx(1 / 30 / 160)
        expected_power = 0.0006192 * 5 + 1.8 * 10 * 1e-3 * (1 - 1/4800)
        assert power == pytest.approx(expected_power)

    def test_higher_sample_rate_increases_power(self):
        config_slow = _make_config(sample_rate=1)
        config_fast = _make_config(sample_rate=10)
        bench = _make_benchmark()
        result = _make_result(100, benchmark=bench)

        _, _, _, power_slow = derive_metrics(config_slow, result)
        _, _, _, power_fast = derive_metrics(config_fast, result)
        assert power_fast > power_slow

    def test_zero_instructions_gives_zero_duty_cycle(self):
        config = _make_config()
        bench = _make_benchmark(total_instructions=0)
        result = _make_result(100, benchmark=bench)

        _, _, duty, _ = derive_metrics(config, result)
        assert duty == 0.0


# MARK: save_report

class TestSaveReport:
    '''Tests for the full report generation and JSON output.'''

    def _save_and_load(self, tmp_path, config=None, data=None, result=None, binary='testbin'):
        '''Call save_report and return the parsed JSON.'''
        config = config or _make_config()
        if data is None:
            n = 10
            ground = np.random.rand(n, 3)
            data = {'Mw': ground, 'Aw': ground, '_source': 'test.mat'}
            result = result or _make_result(n, Mw=ground.copy(), Aw=ground.copy())
        save_report(binary, config, data, result, tmp_path)
        report_file = next(tmp_path.glob('simreport_*.json'))
        with open(report_file) as f:
            return json.load(f), report_file

    def test_creates_json_file(self, tmp_path):
        _, report_file = self._save_and_load(tmp_path)
        assert report_file.is_file()

    def test_report_name_format(self, tmp_path):
        n = 10
        ground = np.random.rand(n, 3)
        data = {'Mw': ground, 'Aw': ground, '_source': 'mn11_157aprh.mat'}
        result = _make_result(n, Mw=ground.copy(), Aw=ground.copy())
        _, report_file = self._save_and_load(tmp_path, data=data, result=result, binary='variable')
        assert re.match(r'simreport_variable_mn11-157aprh_[0-9a-f]{4}\.json', report_file.name)

    def test_json_structure(self, tmp_path):
        report, _ = self._save_and_load(tmp_path)
        assert 'name' in report
        assert 'binary' in report
        assert 'timestamp' in report
        assert 'data_file' in report
        assert 'config' in report
        assert 'benchmark' in report
        assert 'derived' in report
        assert 'error' in report

    def test_benchmark_keys(self, tmp_path):
        report, _ = self._save_and_load(tmp_path)
        bench = report['benchmark']
        assert 'file_size_KB' in bench
        assert 'peak_memory_KB' in bench
        assert 'instructions_per_inference' in bench
        assert 'FLOPS_per_inference' in bench
        assert 'cpu_time_s' in bench
        assert 'wall_time_s' in bench
        assert 'output_count' in bench
        assert 'output_ratio' in bench

    def test_benchmark_output_count(self, tmp_path):
        report, _ = self._save_and_load(tmp_path)
        assert report['benchmark']['output_count'] == 10
        assert report['benchmark']['output_ratio'] == 1.0

    def test_benchmark_output_count_sparse(self, tmp_path):
        config = _make_config()
        ground = np.random.rand(10, 3)
        predicted = np.random.rand(3, 3)
        data = {'Mw': ground, 'Aw': ground, '_source': 'test.mat'}
        result = _make_result(10, Mw=predicted, Aw=predicted, output_indices=[1, 5, 9])
        report, _ = self._save_and_load(tmp_path, config=config, data=data, result=result)
        assert report['benchmark']['output_count'] == 3
        assert report['benchmark']['output_ratio'] == pytest.approx(0.3)

    def test_derived_keys(self, tmp_path):
        report, _ = self._save_and_load(tmp_path)
        derived = report['derived']
        assert 'minimum_operating_frequency_MHz' in derived
        assert 'energy_per_inference_mJ' in derived
        assert 'duty_cycle' in derived
        assert 'power_consumption_mW' in derived

    def test_error_keys(self, tmp_path):
        report, _ = self._save_and_load(tmp_path)
        error = report['error']
        assert 'MAE_Mw_uT' in error
        assert 'RMSE_Mw_uT' in error
        assert 'MAE_Aw_g' in error
        assert 'RMSE_Aw_g' in error

    def test_timestamp_format(self, tmp_path):
        report, _ = self._save_and_load(tmp_path)
        assert re.match(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}', report['timestamp'])

    def test_error_values_are_3_element_lists(self, tmp_path):
        config = _make_config()
        n = 10
        ground = np.random.rand(n, 3)
        data = {'Mw': ground, 'Aw': ground, '_source': 'test.mat'}
        result = _make_result(n, Mw=np.random.rand(n, 3), Aw=np.random.rand(n, 3))
        report, _ = self._save_and_load(tmp_path, config=config, data=data, result=result)
        for key in ['MAE_Mw_uT', 'RMSE_Mw_uT', 'MAE_Aw_g', 'RMSE_Aw_g']:
            assert isinstance(report['error'][key], list)
            assert len(report['error'][key]) == 3
