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
    bench.total_instructions = overrides.get('total_instructions', 1_000_000)
    bench.total_flops = overrides.get('total_flops', 50_000)
    bench.cpu_time = overrides.get('cpu_time', 0.5)
    return bench


def _make_result(N: int, benchmark=None, prh=None, output_indices=None) -> SimResult:
    '''Create a SimResult with controlled data.'''
    result = SimResult(N)
    result.benchmark = benchmark or _make_benchmark()
    if prh is not None:
        result.prh = prh
    result.output_indices = output_indices if output_indices is not None else list(range(N))
    result.sample_index = len(result.output_indices)
    result.wall_time = 0.1
    result.arena_used_bytes = 65536
    return result


def _make_data(n: int, *, pitch=None, roll=None, head=None, A=None, Aw=None, source='test.mat'):
    '''Build a data dict with the new I/O fields. Defaults are zeros.'''
    return {
        'A': A if A is not None else np.zeros((n, 3)),
        'Aw': Aw if Aw is not None else np.zeros((n, 3)),
        'p': np.zeros(n),
        'pitch': pitch if pitch is not None else np.zeros(n),
        'roll': roll if roll is not None else np.zeros(n),
        'head': head if head is not None else np.zeros(n),
        '_source': source,
    }


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
        '''When predicted PRH matches ground truth and A=Aw=0, all errors are 0.'''
        n = 4
        prh = np.array([[0.1, 0.2, 0.3]] * n)
        data = _make_data(n, pitch=prh[:, 0], roll=prh[:, 1], head=prh[:, 2])
        result = _make_result(n, prh=prh.copy())
        errs = calculate_errors(data, result)
        np.testing.assert_array_almost_equal(errs['mae_prh'], [0.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(errs['rmse_prh'], [0.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(errs['mae_aw'], [0.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(errs['rmse_aw'], [0.0, 0.0, 0.0])

    def test_known_prh_mae(self):
        '''Small PRH offsets (no wrap): MAE per axis equals mean of |offset|.'''
        n = 2
        ground = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        predicted = np.array([[0.1, -0.2, 0.3], [-0.1, 0.2, -0.3]])
        data = _make_data(n, pitch=ground[:, 0], roll=ground[:, 1], head=ground[:, 2])
        result = _make_result(n, prh=predicted)
        errs = calculate_errors(data, result)
        np.testing.assert_array_almost_equal(errs['mae_prh'], [0.1, 0.2, 0.3])

    def test_prh_wrap_around(self):
        '''Difference across the +/- pi wrap should be small, not ~2 pi.'''
        n = 1
        ground = np.array([[np.pi - 0.1, 0.0, 0.0]])
        predicted = np.array([[-np.pi + 0.1, 0.0, 0.0]])
        data = _make_data(n, pitch=ground[:, 0], roll=ground[:, 1], head=ground[:, 2])
        result = _make_result(n, prh=predicted)
        errs = calculate_errors(data, result)
        # Wrapped diff is ~0.2 rad, not ~2*pi - 0.2
        assert errs['mae_prh'][0] == pytest.approx(0.2, abs=1e-6)

    def test_aw_error_from_rotation(self):
        '''When predicted PRH = 0 and ground Aw matches A, Aw error is zero
        (R(0)·A = A).'''
        n = 3
        A = np.random.rand(n, 3)
        prh = np.zeros((n, 3))
        data = _make_data(n, A=A, Aw=A.copy(), pitch=prh[:, 0], roll=prh[:, 1], head=prh[:, 2])
        result = _make_result(n, prh=prh)
        errs = calculate_errors(data, result)
        np.testing.assert_array_almost_equal(errs['mae_aw'], [0.0, 0.0, 0.0])

    def test_returns_per_axis_shapes(self):
        n = 50
        prh_truth = np.random.rand(n, 3) * 0.5
        prh_pred = np.random.rand(n, 3) * 0.5
        A = np.random.rand(n, 3)
        Aw = np.random.rand(n, 3)
        data = _make_data(n, A=A, Aw=Aw,
                          pitch=prh_truth[:, 0], roll=prh_truth[:, 1], head=prh_truth[:, 2])
        result = _make_result(n, prh=prh_pred)
        errs = calculate_errors(data, result)
        assert errs['mae_prh'].shape == (3,)
        assert errs['rmse_prh'].shape == (3,)
        assert errs['mae_aw'].shape == (3,)
        assert errs['rmse_aw'].shape == (3,)

    def test_rmse_geq_mae(self):
        '''RMSE is always >= MAE for any distribution.'''
        n = 100
        prh_truth = np.random.rand(n, 3) * 0.3
        prh_pred = prh_truth + np.random.randn(n, 3) * 0.05
        A = np.random.rand(n, 3)
        Aw = np.random.rand(n, 3)
        data = _make_data(n, A=A, Aw=Aw,
                          pitch=prh_truth[:, 0], roll=prh_truth[:, 1], head=prh_truth[:, 2])
        result = _make_result(n, prh=prh_pred)
        errs = calculate_errors(data, result)
        assert np.all(errs['rmse_prh'] >= errs['mae_prh'] - 1e-12)
        assert np.all(errs['rmse_aw'] >= errs['mae_aw'] - 1e-12)

    def test_interpolation_identity_when_all_output(self):
        '''When all samples have output, np.interp is a no-op on PRH error.'''
        n = 4
        prh_truth = np.array([[0.0, 0.0, 0.0]] * n)
        prh_pred = np.array([[0.1, 0.0, 0.0],
                             [0.2, 0.0, 0.0],
                             [0.3, 0.0, 0.0],
                             [0.4, 0.0, 0.0]])
        data = _make_data(n, pitch=prh_truth[:, 0], roll=prh_truth[:, 1], head=prh_truth[:, 2])
        result = _make_result(n, prh=prh_pred)
        errs = calculate_errors(data, result)
        expected = np.mean(np.abs(prh_truth - prh_pred), axis=0)
        np.testing.assert_array_almost_equal(errs['mae_prh'], expected)

    def test_interpolated_error_differs_from_sparse(self):
        '''With sparse output, interpolated predictions vary across skipped samples.'''
        # 4 ground truth steps; predictions only at indices 0 and 3
        ground_pitch = np.array([0.0, 0.5, 0.5, 0.0])
        # Predictions at indices 0 and 3 are both zero; interpolated stays 0.
        prh_pred = np.array([[0.0, 0.0, 0.0],
                             [0.0, 0.0, 0.0]])
        data = _make_data(4, pitch=ground_pitch,
                          roll=np.zeros(4), head=np.zeros(4))
        result = _make_result(4, prh=prh_pred, output_indices=[0, 3])
        errs = calculate_errors(data, result)
        # MAE on pitch: |0 - 0|, |0 - 0.5|, |0 - 0.5|, |0 - 0| -> 0.25
        assert errs['mae_prh'][0] == pytest.approx(0.25)


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
            data = _make_data(n, source='test.mat')
            result = result or _make_result(n)
        save_report(binary, config, data, result, tmp_path)
        report_file = next(tmp_path.glob('simreport_*.json'))
        with open(report_file) as f:
            return json.load(f), report_file

    def test_creates_json_file(self, tmp_path):
        _, report_file = self._save_and_load(tmp_path)
        assert report_file.is_file()

    def test_report_name_format(self, tmp_path):
        n = 10
        data = _make_data(n, source='mn11_157aprh.mat')
        result = _make_result(n)
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
        assert 'file_size_KiB' in bench
        assert 'memory_usage_KiB' in bench
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
        data = _make_data(10, source='test.mat')
        result = _make_result(10, prh=np.random.rand(3, 3), output_indices=[1, 5, 9])
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
        assert 'MAE_PRH_rad' in error
        assert 'RMSE_PRH_rad' in error
        assert 'MAE_Aw_g' in error
        assert 'RMSE_Aw_g' in error

    def test_timestamp_format(self, tmp_path):
        report, _ = self._save_and_load(tmp_path)
        assert re.match(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}', report['timestamp'])

    def test_error_values_are_3_element_lists(self, tmp_path):
        config = _make_config()
        n = 10
        data = _make_data(n, A=np.random.rand(n, 3), Aw=np.random.rand(n, 3),
                          pitch=np.random.rand(n)*0.5,
                          roll=np.random.rand(n)*0.5,
                          head=np.random.rand(n)*0.5,
                          source='test.mat')
        result = _make_result(n, prh=np.random.rand(n, 3) * 0.5)
        report, _ = self._save_and_load(tmp_path, config=config, data=data, result=result)
        for key in ['MAE_PRH_rad', 'RMSE_PRH_rad', 'MAE_Aw_g', 'RMSE_Aw_g']:
            assert isinstance(report['error'][key], list)
            assert len(report['error'][key]) == 3
