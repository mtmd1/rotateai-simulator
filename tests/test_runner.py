'''
tests/test_runner.py
Unit tests for simulator/runner.py

Created: 2026-02-24
 Author: Maxence Morel Dierckx
'''
import pytest
import numpy as np
from unittest.mock import patch
from pathlib import Path
from simulator.runner import SimResult, Simulator
from conftest import mock_benchmarker


# MARK: fixtures

def _make_data(n: int = 10) -> dict[str, np.ndarray]:
    '''Build a small valid data dict.'''
    return {
        'A': np.random.rand(n, 3),
        'M': np.random.rand(n, 3),
        'Aw': np.random.rand(n, 3),
        'Mw': np.random.rand(n, 3),
        'p': np.random.rand(n),
    }


def _make_simulator(binary_path: Path) -> Simulator:
    '''Create a Simulator without running __init__.'''
    obj = object.__new__(Simulator)
    obj.binary = binary_path
    obj.cmdline = None
    return obj


# MARK: SimResult

class TestSimResult:
    '''Tests for SimResult initialisation and add_row.'''

    def test_init_shapes(self):
        r = SimResult(50)
        assert r.Mw.shape == (50, 3)
        assert r.Aw.shape == (50, 3)
        assert r.sample_index == 0
        assert r.output_indices == []
        assert r.benchmark is None

    def test_add_row_splits_correctly(self):
        r = SimResult(1)
        r.add_row([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 0)
        np.testing.assert_array_equal(r.Mw[0], [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(r.Aw[0], [4.0, 5.0, 6.0])
        assert r.sample_index == 1
        assert r.output_indices == [0]

    def test_add_row_ignores_extra_values(self):
        r = SimResult(1)
        r.add_row([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], 0)
        np.testing.assert_array_equal(r.Mw[0], [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(r.Aw[0], [4.0, 5.0, 6.0])

    def test_add_multiple_rows(self):
        r = SimResult(3)
        for i in range(3):
            r.add_row([float(i)] * 6, i)
        assert r.sample_index == 3
        assert r.output_indices == [0, 1, 2]
        for i in range(3):
            np.testing.assert_array_equal(r.Mw[i], [float(i)] * 3)
            np.testing.assert_array_equal(r.Aw[i], [float(i)] * 3)

    def test_trim_slices_arrays(self):
        r = SimResult(10)
        r.add_row([1.0] * 6, 0)
        r.add_row([2.0] * 6, 3)
        r.trim()
        assert r.Mw.shape == (2, 3)
        assert r.Aw.shape == (2, 3)
        assert r.output_indices == [0, 3]

    def test_output_count_and_ratio(self):
        r = SimResult(10)
        r.add_row([1.0] * 6, 0)
        r.add_row([2.0] * 6, 5)
        assert r.output_count == 2
        assert r.output_ratio == 0.2


# MARK: Simulator init

class TestSimulatorInit:
    '''Tests for Simulator.__init__.'''

    def test_valid_binary(self, repeater):
        s = Simulator(str(repeater))
        assert s.binary == repeater
        assert s.cmdline is None

    def test_valid_binary_with_cmdline(self, repeater):
        s = Simulator(str(repeater), cmdline='--offset 50')
        assert s.cmdline == '--offset 50'

    def test_nonexistent_binary_exits(self, tmp_path):
        with pytest.raises(SystemExit):
            Simulator(str(tmp_path / 'no_such_binary'))


# MARK: Simulator.run

class TestSimulatorRun:
    '''Tests for the main simulation loop, with Benchmarker mocked out.'''

    def test_run_returns_sim_result(self, repeater):
        sim = _make_simulator(repeater)
        data = _make_data(5)
        with patch('simulator.runner.Benchmarker', return_value=mock_benchmarker()):
            result = sim.run(data)
        assert isinstance(result, SimResult)
        assert result.sample_index == 5

    def test_run_benchmarker_assigned(self, repeater):
        sim = _make_simulator(repeater)
        data = _make_data(3)
        mock_bench = mock_benchmarker()
        with patch('simulator.runner.Benchmarker', return_value=mock_bench):
            result = sim.run(data)
        assert result.benchmark is mock_bench

    def test_run_benchmarker_collect_called(self, repeater):
        sim = _make_simulator(repeater)
        data = _make_data(3)
        mock_bench = mock_benchmarker()
        with patch('simulator.runner.Benchmarker', return_value=mock_bench):
            sim.run(data)
        mock_bench.collect.assert_called_once()

    def test_repeater_echoes_input(self, repeater):
        '''Repeater reads 7 float32s and writes back the first 6.
        Input is: p mx my mz ax ay az
        Output is: p mx my mz ax ay
        So Mw gets (p, mx, my) and Aw gets (mz, ax, ay).
        Values go through float64->float32->float64 so we use atol for precision.'''
        sim = _make_simulator(repeater)
        n = 5
        data = _make_data(n)
        with patch('simulator.runner.Benchmarker', return_value=mock_benchmarker()):
            result = sim.run(data)

        for i in range(n):
            expected_mw = np.float32([data['p'][i], data['M'][i][0], data['M'][i][1]])
            expected_aw = np.float32([data['M'][i][2], data['A'][i][0], data['A'][i][1]])
            np.testing.assert_allclose(result.Mw[i], expected_mw, atol=1e-7)
            np.testing.assert_allclose(result.Aw[i], expected_aw, atol=1e-7)

    def test_run_single_step(self, repeater):
        sim = _make_simulator(repeater)
        data = _make_data(1)
        with patch('simulator.runner.Benchmarker', return_value=mock_benchmarker()):
            result = sim.run(data)
        assert result.sample_index == 1
        assert result.Mw.shape == (1, 3)
        assert result.Aw.shape == (1, 3)

    def test_run_output_count_equals_input(self, repeater):
        '''Repeater outputs every sample, so output_count == input count.'''
        sim = _make_simulator(repeater)
        n = 5
        data = _make_data(n)
        with patch('simulator.runner.Benchmarker', return_value=mock_benchmarker()):
            result = sim.run(data)
        assert result.output_count == n
        assert result.output_ratio == 1.0

    def test_run_output_indices_sequential(self, repeater):
        '''Repeater outputs every sample, so indices are sequential.'''
        sim = _make_simulator(repeater)
        n = 5
        data = _make_data(n)
        with patch('simulator.runner.Benchmarker', return_value=mock_benchmarker()):
            result = sim.run(data)
        assert result.output_indices == list(range(n))


# MARK: Simulator.run with skipper

class TestSimulatorRunSkipper:
    '''Tests for sparse output using the skipper binary.'''

    def test_skipper_outputs_odd_indices(self, skipper):
        sim = _make_simulator(skipper)
        n = 10
        data = _make_data(n)
        with patch('simulator.runner.Benchmarker', return_value=mock_benchmarker()):
            result = sim.run(data)
        assert result.output_count == 5
        assert result.output_ratio == 0.5
        assert result.output_indices == [1, 3, 5, 7, 9]

    def test_skipper_shapes_trimmed(self, skipper):
        sim = _make_simulator(skipper)
        n = 10
        data = _make_data(n)
        with patch('simulator.runner.Benchmarker', return_value=mock_benchmarker()):
            result = sim.run(data)
        assert result.Mw.shape == (5, 3)
        assert result.Aw.shape == (5, 3)

    def test_skipper_echoes_correct_values(self, skipper):
        '''Skipper outputs odd-indexed samples. Verify echoed values match.'''
        sim = _make_simulator(skipper)
        n = 10
        data = _make_data(n)
        with patch('simulator.runner.Benchmarker', return_value=mock_benchmarker()):
            result = sim.run(data)

        for out_idx, in_idx in enumerate(result.output_indices):
            expected_mw = np.float32([data['p'][in_idx], data['M'][in_idx][0], data['M'][in_idx][1]])
            expected_aw = np.float32([data['M'][in_idx][2], data['A'][in_idx][0], data['A'][in_idx][1]])
            np.testing.assert_allclose(result.Mw[out_idx], expected_mw, atol=1e-7)
            np.testing.assert_allclose(result.Aw[out_idx], expected_aw, atol=1e-7)
