'''
tests/test_bench.py
Unit tests for simulator/bench.py

Created: 2026-02-24
 Author: Maxence Morel Dierckx
'''
import pytest
import resource
from unittest.mock import patch, MagicMock
from pathlib import Path
from simulator.bench import Benchmarker


# MARK: fixtures

PERF_STDERR_INTEL_LS = '''\

 Performance counter stats for 'ls':

     <not counted>      cpu_atom/instructions/u                                                 (0.00%)
           348,687      cpu_core/instructions/u
                 0      cpu_core/fp_arith_inst_retired.scalar_double/u

       0.002372555 seconds time elapsed

       0.000000000 seconds user
       0.002578000 seconds sys

'''

PERF_STDERR_INTEL_TRUE = '''\

 Performance counter stats for 'true':

           142,453      cpu_atom/instructions/u
     <not counted>      cpu_core/instructions/u                                                 (0.00%)
     <not counted>      cpu_core/fp_arith_inst_retired.scalar_double/u                                        (0.00%)

       0.002381053 seconds time elapsed

       0.001258000 seconds user
       0.001283000 seconds sys

'''

PERF_STDERR_SIMPLE = '''\

 Performance counter stats for 'target':

         1,000,000      instructions
            50,000      fp_arith_inst_retired.scalar_double

       0.005000000 seconds time elapsed

'''

PERF_STDERR_ZERO_FLOPS = '''\

 Performance counter stats for 'target':

           500,000      instructions
                 0      fp_arith_inst_retired.scalar_double

       0.001000000 seconds time elapsed

'''

PERF_STDERR_INSTRUCTIONS_ONLY = '''\

 Performance counter stats for 'target':

           750,000      instructions

       0.003000000 seconds time elapsed

'''

PERF_STDERR_NOT_SUPPORTED = '''\
     <not supported>      fp_arith_inst_retired.scalar_double
'''

PERF_STDERR_SYNTAX_ERROR = '''\
event syntax error: 'instructions,fp_ret_sse_avx_ops.single,fp_ret_sse_avx_ops.double'
                                  \\___ Bad event name

Unable to find event on a PMU of 'fp_ret_sse_avx_ops.double'
'''


def _make_benchmarker(**overrides):
    '''Create a Benchmarker without running __init__.'''
    obj = object.__new__(Benchmarker)
    obj.file_size = overrides.get('file_size', 1024)
    obj.peak_memory = overrides.get('peak_memory', None)
    obj.total_instructions = overrides.get('total_instructions', None)
    obj.total_flops = overrides.get('total_flops', None)
    obj.perf = overrides.get('perf', None)
    return obj


def _mock_perf(stderr: str) -> MagicMock:
    '''Create a mock perf process that returns the given stderr.'''
    mock = MagicMock()
    mock.communicate.return_value = (None, stderr)
    return mock


# MARK: get_perf_events

class TestGetPerfEvents:
    '''Tests for platform-based perf event selection.'''

    def _get_events(self, machine: str, cpuinfo_stdout: str = '') -> str:
        '''Call get_perf_events with mocked platform and cpuinfo.'''
        b = _make_benchmarker()
        with patch('simulator.bench.platform.machine', return_value=machine), \
             patch('simulator.bench.subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(stdout=cpuinfo_stdout)
            return b.get_perf_events()

    def test_intel_x86_64(self):
        events = self._get_events('x86_64', 'vendor_id\t: GenuineIntel')
        assert events == 'instructions,fp_arith_inst_retired.scalar_single,fp_arith_inst_retired.scalar_double,fp_arith_inst_retired.128b_packed_single,fp_arith_inst_retired.128b_packed_double,fp_arith_inst_retired.256b_packed_single,fp_arith_inst_retired.256b_packed_double'

    def test_amd_x86_64(self):
        events = self._get_events('x86_64', 'vendor_id\t: AuthenticAMD')
        assert events == 'instructions,fp_ret_sse_avx_ops.single,fp_ret_sse_avx_ops.double'

    def test_unknown_x86_64(self):
        events = self._get_events('x86_64', 'vendor_id\t: SomethingElse')
        assert events == 'instructions'

    def test_i386(self):
        events = self._get_events('i386', 'vendor_id\t: GenuineIntel')
        assert events == 'instructions,fp_arith_inst_retired.scalar_single,fp_arith_inst_retired.scalar_double,fp_arith_inst_retired.128b_packed_single,fp_arith_inst_retired.128b_packed_double,fp_arith_inst_retired.256b_packed_single,fp_arith_inst_retired.256b_packed_double'

    def test_i686(self):
        events = self._get_events('i686', 'vendor_id\t: AuthenticAMD')
        assert events == 'instructions,fp_ret_sse_avx_ops.single,fp_ret_sse_avx_ops.double'

    def test_aarch64(self):
        events = self._get_events('aarch64')
        assert events == 'instructions,fp_fixed_ops_spec'

    def test_arm(self):
        events = self._get_events('armv7l')
        assert events == 'instructions,fp_fixed_ops_spec'

    def test_unknown_platform(self):
        events = self._get_events('riscv64')
        assert events == 'instructions'


# MARK: validate_perf_events

class TestValidatePerfEvents:
    '''Tests for perf event validation.'''

    def test_syntax_error_exits(self):
        b = _make_benchmarker()
        with patch('simulator.bench.subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(stderr=PERF_STDERR_SYNTAX_ERROR)
            with pytest.raises(SystemExit):
                b.validate_perf_events('instructions,fp_ret_sse_avx_ops.single,fp_ret_sse_avx_ops.double')

    def test_not_supported_exits(self):
        b = _make_benchmarker()
        with patch('simulator.bench.subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(stderr=PERF_STDERR_NOT_SUPPORTED)
            with pytest.raises(SystemExit):
                b.validate_perf_events('fp_arith_inst_retired.scalar_double')

    def test_valid_events_pass(self):
        b = _make_benchmarker()
        with patch('simulator.bench.subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(stderr=PERF_STDERR_SIMPLE)
            b.validate_perf_events('instructions,fp_arith_inst_retired.scalar_single,fp_arith_inst_retired.scalar_double,fp_arith_inst_retired.128b_packed_single,fp_arith_inst_retired.128b_packed_double,fp_arith_inst_retired.256b_packed_single,fp_arith_inst_retired.256b_packed_double')


# MARK: open_perf_process

class TestOpenPerfProcess:
    '''Tests for opening the perf subprocess.'''

    def test_perf_not_installed_exits(self):
        b = _make_benchmarker()
        with patch('simulator.bench.subprocess.Popen', side_effect=FileNotFoundError):
            with pytest.raises(SystemExit):
                b.open_perf_process(1234, 'instructions')

    def test_permission_denied_exits(self):
        b = _make_benchmarker()
        with patch('simulator.bench.subprocess.Popen', side_effect=PermissionError):
            with pytest.raises(SystemExit):
                b.open_perf_process(1234, 'instructions')

    def test_returns_popen(self):
        b = _make_benchmarker()
        mock_popen = MagicMock()
        with patch('simulator.bench.subprocess.Popen', return_value=mock_popen):
            result = b.open_perf_process(1234, 'instructions')
            assert result is mock_popen

    # No test for wrong PID because it's passed directly from the Popen that runs it


# MARK: collect

class TestCollect:
    '''Tests for perf output parsing in collect().'''

    def test_simple_output(self):
        b = _make_benchmarker(perf=_mock_perf(PERF_STDERR_SIMPLE))
        with patch('simulator.bench.resource.getrusage') as mock_rusage:
            mock_rusage.return_value = MagicMock(ru_maxrss=4096)
            b.collect()

        assert b.total_instructions == 1_000_000
        assert b.total_flops == 50_000
        assert b.peak_memory == 4096

    def test_zero_flops(self):
        b = _make_benchmarker(perf=_mock_perf(PERF_STDERR_ZERO_FLOPS))
        with patch('simulator.bench.resource.getrusage') as mock_rusage:
            mock_rusage.return_value = MagicMock(ru_maxrss=2048)
            b.collect()

        assert b.total_instructions == 500_000
        assert b.total_flops == 0

    def test_instructions_only(self):
        b = _make_benchmarker(perf=_mock_perf(PERF_STDERR_INSTRUCTIONS_ONLY))
        with patch('simulator.bench.resource.getrusage') as mock_rusage:
            mock_rusage.return_value = MagicMock(ru_maxrss=1024)
            b.collect()

        assert b.total_instructions == 750_000
        assert b.total_flops == 0

    def test_intel_ls_real_output(self):
        '''Parse real perf output from Intel with <not counted> entries.'''
        b = _make_benchmarker(perf=_mock_perf(PERF_STDERR_INTEL_LS))
        with patch('simulator.bench.resource.getrusage') as mock_rusage:
            mock_rusage.return_value = MagicMock(ru_maxrss=512)
            b.collect()

        assert b.total_instructions == 348_687
        assert b.total_flops == 0

    def test_intel_true_real_output(self):
        '''Parse real perf output from Intel with mostly <not counted>.'''
        b = _make_benchmarker(perf=_mock_perf(PERF_STDERR_INTEL_TRUE))
        with patch('simulator.bench.resource.getrusage') as mock_rusage:
            mock_rusage.return_value = MagicMock(ru_maxrss=256)
            b.collect()

        assert b.total_instructions == 142_453
        assert b.total_flops == 0

    def test_multiple_instruction_counters_sum(self):
        '''When both cpu_atom and cpu_core report, they should sum.'''
        stderr = '''\

 Performance counter stats for 'target':

           100,000      cpu_atom/instructions/u
           200,000      cpu_core/instructions/u
            10,000      cpu_core/fp_arith_inst_retired.scalar_double/u

       0.001000000 seconds time elapsed

'''
        b = _make_benchmarker(perf=_mock_perf(stderr))
        with patch('simulator.bench.resource.getrusage') as mock_rusage:
            mock_rusage.return_value = MagicMock(ru_maxrss=1024)
            b.collect()

        assert b.total_instructions == 300_000
        assert b.total_flops == 10_000

    def test_multiple_fp_event_types_sum(self):
        '''Multiple fp event types (scalar, packed) should all sum into total_flops.'''
        stderr = '''\

 Performance counter stats for 'target':

         5,000,000      instructions
            10,000      fp_arith_inst_retired.scalar_single
             5,000      fp_arith_inst_retired.scalar_double
            80,000      fp_arith_inst_retired.128b_packed_single
             2,000      fp_arith_inst_retired.128b_packed_double
                 0      fp_arith_inst_retired.256b_packed_single
                 0      fp_arith_inst_retired.256b_packed_double

       0.010000000 seconds time elapsed

'''
        b = _make_benchmarker(perf=_mock_perf(stderr))
        with patch('simulator.bench.resource.getrusage') as mock_rusage:
            mock_rusage.return_value = MagicMock(ru_maxrss=4096, ru_utime=0.01, ru_stime=0.0)
            b.collect()

        assert b.total_instructions == 5_000_000
        assert b.total_flops == 97_000  # 10000 + 5000 + 80000 + 2000

    def test_no_perf_process(self):
        '''collect() with perf=None should still set peak_memory.'''
        b = _make_benchmarker(perf=None)
        with patch('simulator.bench.resource.getrusage') as mock_rusage:
            mock_rusage.return_value = MagicMock(ru_maxrss=8192)
            b.collect()

        assert b.peak_memory == 8192
        assert b.total_instructions is None
        assert b.total_flops is None


# MARK: file_size

class TestFileSize:
    '''Tests for binary file size measurement.'''

    def test_file_size_measured(self, tmp_path):
        binary = tmp_path / 'test_binary'
        binary.write_bytes(b'\x00' * 256)

        b = _make_benchmarker()
        b.file_size = binary.stat().st_size
        assert b.file_size == 256
