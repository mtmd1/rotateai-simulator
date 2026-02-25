'''
simulator/bench.py
Utility for measuring performance of the live inference process.

Created: 2026-02-24
 Author: Maxence Morel Dierckx
'''
import sys
import resource
import platform
import subprocess
from pathlib import Path


class Benchmarker:
    '''Main benchmarking class.'''

    def __init__(self, binary_path: Path, process: subprocess.Popen) -> None:
        '''Initialise one time measurements and start the perf process'''
        self.file_size = binary_path.stat().st_size
        self.peak_memory = None
        self.total_instructions = None
        self.total_flops = None
        self.cpu_time = None

        try:
            perf_events = self.get_perf_events()
        except Exception as e:
            print(f'Failed to get perf events: {e}.', file=sys.stderr)
            exit(1)

        self.validate_perf_events(perf_events)

        self.perf = self.open_perf_process(process.pid, perf_events)

    
    def get_perf_events(self) -> str:
        '''Get the appropriate perf events based on platform.'''
        machine = platform.machine()

        if 'x86_64' in machine or 'i386' in machine or 'i686' in machine:
            cpu_info = subprocess.run(
                ['grep', '-i', 'vendor_id', '/proc/cpuinfo'],
                capture_output=True,
                text=True
            ).stdout

            if 'GenuineIntel' in cpu_info: 
                # Intel
                return 'instructions,fp_arith_inst_retired.128b_packed_single'
            elif 'AuthenticAMD' in cpu_info: 
                # AMD
                return 'instructions,fp_ret_sse_avx_ops.double'
            else:
                return 'instructions'
        
        elif 'aarch64' in machine or 'arm' in machine:
            # ARM -- couldn't confirm so let's hope we never get here!!
            return 'instructions,fp_fixed_ops_spec'
        
        else:
            return 'instructions'

    
    def validate_perf_events(self, perf_events: str):
        result = subprocess.run(
            ['perf', 'stat', '-e', perf_events, '--', 'true'],
            capture_output=True, 
            text=True
        )
        if 'event syntax error' in result.stderr.lower() or '<not supported>' in result.stderr.lower():
            print(f'Perf events not supported: {perf_events}.', file=sys.stderr)
            exit(1)


    def open_perf_process(self, process_pid: int, perf_events: str) -> subprocess.Popen:        
        try:
            perf = subprocess.Popen(
                ['perf', 'stat', '-p', str(process_pid), '-e', perf_events],
                stderr=subprocess.PIPE,
                text=True
            )
            return perf

        except FileNotFoundError:
            print(f'perf is not installed.', file=sys.stderr)
            exit(1)

        except PermissionError:
            print('Permission denied to run perf. Elevated privileges are required.', file=sys.stderr)
            exit(1)


    def collect(self) -> None:
        '''Collect the results of perf and resource.getrusage.'''
        usage = resource.getrusage(resource.RUSAGE_CHILDREN)
        self.peak_memory = usage.ru_maxrss
        self.cpu_time = usage.ru_utime + usage.ru_stime
        
        if self.perf is not None:
            _, stderr = self.perf.communicate()

            perf_results_raw = stderr.strip().split()
            perf_results = {}

            for i in range(len(perf_results_raw) - 1):
                try:
                    stat = int(perf_results_raw[i].replace(',', ''))
                    perf_results[perf_results_raw[i + 1]] = stat
                except ValueError:
                    continue
            
            self.total_instructions = 0
            self.total_flops = 0

            for key, value in perf_results.items():
                if 'instructions' in key:
                    self.total_instructions += value
                elif 'fp' in key:
                    self.total_flops += value
