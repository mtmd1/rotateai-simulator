'''
simulator/runner.py
Orchestrator for the simulation layer logic.

Created: 2026-02-24
 Author: Maxence Morel Dierckx
'''
import sys
import struct
import subprocess
import numpy as np
from pathlib import Path
from simulator.bench import Benchmarker


class SimResult:
    '''An object holding all the measured results of the simulation'''

    def __init__(self, N: int) -> None:
        self.N = N
        self.Mw: np.ndarray = np.zeros((N, 3))
        self.Aw: np.ndarray = np.zeros((N, 3))
        self.sample_index = 0
        self.benchmark = None


    def add_row(self, sample: list[float]) -> None:
        '''Add a corrected sample. Format: mwx mwy mwz awx awy awz.'''
        self.Mw[self.sample_index] = sample[:3]
        self.Aw[self.sample_index] = sample[3:6]
        self.sample_index += 1


class Simulator:
    '''The executor of single simulations.'''

    def __init__(self, binary_path_str: str) -> None:
        '''Load and validate the binary file.'''
        binary_path = Path(binary_path_str)
        if not binary_path.is_absolute():
            binary_path = Path.cwd() / binary_path_str
        
        if binary_path.is_file():
            self.binary = binary_path

        else:
            print(f'Binary path {binary_path} not found.', file=sys.stderr)
            exit(1)


    def run(self, data: dict[str, np.ndarray], progress=None) -> SimResult:
        '''Run the binary on the given data and return the simulation result.
        progress: optional callable(iterable, total=int) -> iterable (for tqdm).'''
        p = data['p'] # Pressure
        M = data['M'] # Magnetometer
        A = data['A'] # Accelerometer
        steps = len(A) # Data lengths all the same (from data.py)
        result = SimResult(steps)

        # Open the binary process
        process = subprocess.Popen(
            [self.binary],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
        )

        # Pass it to the benchmarker
        benchmarker = Benchmarker(self.binary, process)
        result.benchmark = benchmarker

        # Run the main simulation loop
        loop = range(steps)
        if progress:
            loop = progress(loop, total=steps)
        for i in loop:

            # Input contract p mx my mz ax ay az
            sample = struct.pack(
                '7f', 
                p[i], 
                M[i][0], M[i][1], M[i][2], 
                A[i][0], A[i][1], A[i][2]
            )

            process.stdin.write(sample)
            process.stdin.flush() # Binary receives it immediately

            # Block until output is available
            output = process.stdout.read(24) # 6 float32s = 24 bytes
            if len(output) != 24:
                print(f'Binary returned {len(output)} bytes, expected 24.', file=sys.stderr)
                exit(1)

            # Output contract mwx mwy mwz awx awy awz
            try:
                corrected_sample = struct.unpack('6f', output)
            except struct.error as e:
                print(f'Parsing binary output failed: {e}', file=sys.stderr)
                exit(1)

            result.add_row(corrected_sample)
        
        process.stdin.close()
        remaining = process.stdout.read()
        if remaining:
            print(f'Warning: binary wrote {len(remaining)} extra bytes after expected output.')
        process.wait()
        benchmarker.collect()

        return result
