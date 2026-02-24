'''
simulator/runner.py
Orchestrator for the simulation layer logic.

Created: 2026-02-24
 Author: Maxence Morel Dierckx
'''
import sys
import subprocess
import numpy as np
from pathlib import Path
from .bench import Benchmarker


class SimResult:
    '''An object holding all the measured results of the simulation'''

    def __init__(self, N: int) -> None:
        # Sample data
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


    def run(self, data: dict[str, np.ndarray]) -> SimResult:
        '''Run the binary on the given data and return the simulation result.'''
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
            text=True,
            bufsize=1
        )

        # Pass it to the benchmarker
        benchmarker = Benchmarker(self.binary, process)
        result.benchmark = benchmarker

        # Run the main simulation loop
        for i in range(steps):

            # Input contract p mx my mz ax ay az
            sample = f'{p[i]} {M[i][0]} {M[i][1]} {M[i][2]} {A[i][0]} {A[i][1]} {A[i][2]}'

            process.stdin.write(sample + '\n')
            process.stdin.flush() # Binary receives it immediately

            # Block until output is available
            output = process.stdout.readline()

            # Output contract mwx mwy mwz awx awy awz
            try:
                corrected_sample = [float(v) for v in output.strip().split()]
            except ValueError:
                print(f'Binary returned non-float value in output.', file=sys.stderr)
                exit(1)

            result.add_row(corrected_sample)
        
        process.wait()
        benchmarker.collect()

        return result
