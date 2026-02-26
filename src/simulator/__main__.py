'''
simulator/__main__.py
CLI entry point and orchestrator for the simulation pipeline.

Created: 2026-02-21
 Author: Maxence Morel Dierckx
'''
import sys
import signal
import argparse
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from functools import partial

from simulator.config import Config
from simulator.data import Data
from simulator.runner import Simulator
from simulator.report import save_report


def handle_shutdown(signum, frame):
    '''Clean exit on SIGINT/SIGTERM.'''
    name = signal.Signals(signum).name
    print(f'\nReceived {name}, shutting down.', file=sys.stderr)
    sys.exit(1)


def validate_output_path(output_path_str: str) -> Path:
    '''Validate that the output path exists.'''
    if output_path_str is None:
        return Path.cwd()
    
    output_path = Path(output_path_str)
    if not output_path.is_absolute():
        output_path = Path.cwd() / output_path_str
    
    if output_path.is_file():
        print(f'Output path {output_path} is not a directory.', file=sys.stderr)
        sys.exit(1)
    
    if output_path.is_dir():
        return output_path
    
    else:
        try:
            output_path.mkdir(parents=True)
            return output_path
        except Exception as e:
            print(f'Output path {output_path} does not exist, and creation failed: {e}', file=sys.stderr)
            sys.exit(1)


def simulate(args):
    '''Load the config, data and binary and run the full simulation.'''
    config = Config(args.config)
    data = Data(args.data)
    simulator = Simulator(args.binary)
    output_path = validate_output_path(args.output)

    report_prefix = f'simreport_{args.binary.split("/")[-1]}'

    for i, batch in enumerate(data):
        result = simulator.run(batch, progress=partial(tqdm, desc=f'Batch {i + 1}/{len(data)}'))

        batch_name = batch['_source'].removesuffix('.mat').replace('_', '-')
        timestamp = datetime.now().strftime('%H%M%S')

        save_report(f'{report_prefix}_{batch_name}_{timestamp}', config, batch, result, output_path)
    
    print('Done.')


def main():
    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    parser = argparse.ArgumentParser(
        prog='simulate',
        description='A simulator for RotateAI model inference on virtual microcontroller environments.'
    )

    parser.add_argument(
        '--config', '-c',
        required=True,
        help='path to simulation config TOML file'
    )

    parser.add_argument(
        '--binary', '-b',
        required=True,
        help='path to simulation binary file'
    )

    parser.add_argument(
        '--data', '-d',
        required=True,
        help='path to MAT data directory or file'
    )

    parser.add_argument(
        '--output', '-o',
        required=False,
        default=None,
        help='path to save the output reports'
    )

    parser.set_defaults(func=simulate)
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
