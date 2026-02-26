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
from functools import partial
from simulator.config import Config
from simulator.data import Data
from simulator.runner import Simulator
from simulator.report import save_report


HERE = Path(__file__).parent.resolve()
ROOT = HERE.parent.parent


def handle_shutdown(signum, frame):
    '''Clean exit on SIGINT/SIGTERM.'''
    name = signal.Signals(signum).name
    print(f'\nReceived {name}, shutting down.', file=sys.stderr)
    sys.exit(1)


def simulate(args):
    '''Load the config, data and binary and run the full simulation.'''
    config = Config(args.config)
    data = Data(args.data)
    simulator = Simulator(args.binary)
    report_prefix = f'simreport_{args.binary.split("/")[-1]}'

    for i, batch in enumerate(data):
        result = simulator.run(batch, progress=partial(tqdm, desc=f'Batch {i + 1}/{len(data)}'))
        save_report(f'{report_prefix}_{i}', config, batch, result)
    
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

    parser.set_defaults(func=simulate)
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
