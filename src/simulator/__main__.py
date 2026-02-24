'''
simulator/__main__.py
CLI entry point and orchestrator for the simulation pipeline.

Created: 2026-02-21
 Author: Maxence Morel Dierckx
'''
import argparse
from pathlib import Path
from .config import Config
from .data import Data
from .runner import Simulator
# from .report import Report, save_report


HERE = Path(__file__).parent.resolve()
ROOT = HERE.parent.parent


def simulate(args):
    config = Config(args.config)
    data = Data(args.data)
    simulator = Simulator(args.binary)

    for batch in data:
        result = simulator.run(batch)
        # report = Report(config, batch, result)
        # save_report(report)

    print(result.Aw) # testing


def main():
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
