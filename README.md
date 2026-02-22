# RotateAI Simulator

A simulator for RotateAI model inference on virtual microcontroller environments.

## Installation

```sh
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

# Usage

```sh
simulate [-h] --config CONFIG --binary BINARY --data DATA
```

Example:
```sh
simulate --config config/stm32u5.toml --binary working/inference --data working/mn11_157aprh.mat
```

Arguments:
- `--config`: Path to the simulation config TOML file, absolute or relative to cwd.
- `--binary`: Path to the binary file to test.
- `--data` : Path to a MAT file or directory containing MAT files.
