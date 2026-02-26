# RotateAI Simulator

Tests ML inference binaries against real whale sensor data without embedded hardware. Measures file size, peak memory, instruction count, and FLOPs, then derives estimates of operating frequency, energy, duty cycle, and power consumption for a target MCU. Computes per-channel MAE and RMSE against ground truth.

Outputs a JSON report per deployment.

## Installation

```sh
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Requires `perf` (Linux) for instruction and FLOP counting.

## Usage

```sh
simulate --config CONFIG --binary BINARY --data DATA [--output OUTPUT]
```

| Flag | Description |
|------|-------------|
| `--config`, `-c` | Path to TOML config file |
| `--binary`, `-b` | Path to inference binary |
| `--data`, `-d` | Path to `.mat` file or directory of `.mat` files |
| `--output`, `-o` | Directory for reports (default: cwd) |

Example:

```sh
sim -c config/stm32u5.toml -b working/inference -d ~/data/mn11_157aprh.mat -o reports/
```

## Configuration

All hardware-specific constants are set in the config TOML. See `config/stm32u5.toml` for defaults.

| Key | Description | Unit |
|-----|-------------|------|
| `sample_rate` | Sensor sampling frequency | Hz |
| `voltage` | Supply voltage (VDD) | V |
| `DMIPS_per_MHz` | Dhrystone throughput | DMIPS/MHz |
| `uA_per_MHz` | Current draw per MHz | uA/MHz |
| `max_frequency` | Maximum operating frequency | MHz |

## Report Format

The simulator produces JSON file(s) containing:

- **benchmark** stats - file size, peak memory, instructions and FLOPs per inference
- **derived** stats - minimum operating frequency, energy per inference, duty cycle, power consumption
- **error** stats - per-channel MAE and RMSE for accelerometer (g) and magnetometer (uT)

See `docs/derivations.pdf` for how derived metrics are estimated.

## Binary Protocol

The inference binary communicates over stdin/stdout using binary float32:

- **Input**: 7 floats per sample — `p, mx, my, mz, ax, ay, az` (28 bytes)
- **Output**: 6 floats per sample — `mwx, mwy, mwz, awx, awy, awz` (24 bytes)

The binary must flush stdout after each output sample.

## Testing

```sh
pip install -e ".[test]"
pytest
```
