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
| `cmdline` | Extra arguments passed to binary (optional) | — |

## Report Format

The simulator produces JSON file(s) containing:

- **benchmark** stats - file size, peak memory, instructions and FLOPs per inference, output count and ratio
- **derived** stats - minimum operating frequency, energy per inference, duty cycle, power consumption
- **error** stats - per-channel MAE and RMSE for accelerometer (g) and magnetometer (uT)

See `docs/derivations.pdf` for how derived metrics are estimated.

## Binary Protocol

The inference binary communicates over stdin/stdout in lock-step. For each input sample:

- **Input**: 7 float32s — `ax, ay, az, mx, my, mz, p` (28 bytes)
- **Flag byte**: 1 byte — `0x01` if output follows, `0x00` if no output for this sample
- **Output** (only when flag is `0x01`): 6 float32s — `awx, awy, awz, mwx, mwy, mwz` (24 bytes)

The binary must write the flag byte after consuming each input sample, and flush stdout after each response. Binaries that produce output on every sample write `0x01` followed by 24 bytes each time.

## Testing

```sh
pip install -e ".[test]"
pytest
```
