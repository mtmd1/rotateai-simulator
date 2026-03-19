'''
simulator/report.py
Calculate derived estimates, error, and save them with 
the benchmarking statistics as a JSON report.

Created: 2026-02-25
 Author: Maxence Morel Dierckx
'''
import sys
import json
import numpy as np
from datetime import datetime
from pathlib import Path

from simulator.config import Config
from simulator.runner import SimResult


def r(value, sigfigs=3):
    '''Round a value or numpy array to a given number of significant figures.'''
    if isinstance(value, np.ndarray):
        return [r(v, sigfigs) for v in value]
    value = float(value)
    if value == 0:
        return 0.0
    digits = -int(np.floor(np.log10(abs(value)))) + (sigfigs - 1)
    return round(value, digits)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super().default(obj)


def save_report(binary: str, config: Config, data: dict[str, np.ndarray], result: SimResult, output_path: Path) -> None:
    '''Calculate derived estimates, error, and save them with
    the benchmarking statistics as a JSON report.'''
    (minimum_frequency,
     energy_per_inference,
     duty_cycle,
     power_consumption) = derive_metrics(config, result)

    (mae_mw,
     mae_aw,
     rmse_mw,
     rmse_aw) = calculate_errors(data, result)

    data_file = data['_source']
    batch_name = data_file.removesuffix('.mat').replace('_', '-')
    suffix = format(hash(result) % 0xFFFF, '04x')
    name = f'simreport_{binary}_{batch_name}_{suffix}'

    report = {
        'name': name,
        'binary': binary,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'data_file': data_file,
        'config': config.to_dict(),
        'benchmark': {
            'file_size_KiB': r(result.benchmark.file_size / 1024),
            'memory_usage_KiB': r(result.arena_used_bytes / 1024) if result.arena_used_bytes else None,
            'instructions_per_inference': int(result.benchmark.total_instructions / result.N),
            'FLOPS_per_inference': int(result.benchmark.total_flops / result.N),
            'simulation_time_s': r(result.benchmark.cpu_time),
            'cpu_time_s': r(result.benchmark.cpu_time),
            'wall_time_s': r(result.wall_time),
            'output_count': result.output_count,
            'output_ratio': r(result.output_ratio),
        },
        'derived': {
            'minimum_operating_frequency_MHz': r(minimum_frequency),
            'energy_per_inference_mJ': r(energy_per_inference),
            'duty_cycle': r(duty_cycle),
            'power_consumption_mW': r(power_consumption),
        },
        'error': {
            'MAE_Mw_uT': r(mae_mw),
            'RMSE_Mw_uT': r(rmse_mw),
            'MAE_Aw_g': r(mae_aw),
            'RMSE_Aw_g': r(rmse_aw),
        }
    }
    with open(output_path / f'{name}.json', 'w') as f:
        json.dump(report, f, indent=4, cls=NumpyEncoder)
    
    
def derive_metrics(config: Config, result: SimResult) -> tuple[float]:
    '''Return the estimated values derived as shown in docs/derivations.pdf.'''

    instructions_per_inference = result.benchmark.total_instructions / result.N

    # Minimum operating frequency (MHz) for duty cycle = 1
    minimum_frequency = (instructions_per_inference * config.sample_rate) / (config.DMIPS_per_MHz * 1e6)

    # Energy per inference (mJ) — frequency-independent within a voltage range
    charge_per_inference = config.uA_per_MHz * instructions_per_inference / (config.DMIPS_per_MHz * 1e6)
    energy_per_inference = config.voltage * charge_per_inference / 1e3

    # Duty cycle at maximum operating frequency
    duty_cycle = minimum_frequency / config.max_frequency

    # Average power consumption (mW): active inference + sleep current
    power_consumption = (energy_per_inference * config.sample_rate
                         + config.voltage * config.sleep_current_uA * 1e-3 * (1 - duty_cycle))

    return (minimum_frequency, energy_per_inference, duty_cycle, power_consumption)
    

def calculate_errors(data: dict[str, np.ndarray], result: SimResult) -> tuple[tuple[float]]:
    '''Return the MAE and RMSE values for Aw and Mw over all input samples.
    Sparse predictions are linearly interpolated to full length before comparison.'''
    N = len(data['Mw'])
    x_out = np.array(result.output_indices)
    x_all = np.arange(N)

    predicted_Mw = np.column_stack([np.interp(x_all, x_out, result.Mw[:, c]) for c in range(3)])
    predicted_Aw = np.column_stack([np.interp(x_all, x_out, result.Aw[:, c]) for c in range(3)])

    ground_Mw = data['Mw'].astype('float')
    ground_Aw = data['Aw'].astype('float')

    mae_mw = np.mean(np.abs(ground_Mw - predicted_Mw), axis=0)
    mae_aw = np.mean(np.abs(ground_Aw - predicted_Aw), axis=0)

    rmse_mw = np.sqrt(np.mean((ground_Mw - predicted_Mw) ** 2, axis=0))
    rmse_aw = np.sqrt(np.mean((ground_Aw - predicted_Aw) ** 2, axis=0))

    return (mae_mw, mae_aw, rmse_mw, rmse_aw)

