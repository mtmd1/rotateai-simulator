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
from simulator.config import Config
from simulator.runner import SimResult


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super().default(obj)


def save_report(name: str, config: Config, data: dict[str, np.ndarray], result: SimResult) -> None:
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

    report = {
        'name': name,
        'config': config.to_dict(),
        'benchmark': {
            'file_size_bytes': result.benchmark.file_size,
            'peak_memory_KB': result.benchmark.peak_memory,
            'instructions_per_inference': result.benchmark.total_instructions / result.N,
            'FLOPS_per_inference': result.benchmark.total_flops / result.N
        },
        'derived': {
            'minimum_operating_frequency_MHz': minimum_frequency,
            'energy_per_inference_uJ': energy_per_inference,
            'duty_cycle_ratio': duty_cycle,
            'power_consumption_uW': power_consumption
        },
        'error': { 
            'MAE_Mw': mae_mw, 
            'RMSE_Mw': rmse_mw, 
            'MAE_Aw': mae_aw, 
            'RMSE_Aw': rmse_aw}
    }
    with open(f'{name}.json', 'w') as f:
        json.dump(report, f, indent=4, cls=NumpyEncoder)
    
    
def derive_metrics(config: Config, result: SimResult) -> tuple[float]:
    '''Return the estimated values derived as shown in docs/derivations.pdf.'''
    # TODO: write docs/derivations.pdf

    instructions_per_inference = result.benchmark.total_instructions / result.N

    # Minimum operating frequency
    minimum_frequency = (instructions_per_inference * config.sample_rate * config.safety_margin) / (config.DMIPS_per_MHz * 1e6)

    # Energy per inference
    charge_per_inference = config.uA_per_MHz * instructions_per_inference / (config.DMIPS_per_MHz * 1e6)
    energy_per_inference = config.voltage * charge_per_inference

    # Duty cycle
    time_per_inference = result.benchmark.cpu_time / result.N
    sample_period = 1 / config.sample_rate
    duty_cycle = time_per_inference / sample_period
    
    # Power consumption
    power_consumption = energy_per_inference * config.sample_rate

    return (minimum_frequency, energy_per_inference, duty_cycle, power_consumption)
    

def calculate_errors(data: dict[str, np.ndarray], result: SimResult) -> tuple[tuple[float]]:
    '''Return the MAE and RMSE values for Aw and Mw.'''
    ground_Mw = data['Mw']
    ground_Aw = data['Aw']

    predicted_Mw = result.Mw
    predicted_Aw = result.Aw

    mae_mw = np.mean(np.abs(ground_Mw.astype('float') - predicted_Mw.astype('float')), axis=0)
    mae_aw = np.mean(np.abs(ground_Aw.astype('float') - predicted_Aw.astype('float')), axis=0)

    rmse_mw = np.sqrt(np.mean((ground_Mw.astype('float') - predicted_Mw.astype('float')) ** 2, axis=0))
    rmse_aw = np.sqrt(np.mean((ground_Aw.astype('float') - predicted_Aw.astype('float')) ** 2, axis=0))

    return (mae_mw, mae_aw, rmse_mw, rmse_aw)

