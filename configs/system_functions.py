import numpy as np
import configs.constants as const
from dataclasses import dataclass
from configs.parameters import Sim_params
from typing import Callable

@dataclass(frozen=True)
class SystemFunctionsData:
    """Immutable container for system function parameters"""
    p: Sim_params
    particle_kind: str  # 'photon' or 'phonon'

    def __post_init__(self):
        if self.particle_kind not in ['photon', 'phonon']:
            raise ValueError("Invalid particle kind. Must be 'photon' or 'phonon'")

# Core physics calculations
def ohmic_spectrum(data: SystemFunctionsData, w: float) -> float:
    """Ohmic spectral density function"""
    gamma1 = data.p.ohmic_spec_gamma
    if w == 0.0:  # dephasing inducing noise
        return gamma1
    else:  # relaxation inducing noise
        return gamma1 / 2 * (w / (2 * np.pi)) * (w > 0.0)

def bose_einstein(data: SystemFunctionsData, omega: float) -> float:
    """Compute Bose-Einstein occupation number"""
    T_K = data.p.temperatures[data.particle_kind]
    if T_K == 0.0:
        return 0.0
    if (np.abs(omega) <= data.p.bose_omega_lower_cutoff or
        np.abs(omega) >= data.p.bose_omega_higher_cutoff):
        return 0.0

    exponent = omega / (const.KB_EV_PER_K * T_K)
    return 1.0 / (np.exp(exponent) - 1.0)

def spectral_density(data: SystemFunctionsData, omega: float) -> float:
    """Compute spectral density J(omega)"""
    if omega <= 0:
        return 0.0

    if data.particle_kind == "photon":
        return (data.p.spectral_photon_scaling * omega *
               np.exp(-omega / data.p.spectral_omega_c_photon))

    elif data.particle_kind == "phonon":
        return (2 * data.p.spectral_phonon_scaling * omega *
               np.exp(-(omega / data.p.spectral_omega_c_phonon) ** 2))

# Noise power calculations
def noise_power_spectrum(data: SystemFunctionsData, omega: float) -> float:
    """Total noise power spectrum S(omega)"""
    return (noise_power_emission(data, omega) +
            noise_power_absorption(data, omega))

def noise_power_emission(data: SystemFunctionsData, omega: float) -> float:
    """Emission component of noise power spectrum"""
    J = spectral_density(data, omega)
    return 2 * np.pi * J * (bose_einstein(data, omega) + 1)

def noise_power_absorption(data: SystemFunctionsData, omega: float) -> float:
    """Absorption component of noise power spectrum"""
    J = spectral_density(data, -omega)
    return 2 * np.pi * J * bose_einstein(data, -omega)

# Utility functions
def kelvin_to_energy(T_K: float) -> float:
    """Convert temperature in Kelvin to energy in eV"""
    return const.KB_EV_PER_K * T_K

def zero_func(omega: float) -> float:
    """Zero function for unused couplings"""
    return 0.0
