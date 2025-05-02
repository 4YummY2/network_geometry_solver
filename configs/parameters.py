from dataclasses import dataclass, field
import numpy as np

@dataclass
class Sim_params:
    num_emitters: int
    d_real: float
    emitter_omega: float = 2.0
    trap_omega: float = 1.0
    ground_omega: float = 0.0
    coupling_strength: float = 0.03
    temperatures: dict = field(default_factory=lambda: {
        'photon': 6000,
        'phonon': 300
    })
    # decay_rates: dict = field(default_factory=lambda: {
    #     'trap': 0.002,
    #     'emitter': 0.001
    # })
    trap_decay_rate = 0.0002

    #emitter_omega_vals: list[float] = [] * (num_emitters + 1)
    using_traps: bool = True
    using_photons: bool = True
    using_phonons: bool = True
    using_L_decay_rate: bool = False
    #trap_omega_vals = None
    #trap_edges_and_coupling = np.array([[num_emitters, num_emitters + 1, 0]])

    # System Functions
    bose_omega_lower_cutoff = 1e-12
    bose_omega_higher_cutoff = 20

    spectral_omega_c_photon: float = 2.0
    spectral_omega_c_phonon: float = 0.1
    spectral_photon_scaling: float = 0.0000014
    spectral_phonon_scaling: float = 0.025

    # For c_ops (from "Photocell Optimization Using Dark State Protection" paper)
    phonon_decay = 0.005


    dipole_strength_min = 0.00001
    dipole_strength_scaling = 0.02

    ohmic_spec_gamma = 0.5

    dimension: int = field(init=False)
    emitter_omega_vals: np.ndarray = field(init=False)
    trap_omega_vals: np.ndarray = field(init=False)

    def __post_init__(self):
        self.emitter_omega_vals = np.array([self.ground_omega] + [self.emitter_omega] * self.num_emitters)
        self.trap_omega_vals = np.array([self.trap_omega])  # Trap energy
        self.dimension = len(self.emitter_omega_vals) + len(self.trap_omega_vals)
