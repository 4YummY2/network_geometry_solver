# type: ignore
from dataclasses import dataclass
from configs.parameters import Sim_params
from core.geometry import Geometry
from configs.system_functions import SystemFunctionsData, noise_power_spectrum, noise_power_emission, noise_power_absorption, bose_einstein
import configs.system_functions as sys_funcs
from qutip import *
import numpy as np

@dataclass
class Hamiltonian:
    H: Qobj
    a_ops: list
    c_ops: list
    c_ops_photon_phonon: list
    c_ops_photon_phonon_str: list
    H_str: str
    a_ops_str: str
    c_ops_str: str

    @classmethod
    def build(cls, p: Sim_params, geometry: Geometry):
        """Factory method to construct all components"""
        H, H_str = build_hamiltonian(p, geometry)
        A_ops, A_str = build_a_ops(p, geometry)
        C_ops, C_str = build_c_ops(p, geometry)
        C_ops_photon_phonon = None
        C_ops_photon_phonon_str = None
        if p.using_L_decay_rate:
            C_ops_photon_phonon, C_ops_photon_phonon_str = build_c_ops_photon_phonon(p, geometry)

        return cls(
            H=H,
            a_ops=A_ops,
            c_ops=C_ops,
            c_ops_photon_phonon = C_ops_photon_phonon,
            c_ops_photon_phonon_str = C_ops_photon_phonon_str,
            H_str=H_str,
            a_ops_str="\n".join(A_str),
            c_ops_str="\n".join(C_str)

        )

def build_hamiltonian(p: Sim_params, geometry: Geometry) -> tuple[Qobj, str]:
    """Construct the system Hamiltonian"""
    emitter_omega = p.emitter_omega_vals
    trap_omega = p.trap_omega_vals
    couplings = geometry.emitter_edges_n_coupling

    num_emit = len(emitter_omega)
    num_trap = len(trap_omega) if p.using_traps else 0
    num_levels = num_emit + num_trap

    parts = []

    # Emitter terms
    H = Qobj(np.zeros((num_levels, num_levels)))  # Initialize with correct dimensions

    for i, omega in enumerate(emitter_omega):
        e = basis(num_levels, i)
        H += omega * e * e.dag()
        parts.append(f"{omega:.2f}|{i}⟩⟨{i}|")

    # Trap terms
    if p.using_traps:
        for i, omega in enumerate(trap_omega, start=num_emit):
            t = basis(num_levels, i)
            H += omega * t * t.dag()
            parts.append(f"{omega:.2f}|{i}⟩⟨{i}|")

    # Emitter couplings
    if couplings is not None:
        for edge in couplings:
            i, j, strength = map(float, edge)
            i, j = int(i), int(j)
            op = (basis(num_levels, i) * basis(num_levels, j).dag() +
                basis(num_levels, j) * basis(num_levels, i).dag())
            H += strength * op
            parts.append(f"{strength:.2f}(|{i}⟩⟨{j}| + |{j}⟩⟨{i}|)")

    return H, " + ".join(parts)

def build_a_ops(p: Sim_params, geometry: Geometry) -> tuple[list, list[str]]:
    """Build Bloch-Redfield a_ops using functional approach"""
    A_ops = []
    A_str = []
    num_emit = len(p.emitter_omega_vals)
    num_levels = num_emit + (len(p.trap_omega_vals) if p.using_traps else 0)
    basis_vects = [basis(num_levels, i) for i in range(num_levels)]
    g = basis_vects[0]

    if p.using_photons:
        photon_data = SystemFunctionsData(p, 'photon')
        a_ops = 0
        for i in range(1, num_emit):
            # Emission operator and noise spectrum

            sigma_minus = g * basis_vects[i].dag()
            sigma_plus = basis_vects[i] * g.dag()
            a_ops += sigma_minus + sigma_plus
            #a_op = sigma_plus + sigma_minus

            # A_ops.append([a_op, lambda w: noise_power_emission(photon_data, w)])
            # A_str.append(f"Photon emission |0⟩⟨{i}|")

            # A_ops.append([a_op, lambda w: noise_power_absorption(photon_data, w)])
            # A_str.append(f"Photon absorption |{i}⟩⟨0|")


            # photon_data = SystemFunctionsData(p, 'photon')
            # op_emission = g * basis_vects[i].dag()
            # A_ops.append([op_emission, lambda w: noise_power_spectrum(photon_data, -w)])

            # op_absorption = op_emission.dag()
            # A_ops.append([op_absorption, lambda w: noise_power_spectrum(photon_data, w)])
        A_ops.append([a_ops, lambda w: noise_power_emission(photon_data, w)])
        A_ops.append([a_ops, lambda w: noise_power_absorption(photon_data, w)])

    if p.using_phonons:
        phonon_data = SystemFunctionsData(p, 'phonon')
        for i in range(1, num_emit):
            op = basis_vects[i] * basis_vects[i].dag()
            A_ops.append([op, lambda w: noise_power_spectrum(phonon_data, w)])
            A_str.append(f"Phonon dephasing |{i}⟩⟨{i}|")

    return A_ops, A_str

def build_c_ops(p: Sim_params, geometry: Geometry) -> tuple[list[Qobj], list[str]]:
    """Build Lindblad collapse operators using functional approach"""
    c_ops = []
    c_str = []

    if p.using_traps:
        trap_data = SystemFunctionsData(p, 'photon')
        #trap_data = SystemFunctionsData(p, 'phonon')
        num_emit = len(p.emitter_omega_vals)
        num_levels = num_emit + len(p.trap_omega_vals)

        for i, (src_idx, trap_idx, rate) in enumerate(geometry.trap_edges_n_coupling):
            src = int(src_idx)
            trap = int(trap_idx)

            # Calculate energy difference
            omega_eT = abs(p.emitter_omega_vals[src] - p.trap_omega_vals[i])

            # Get Bose-Einstein distribution
            N_eT = bose_einstein(trap_data, omega_eT)

            # Decay into trap
            # emitter_omega = 2
            # trap_omega = 1
            # N_eT = N(omega = emitter_omega - trap_omega, T = (6000K))
            # rate = 0.002
            c_op = np.sqrt((N_eT + 1) * rate) * basis(num_levels, trap) * basis(num_levels, src).dag()
            c_ops.append(c_op)
            c_str.append(f"Trap decay {src}→{trap} Γ={rate:.2e}")

            # Reverse process
            c_op = np.sqrt(N_eT * rate) * basis(num_levels, src) * basis(num_levels, trap).dag()
            c_ops.append(c_op)
            c_str.append(f"Reverse trap {trap}→{src} Γ={rate:.2e}")

    return c_ops, c_str


def build_c_ops_photon_phonon(p: Sim_params, geometry: Geometry) -> tuple[list[Qobj], list[str]]:
    """Build Lindblad collapse operators including emitter photon/phonon processes"""
    c_ops = []
    c_str = []
    basis_vects = _get_basis_vects(p.dimension)
    g = basis_vects[0]

    # Photon processes (emission/absorption between ground and emitters)
    if p.using_photons:
        photon_data = SystemFunctionsData(p, 'photon')
        c_ops_abs = 0
        c_ops_emis = 0
        for i in range(1, p.num_emitters + 1):  # Emitters start at index 1
            omega_trans = p.emitter_omega_vals[i] - p.emitter_omega_vals[0]

            # Emission: excited -> ground
            gamma_emission = noise_power_emission(photon_data, omega_trans)
            c_op_emission = np.sqrt(gamma_emission) * (g * basis_vects[i].dag())
            c_ops_emis += c_op_emission


            # Absorption: ground -> excited
            gamma_absorption = noise_power_absorption(photon_data, -omega_trans)
            c_op_absorption = np.sqrt(gamma_absorption) * (basis_vects[i] * g.dag())
            c_ops_abs += c_op_absorption

        c_ops.append(c_ops_abs)
        c_ops.append(c_ops_emis)

    # Phonon processes (dephasing on emitters)
    if p.using_phonons:
        phonon_data = SystemFunctionsData(p, 'phonon')
        for i in range(1, p.num_emitters + 1):
            #gamma_phonon = noise_power_spectrum(phonon_data, 0)  # Zero frequency for dephasing
            #c_op_phonon = np.sqrt(gamma_phonon) * (basis_vects[i] * basis_vects[i].dag())
            sys_func_data = SystemFunctionsData(p, 'phonon')
            c_op_phonon = np.sqrt(p.phonon_decay) * (basis_vects[i] * basis_vects[i].dag())
            c_ops.append(c_op_phonon)
            c_str.append(f"Phonon dephasing |{i}><{i}| Γ={p.phonon_decay:.2e}")

    return c_ops, c_str


def _get_basis_vects(num_levels: int) -> list[Qobj]:
    """Generate basis vectors for the system"""
    return [basis(num_levels, i) for i in range(num_levels)]
