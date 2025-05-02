# type: ignore
from dataclasses import dataclass
from typing import Optional
import numpy as np
from qutip import bloch_redfield_tensor, brmesolve, steadystate, basis, qeye, Qobj, liouvillian
import qutip as qutip
from core.hamiltonian import Hamiltonian
from configs.parameters import Sim_params

@dataclass(frozen=True)
class SimulationData:
    """Immutable container for simulation state"""
    hamiltonian: Hamiltonian  # Hamiltonian dataclass
    results_time_sim: Optional[list[Qobj]] = None
    results_steady_state: Optional[Qobj] = None
    results_decay_rate: Optional[Qobj] = None
    times: Optional[np.ndarray] = None
    rho0: Optional[Qobj] = None

    @property
    def num_levels(self) -> int:
        return self.hamiltonian.H.shape[0]

# Core simulation functions
def compute_steady_state(data: SimulationData,
                        sec_cutoff: int = -1,
                        fock_basis: bool = True) -> SimulationData:
    """Calculate steady state and return new SimulationData"""
    br_tensor = bloch_redfield_tensor(
        data.hamiltonian.H,
        a_ops=data.hamiltonian.a_ops,
        c_ops=data.hamiltonian.c_ops,
        fock_basis=fock_basis,
        sec_cutoff=sec_cutoff
    )
    return SimulationData(
        **{**vars(data), 'results_steady_state': steadystate(br_tensor, method = 'direct')}
    )

def compute_dark_light_pop(params: Sim_params, sim_data, rho_states, emitter1: int, emitter2: int) -> dict:

    # Validate emitter indices
    num_emitters = params.num_emitters
    if emitter1 == emitter2:
        raise ValueError("Emitter indices must be distinct")
    if not (1 <= emitter1 <= num_emitters):
        raise ValueError(f"Emitter1 index must be between 1 and {num_emitters}")
    if not (1 <= emitter2 <= num_emitters):
        raise ValueError(f"Emitter2 index must be between 1 and {num_emitters}")

    # Create basis states for the two emitters (0-based indices in state vector)
    n = sim_data.num_levels
    e1 = qutip.basis(n, emitter1)
    e2 = qutip.basis(n, emitter2)

    # Create normalized dark and light states
    dark_state = (e1 - e2).unit()
    light_state = (e1 + e2).unit()

    # Create projection operators
    dark_proj = dark_state * dark_state.dag()
    light_proj = light_state * light_state.dag()

    # Initialize results dictionary
    results = {
        'dark_pop_steady': None,
        'light_pop_steady': None
    }

    # Calculate steady state populations if available
    results['dark_pop_steady'] = np.mean([np.real(qutip.expect(dark_proj, rho_state)) for rho_state in rho_states])
    results['light_pop_steady'] = np.mean([np.real(qutip.expect(light_proj, rho_state)) for rho_state in rho_states])

    return results

def compute_decay_rate(data: SimulationData) -> SimulationData:

    transport_rate = None
    try:
        total_c_ops = data.hamiltonian.c_ops + data.hamiltonian.c_ops_photon_phonon

        L = liouvillian(H = data.hamiltonian.H, c_ops = total_c_ops)
        evals = L.eigenenergies()

        # Filter and sort eigenvalues
        #nonzero_evals = [ev for ev in evals if abs(ev.real) > 1e-12]
        real_evals = [x.real for x in evals]
        sorted_evals = sorted(real_evals, reverse = True)

        if sorted_evals:
            # Find first nonzero rate
            transport_rate = np.abs(sorted_evals[1])
        else:
            print("\nWarning: No non-zero eigenvalues found")

    except Exception as e:
        print(f"\nLiouvillian calculation failed: {str(e)}")

    return SimulationData(
        **{**vars(data), 'results_decay_rate': transport_rate}
    )

def simulate_time_evolution(data: SimulationData,
                           tlist: np.ndarray,
                           sec_cutoff: int = -1) -> SimulationData:
    """Run time evolution and return new SimulationData"""
    if data.rho0 is None:
        raise ValueError("Initial state rho0 must be set before simulation")

    results = brmesolve(
        data.hamiltonian.H,
        data.rho0,
        tlist,
        a_ops=data.hamiltonian.a_ops,
        c_ops=data.hamiltonian.c_ops,
        sec_cutoff=sec_cutoff
    ).states

    return SimulationData(
        **{**vars(data), 'results_time_sim': results, 'times': tlist}
    )

# Initial state generators
def initial_ground_state(num_levels: int) -> Qobj:
    """Create ground state density matrix (|0><0|)"""
    g = basis(num_levels, 0)
    return g * g.dag()

def initial_excited_state(num_levels: int, index: int = 1) -> Qobj:
    """Create excited state density matrix at specified index (|i><i|)"""
    e = basis(num_levels, index)
    return e * e.dag()

def initial_superposition_state(num_levels: int) -> Qobj:
    """Create equal superposition state density matrix"""
    psi = sum([basis(num_levels, i) for i in range(num_levels)]).unit()
    return psi * psi.dag()

def initial_diagonal_state(num_levels: int) -> Qobj:
    """Create maximally mixed state"""
    return qeye(num_levels) / num_levels

# State management
def set_initial_state(data: SimulationData, state: Qobj) -> SimulationData:
    """Return new SimulationData with updated initial state"""
    if state.shape != (data.num_levels, data.num_levels):
        raise ValueError("State dimension must match number of levels")
    return SimulationData(**{**vars(data), 'rho0': state})

def set_custom_initial_state(data: SimulationData, state_vector: Qobj) -> SimulationData:
    """Set initial state from custom state vector"""
    if len(state_vector) != data.num_levels:
        raise ValueError("State vector dimension must match number of levels")
    return set_initial_state(data, state_vector * state_vector.dag())

# Analysis functions
def get_final_trap_population_steady_state(data: SimulationData) -> float:
    """Get the final trap population from steady state solution"""
    if data.results_steady_state is None:
        raise ValueError("No steady state results available")


    # print(f"Hamiltonain: {data.hamiltonian.H}")
    # print(f"a_ops: {data.hamiltonian.a_ops}")
    # print(f"c_ops: {data.hamiltonian.c_ops}")
    rho_ss = np.abs(np.real(data.results_steady_state.full()))

    # Take the trace
    trace_val = np.trace(rho_ss)

    if trace_val > 1.0:
        print(f'before {rho_ss[-1, -1]}')
        rho_ss = rho_ss / trace_val  # normalize
        print(f'after {rho_ss[-1, -1]}')

    return rho_ss[-1, -1]  # steady state population in last state
    

def get_final_trap_population_time_evolution(data: SimulationData) -> float:
    """Get the final trap population from time evolution results"""
    if data.results_time_sim is None:
        raise ValueError("No time evolution results available")
    return np.real(data.results_time_sim[-1][-1, -1])

def get_decay_rate(data: SimulationData) -> float:
    if data.results_decay_rate is None:
        raise ValueError("No decay rate available")
    return data.results_decay_rate
