# type: ignore
import qutip as qutip
import numpy as np
from typing import Dict, Optional

from configs.parameters import Sim_params

def analyze_dark_light_dynamics(params: Sim_params, sim_data, emitter1: int, emitter2: int, metric_type = 'time_evolution') -> Dict:
    """
    Analyze population dynamics of dark and light states formed by two emitters.

    Args:
        sim_data: SimulationData object containing results
        emitter1: Index of first emitter (1-based)
        emitter2: Index of second emitter (1-based)

    Returns:
        Dictionary containing:
        - dark_pop_time: Population of dark state over time
        - light_pop_time: Population of light state over time
        - dark_pop_steady: Steady state population of dark state
        - light_pop_steady: Steady state population of light state
    """
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
        'dark_pop_time': None,
        'light_pop_time': None,
        'dark_pop_steady': None,
        'light_pop_steady': None
    }

    # Calculate time evolution populations if available
    if sim_data.results_time_sim is not None and metric_type == 'time_evolution':
        results['dark_pop_time'] = np.array([np.real(qutip.expect(dark_proj, rho))
                                           for rho in sim_data.results_time_sim])
        results['light_pop_time'] = np.array([np.real(qutip.expect(light_proj, rho))
                                            for rho in sim_data.results_time_sim])

    # Calculate steady state populations if available
    if sim_data.results_steady_state is not None and metric_type == 'steady_state':
        results['dark_pop_steady'] = np.real(qutip.expect(dark_proj, sim_data.results_steady_state))
        results['light_pop_steady'] = np.real(qutip.expect(light_proj, sim_data.results_steady_state))


    return results
