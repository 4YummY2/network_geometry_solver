from dataclasses import dataclass
from configs.parameters import Sim_params
from scipy.spatial.distance import cdist
import numpy as np


@dataclass
class Geometry:
    positions: dict
    emitter_edges_n_coupling: np.ndarray
    trap_edges_n_coupling: np.ndarray
    d_real : float
    num_emitters: float


def make_geometry_from_positions(p: Sim_params, positions: dict) -> Geometry:
    #positions = _create_symm_ring_positions()
    emitter_edges_n_coupling, trap_edges_n_coupling = _compute_coupling_geom(p, positions)

    return Geometry(
        positions = positions,
        emitter_edges_n_coupling = emitter_edges_n_coupling,
        trap_edges_n_coupling = trap_edges_n_coupling,
        d_real = p.d_real,
        num_emitters = p.num_emitters
    )

def make_symm_ring_geom(p: Sim_params, y_params: dict) -> Geometry:
    positions = _create_symm_ring_positions(p.num_emitters, p.d_real, y_params)
    return make_geometry_from_positions(p, positions)


def make_linear_chain_geom(p: Sim_params) -> Geometry:
    positions = _create_linear_chain_positions(p.num_emitters, p.d_real)
    return make_geometry_from_positions(p, positions)


def _create_linear_chain_positions(num_emitters: int, d_real: float) -> dict:
    positions = {}
    x_spacing = d_real / (num_emitters)

    for i in range(1, num_emitters + 1):
        positions[i] = (i * x_spacing, 0.0)

    positions[num_emitters + 1] = (d_real, 0.0)  # Trap position
    return positions


def _create_symm_ring_positions(num_emitters: int, d_real: float, y_params: dict) -> dict:
    if num_emitters % 2 != 0:
        raise ValueError("Number of emitters must be even for symmetric ring geometry")

    positions = {}
    x_spacing = d_real / ((num_emitters//2) + 1)

    positions[1] = (0.0, 0.0)
    for i in range(1, num_emitters//2 + 1):
        positions[2*i] = (i*x_spacing, -y_params.get(f'y{i}', 0))
        positions[2*i+1] = (i*x_spacing, y_params.get(f'y{i}', 0))

    positions[num_emitters+1] = (d_real, 0.0)
    return positions


def _compute_coupling_geom(p: Sim_params, positions: dict):
    """
    Calculate couplings between all pairs of nodes based on 1/r^3.
    Returns:
    """
    threshold_J = p.dipole_strength_min
    alpha = p.dipole_strength_scaling

    nodes = sorted(positions.keys())
    trap_node = nodes[-1]  # Last noden is the trap

    # Extract coordinates in order of nodes
    coords = np.array([positions[node] for node in nodes])

    # Compute pairwise distances
    dist_matrix = cdist(coords, coords, metric='euclidean')

    # Initialize coupling matrix with zeros
    coupling_matrix = np.zeros_like(dist_matrix)

    # Find indices where distance is non-zero
    non_zero_mask = dist_matrix != 0

    # Compute 1/r^3 only for non-zero distances
    coupling_matrix[non_zero_mask] = 1 * alpha / (dist_matrix[non_zero_mask] ** 3)

    # Build edge lists
    emitter_edges = []
    trap_edges = []

    for i, node_i in enumerate(nodes):
        for j, node_j in enumerate(nodes[i+1:], start=i+1):
            J = coupling_matrix[i, j]
            if J > threshold_J:
                if node_j == trap_node:
                    trap_edges.append([node_i, node_j, J])
                else:
                    emitter_edges.append([node_i, node_j, J])


    # Only couple the trap to the closest emitter
    if len(trap_edges) == 0:
        return None, None
    max_sublist_trap_edges = max(trap_edges, key=lambda x: x[2])
    max_sublist_trap_edges[2] = p.trap_decay_rate
    trap_edges = [max_sublist_trap_edges]

    return np.array(emitter_edges), np.array(trap_edges)
