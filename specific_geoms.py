
# type: ignore
import numpy as np
import core.geometry as geom
from configs.parameters import Sim_params
from core.hamiltonian import Hamiltonian
from analysis import analyze_dark_light_dynamics
import analysis as analysis
import core.simulator as simulator
from core.simulator import (SimulationData, initial_excited_state,
                            simulate_time_evolution, compute_steady_state,
                            set_initial_state)
import qutip as qutip
import matplotlib.pyplot as plt
import visualisation.plotter as plotter
import generator as gen
from configs.system_functions import SystemFunctionsData


def two_emitter_with_trap():

    d_real = 2.0
    tlist = np.linspace(0, 4000, 1000)
    positions = {
        1: (0.0, 0.0),
        2: (d_real / 2.0, 0.0),
        3: (d_real, 0.0)}

    changing_env_params = []

    photon_scaling = 1e2
    phonon_scaling = 1
    omega = 2.0

    params1 = Sim_params(
        num_emitters=2,
        trap_omega=1.0,
        emitter_omega = omega,
        d_real=d_real,
        using_traps=True,
        using_photons=False,
        using_phonons=False,
        using_L_decay_rate=True,
        temperatures={'photon': 6000, 'phonon': 300},
        spectral_photon_scaling=0.0000014,
        spectral_phonon_scaling=0.025
    )
    changing_env_params.append(params1)

    params2 = Sim_params(
        num_emitters=2,
        trap_omega=1.0,
        emitter_omega = omega,
        d_real=d_real,
        using_traps=True,
        using_photons=True,
        using_phonons=False,
        using_L_decay_rate=True,
        temperatures={'photon': 6000, 'phonon': 300},
        spectral_photon_scaling=0.0000014 * photon_scaling,
        spectral_phonon_scaling=0.025 * phonon_scaling
    )
    changing_env_params.append(params2)

    params3 = Sim_params(
        num_emitters=2,
        trap_omega=1.0,
        emitter_omega = omega,
        d_real=d_real,
        using_traps=True,
        using_photons=True,
        using_phonons=True,
        using_L_decay_rate=True,
        temperatures={'photon': 6000, 'phonon': 300},
        spectral_photon_scaling=0.0000014 * photon_scaling,
        spectral_phonon_scaling=0.025 * phonon_scaling
    )
    changing_env_params.append(params3)

    sim_data_list = []
    dark_state_dynamics_list = []

    for i in range(3):
        params = changing_env_params[i]
        print(f'PARAMS: {params.using_photons}')

        geometry = geom.make_geometry_from_positions(
            p=params, positions=positions)

        # === === SIMULATE SPECIFIC GEOM === ===
        hamiltonian = Hamiltonian.build(params, geometry)
        sim_data = SimulationData(hamiltonian=hamiltonian)
        sim_data = set_initial_state(
            sim_data,
            initial_excited_state(sim_data.num_levels, index=1)
        )
        sim_data = simulator.simulate_time_evolution(sim_data, tlist)
        # Compute steady state
        sim_data = simulator.compute_steady_state(sim_data)
        # Compute decay rate
        sim_data = simulator.compute_decay_rate(sim_data)
        sim_data_list.append(sim_data)

        # Analysis
        final_trap_pop_steady = simulator.get_final_trap_population_steady_state(
            sim_data)
        final_trap_pop_t = simulator.get_final_trap_population_time_evolution(
            sim_data)
        decay_rate = simulator.get_decay_rate(sim_data)

        # Analyze dark/light state dynamics between two emitters
        state1 = 1
        state2 = 2
        dark_state_dynamics_list.append(
            analysis.analyze_dark_light_dynamics(params, sim_data, state1, state2))

        # Print results
        print("\n=== System Geometry ===")
        print(f"Emitter positions (nm): {geometry.positions}")
        print("\nEmitter couplings (eV):")
        for edge in geometry.emitter_edges_n_coupling:
            print(f"  {int(edge[0])} <-> {int(edge[1])}: {edge[2]:.8f} eV")

        print("\nTrap connections (eV):")
        for edge in geometry.trap_edges_n_coupling:
            print(f"  Emitter {int(edge[0])} -> Trap: {edge[2]:.8f} eV")

        print("\n=== Simulation Results ===")
        print(f"Final trap population (steady state): {final_trap_pop_steady}")
        print(f"Final trap population (T = {tlist[-1]}): {final_trap_pop_t}")
        print(f"Decay rate: {decay_rate}")

        # Plot dynamics
        # geom_fig = plotter.plot_geometry(sim_data, geometry)
        # dark_light_fig = plotter.plot_dark_light_dynamics(sim_data, dark_state_dynamics, state1, state2, title='Dark/Light State Dynamics Analysis')

    labels = ['No environment', 'Photons', 'Photons + Phonons']
    colors = ['dimgray', 'red', 'blue']
    # colors = ['dimgray', 'gray', 'darkgray']
    # linestyles = [':', '-', '--']
    linestyles = ['-', 'dashdot', '--']
    pop_fig = plotter.plot_population_subplots(sim_data_list, labels=labels,
                                               colors=colors, linestyles=linestyles, fontsize=24)

    linestyles = [['-', '--'], ['-', '--'], ['-', '--']]
    colors = [['red', 'orange'], [
        'blue', '#003d57'], ['black', 'gray']]

    dark_light_fig = plotter.plot_dark_light_dynamics(sim_data_list,
                                                      dark_state_dynamics_list, state1, state2, title='Dark/Light State Dynamics',
                                                      labels=labels, linestyles=linestyles, colors=colors, fontsize=24, dpi=120)

    plt.show()


def two_emitter_with_trap_varying_onsite():

    d_real = 2.0
    tlist = np.linspace(0, 4000, 1000)
    positions = {
        1: (0.0, 0.0),
        2: (d_real / 2.0, 0.0),
        3: (d_real, 0.0)}

    changing_env_params = []

    photon_scaling = 1
    phonon_scaling = 1

    params1 = Sim_params(
        num_emitters=2,
        trap_omega=1.5,
        d_real=d_real,
        using_traps=True,
        using_photons=False,
        using_phonons=False,
        using_L_decay_rate=True,
        temperatures={'photon': 6000, 'phonon': 300},
        spectral_photon_scaling=0.0000014,
        spectral_phonon_scaling=0.025
    )
    changing_env_params.append(params1)

    params2 = Sim_params(
        num_emitters=2,
        trap_omega=1.5,
        d_real=d_real,
        using_traps=True,
        using_photons=True,
        using_phonons=False,
        using_L_decay_rate=True,
        temperatures={'photon': 6000, 'phonon': 300},
        spectral_photon_scaling=0.0000014 * photon_scaling,
        spectral_phonon_scaling=0.025 * phonon_scaling
    )
    changing_env_params.append(params2)

    params3 = Sim_params(
        num_emitters=2,
        trap_omega=1.5,
        d_real=d_real,
        using_traps=True,
        using_photons=True,
        using_phonons=True,
        using_L_decay_rate=True,
        temperatures={'photon': 6000, 'phonon': 300},
        spectral_photon_scaling=0.0000014 * photon_scaling,
        spectral_phonon_scaling=0.025 * phonon_scaling
    )
    changing_env_params.append(params3)

    for i in range(3):
        params = changing_env_params[i]
        geometry = geom.make_geometry_from_positions(
            p=params, positions=positions)

        # === === FIND OPTIMAL OMEGA FOR GIVEN GEOM === ===
        results, best_omegas, best_eff = gen.optimize_emitter_omegas(
            params=params,
            geometry=geometry,
            omega_ranges=[[0.5, 2.5, 60], [0.5, 2.5, 60]],
            metric_type='steady_state',
            t=4000
        )

        print(f"Optimal omegas: {best_omegas}")
        print(f"Best efficiency: {best_eff}")

        # Plot results
        omegas = [r[0] for r in results]
        effs = [r[1] for r in results]

        optimization_fig = plotter.plot_omega_optimization_2emitters(
            results_data=(results, best_omegas, best_eff),
            params=params,
            opt_point=(best_omegas[0], best_omegas[1], best_eff),
            title='Emitter Frequency Optimization'
        )

        plt.show()


def gen_optimal_geom():
    d_real = 3.0
    tlist = np.linspace(0, 40000, 1000)
    positions = {
        1: (0.0, 0.0),
        2: (d_real / d_real, 0.0),
        3: (2 * d_real / d_real, 0.0),
        4: (d_real, 0.0)}

    params = Sim_params(
        num_emitters=3,
        trap_omega=1.0,
        d_real=d_real,
        using_traps=True,
        using_photons=True,
        using_phonons=True,
        using_L_decay_rate=True,
        temperatures={'photon': 6000, 'phonon': 300},
        spectral_photon_scaling=0.0000014,
        spectral_phonon_scaling=0.025
    )

    optimal_pos, max_eff, eff_map, optimal_geom, optimal_sim_data, optimal_dark_pop, optimal_dark_pos, dark_pop_map = gen.optimize_full_grid_geometry(
        params=params,
        x_bounds=(0.05, params.d_real - 1.05),
        y_bounds=(-2.0, 2.0),
        grid_density=(60, 60),
        metric_type='steady_state',
        calc_dark_pop = True
    )

    print(f'All efficiencies are: {eff_map}')
    print()
    print(f'Optimal positions are: {optimal_pos}')
    print()
    print(f'Optimal efficiency is: {max_eff}')
    print()

    geom_fig = plotter.plot_geometry(optimal_sim_data, optimal_geom)
    fig = plotter.plot_dark_population_optimization(dark_pop_map, params)
    fig = plotter.plot_efficiency_optimization_for_varying_one_emitter(
        eff_map=eff_map,
        params=params
    )

    plt.show()


def gen_optimal_geom_genetic():
    d_real = 7.0

    params = Sim_params(
        num_emitters=7,
        trap_omega=1.0,
        d_real=d_real,
        using_traps=True,
        using_photons=True,
        using_phonons=True,
        using_L_decay_rate=True,
        temperatures={'photon': 6000, 'phonon': 300},
        spectral_photon_scaling=0.0000014,
        spectral_phonon_scaling=0.025
    )

    # GA-optimized version instead of grid search
    optimal_pos, max_eff, _, optimal_geom, optimal_sim_data, optimal_dark_pop, _, _ = gen.optimize_ga_geometry(
        params=params,
        x_bounds=(0.05, params.d_real - 1.05),
        y_bounds=(-2.0, 2.0),
        population_size=100,  # Increased from default
        generations=100,       # Increased from default
        cxpb=0.8,            # Crossover probability
        mutpb=0.15,           # Mutation probability
        metric_type='decay_rate'
    )

    print(f'Optimal positions are: {optimal_pos}')
    print(f'Optimal efficiency is: {max_eff}')
    # print(f'Optimal dark population: {optimal_dark_pop:.4f}' if optimal_dark_pop is not None else '')

    # Visualization (same as before)
    geom_fig = plotter.plot_geometry(optimal_sim_data, optimal_geom)

    # For time evolution plots if needed:
    # if optimal_sim_data.time_evolution:
    #     tlist = np.linspace(0, 4000, 1000)
    #     evolution_fig = plotter.plot_time_evolution(optimal_sim_data, tlist)

    plt.show()

def simulate_and_plot_geometry(positions_dict, d_real=2.0, omega=2.0):
    """Simulate and plot geometry based on given emitter positions.

    Args:
        positions_dict: Dictionary of emitter positions {id: (x,y)}
        d_real: Real distance scaling (default 2.0)
        omega: Emitter frequency (default 1.0)
    """
    params = Sim_params(
        num_emitters=len(positions_dict)-1,  # Assuming last position is trap
        trap_omega=1.0,
        emitter_omega=omega,
        d_real=d_real,
        using_traps=True,
        using_photons=True,
        using_phonons=True,
        using_L_decay_rate=True,
        temperatures={'photon': 6000, 'phonon': 300},
        spectral_photon_scaling=0.0000014,
        spectral_phonon_scaling=0.025
    )

    geometry = geom.make_geometry_from_positions(
        p=params, positions=positions_dict)

    hamiltonian = Hamiltonian.build(params, geometry)
    sim_data = SimulationData(hamiltonian=hamiltonian)

    geom_fig = plotter.plot_geometry(sim_data, geometry)
    plt.show()

def gen_optimal_omega_genetic():
    d_real = 8.0

    params = Sim_params(
        num_emitters=8,
        trap_omega=1.0,
        d_real=d_real,
        using_traps=True,
        using_photons=True,
        using_phonons=True,
        using_L_decay_rate=True,
        temperatures={'photon': 6000, 'phonon': 300},
        spectral_photon_scaling=0.0000014,
        spectral_phonon_scaling=0.025
    )
    positions = {
        1: (0.0, 0.0),
        2: (5.219, -0.266),
        3: (6.242, -0.069),
        4: (2.197, -0.234),
        5: (3.471, 0.508),
        6: (4.637, -1.069),
        7: (4.693, 1.135),
        8: (d_real - 1, 0.0),
        9: (d_real, 0.0)}

    geometry = geom.make_geometry_from_positions(p=params, positions=positions)

    omega_ranges = [
        (0.1, 3.0),  # Emitter 1 THE THIRD VAL IN THIS TUPLE IS IGNORED AS IS
        # NOT NEEDED. REMOVE THIS LATER
        (0.1, 3.0),
        (0.1, 3.0),
        (0.1, 3.0),
        (0.1, 3.0),
        (0.1, 3.0),
        (0.1, 3.0),
        (0.1, 3.0)
    ]

    # Optimal omegas are: [1.72049694 2.09504199 1.99402508]
    # Optimal efficiency is: 0.18656578859577572

    best_omegas, best_eff, logbook = gen.optimize_emitter_omegas_ga(
        params=params,
        geometry=geometry,
        omega_ranges=omega_ranges,
        metric_type='decay_rate',
        population_size=200,
        generations=200
    )
    print(f'logbook: {logbook}')
    print(f'Optimal omegas are: {best_omegas}')
    print(f'Optimal efficiency is: {best_eff}')
    # print(f'Optimal dark population: {optimal_dark_pop:.4f}' if optimal_dark_pop is not None else '')

    plt.show()


def gen_many_linear_chains():

    d_real = 3.0
    params = Sim_params(
        num_emitters=4,
        trap_omega=1.0,
        d_real=d_real,
        using_traps=True,
        using_photons=True,
        using_phonons=True,
        using_L_decay_rate=True,
        temperatures={'photon': 6000, 'phonon': 300},
        spectral_photon_scaling=0.0000014,
        spectral_phonon_scaling=0.025
    )

    gen.gen_linear_chains(params = params, num_emitters_range = (3, 8), metric_type = "decay_rate")


#gen_many_linear_chains()
#two_emitter_with_trap()
#two_emitter_with_trap_varying_onsite()
#gen_optimal_geom()
#gen_optimal_omega_genetic()
#gen_optimal_geom_genetic()

positions_3 = {
    1: (0.0, 0.0),
    2: (1.338, 0.000),
    3: (2.0, 0.0),
    4: (3.0, 0.0)
}

positions_4 = {
    1: (0.0, 0.0),
    2: (2.519, -0.567),
    3: (2.398, 0.302),
    4: (3.0, 0.0),
    5: (4.0, 0.0)
}

positions_5 = {
    1: (0.0, 0.0),
    2: (3.482, 0.319),
    3: (2.983, -0.505),
    4: (3.614, -0.797),
    5: (4.0, 0.0),
    6: (5.0, 0.0)
}

positions_6 = {
    1: (0.0, 0.0),
    2: (2.796, -0.190),
    3: (0.764, -0.016),
    4: (3.545, -0.165),
    5: (4.394, -0.024),
    6: (5.0, 0.0),
    7: (6.0, 0.0)
}

positions_7 = {
    1: (0.0, 0.0),
    2: (4.643, -0.036),
    3: (0.783, 0.007),
    4: (4.474, 0.585),
    5: (5.449, -0.037),
    6: (3.918, 0.112),
    7: (6.0, 0.0),
    8: (7.0, 0.0)
}

positions_8 = {
    1: (0.0, 0.0),
    2: (5.637, 0.399),
    3: (5.008, 0.752),
    4: (5.620, 1.162),
    5: (6.404, -0.105),
    6: (1.201, 0.750),
    7: (0.856, -1.975),
    8: (7.0, 0.0),
    9: (8.0, 0.0)
}

positions = positions_8
simulate_and_plot_geometry(positions_dict = positions ,d_real = len(positions) - 1)
