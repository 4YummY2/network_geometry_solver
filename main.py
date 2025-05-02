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

def main():


    # === === SIM EMITTER GEOM === ===

    d_real = 5.0
    #tlist = np.linspace(0, 4000, 1000)
    # positions = {
    #             1: (0.0, 0.0),
    #             2: (0.4, 0.0),
    #             3: (2, 0.0),
    #             4: (d_real, 0.0)}

    positions = {
                1: (0.0, 0.0),
                2: (1.0147227830089178, 1.2428181942878358),
                3: (1.0607651727307772, -0.48564518802312673),
                4: (3.9351399111171235, -0.00020878870811574145),
                5: (d_real - 1.0, 0.0),
                6: (d_real, 0.0)}

    params = Sim_params(
        num_emitters=5,
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

    # params.emitter_omega_vals = np.array([0] + [2.0, 2.5])

    # params = Sim_params(
    #     num_emitters=4,
    #     d_real=4.0,              # (system size)
    #     using_traps=True,
    #     using_photons=False,
    #     using_phonons=False,
    #     using_L_decay_rate=True,
    #     temperatures = {'photon' : 600000, 'phonon': 300}
    # )


    # params.emitter_omega_vals = np.array([0] + [2.0, 2.0, 3.0, 2.0])




    # # Create symmetric ring geometry with y-offsets
    # y_params = {'y1': 1.36}  # Vertical displacements (nm)
    # geometry = geom.make_symm_ring_geom(
    #     p=params,
    #     y_params=y_params
    # )

    geometry = geom.make_geometry_from_positions(p = params, positions = positions)

    # results, best_omegas, best_eff = gen.optimize_emitter_omegas(
    #     params=params,
    #     geometry=geometry,
    #     min_omega=1.5,
    #     max_omega=3.0,
    #     num_points=5,
    #     metric_type='steady_state'
    # )


    # === === SIMULATE SPECIFIC GEOM === ===
    #
    # Build Hamiltonian
    hamiltonian = Hamiltonian.build(params, geometry)

    # Initialize simulation
    sim_data = SimulationData(hamiltonian=hamiltonian)

    # Set initial state (excite first emitter)
    sim_data = set_initial_state(
        sim_data,
        initial_excited_state(sim_data.num_levels, index=1)
    )

    # Run time evolution
    tlist = np.linspace(0, 400, 10000)
    # sim_data = simulator.simulate_time_evolution(sim_data, tlist)

    # Compute steady state
    # sim_data = simulator.compute_steady_state(sim_data)

    # Compute decay rate
    # sim_data = simulator.compute_decay_rate(sim_data)

    # Analysis
    # final_trap_pop_steady = simulator.get_final_trap_population_steady_state(sim_data)
    # final_trap_pop_t = simulator.get_final_trap_population_time_evolution(sim_data)
    # decay_rate = simulator.get_decay_rate(sim_data)

    # Analyze dark/light state dynamics between two emitters
    # state1 = 1
    # state2 = 2
    # #dark_state_dynamics = analysis.analyze_dark_light_dynamics(params, sim_data, state1, state2)

    # # Print results
    # print("\n=== System Geometry ===")
    # print(f"Emitter positions (nm): {geometry.positions}")
    # print("\nEmitter couplings (eV):")
    # for edge in geometry.emitter_edges_n_coupling:
    #     print(f"  {int(edge[0])} <-> {int(edge[1])}: {edge[2]:.8f} eV")

    # print("\nTrap connections (eV):")
    # for edge in geometry.trap_edges_n_coupling:
    #     print(f"  Emitter {int(edge[0])} -> Trap: {edge[2]:.8f} eV")

    # print("\n=== Simulation Results ===")
    # print(f"Final trap population (steady state): {final_trap_pop_steady}")
    # print(f"Final trap population (T = {tlist[-1]}): {final_trap_pop_t}")
    # print(f"Decay rate: {decay_rate}")

    # # Plot dynamics
    # pop_fig = plotter.plot_population_subplots([sim_data])
    # geom_fig = plotter.plot_geometry(sim_data, geometry)
    #dark_light_fig = plotter.plot_dark_light_dynamics([sim_data], [dark_state_dynamics], state1, state2, title='Dark/Light State Dynamics Analysis', colors = [('black', 'gray')])





    # === === FIND OPTIMAL 4 EMITTER RING GEOMETRY === ===

    # opt_results = gen.optimize_4emitter_ring_y_spacing(
    #     params=params,
    #     y_spacing_min=0.2,
    #     y_spacing_max=1.8,
    #     num_points=100,
    #     metric_type='time_evolution',
    #     t = 100,
    #     dark_light=True,
    #     emitter1 = 2,
    #     emitter2 = 3
    # )


    # # # Plot optimization results
    # optimization_plot = plotter.plot_4emitter_optimization(opt_results, params)

    # # # Access numerical results
    # y_spacings, efficiencies, opt_y, opt_eff, dark_light_pops = opt_results
    # dark_pops = dark_light_pops[0]
    # light_pops = dark_light_pops[1]

    # print(f'dark pops : {dark_pops}')

    # print(f"\nOptimal y-spacing: {opt_y:.2f} nm")
    # print(f"Maximum efficiency: {opt_eff:.3f}")


    #=== === FIND OPTIMAL OMEGA FOR GIVEN GEOM === ===
    # results, best_omegas, best_eff = gen.optimize_emitter_omegas(
    #     params=params,
    #     geometry=geometry,
    #     omega_ranges = [[1.0, 2.0, 30], [1.0, 2.0, 30]],
    #     metric_type='decay_rate',
    #     t = 4000
    # )

    # print(f"Optimal omegas: {best_omegas}")
    # print(f"Best efficiency: {best_eff}")


    # Plot results
    # omegas = [r[0] for r in results]
    # effs = [r[1] for r in results]


    # optimization_fig = plotter.plot_omega_optimization_2emitters(
    #     results_data=(results, best_omegas, best_eff),
    #     params=params,
    #     opt_point=(best_omegas[0], best_omegas[1], best_eff),
    #     title='Emitter Frequency Optimization'
    # )


    # === === FIND OPTIMAL 6 EMITTER RING GEOMETRY === ===

    # params_6em = Sim_params(
    #     num_emitters=6,
    #     d_real=5.0,
    #     using_traps=True,
    #     using_photons=True,
    #     using_phonons=True,
    #     using_L_decay_rate = True
    # )

    # # Run optimization
    # results = gen.optimize_6emitter_ring_yspacings(
    #     params=params_6em,
    #     y1_bounds=(0.6, 2.2),
    #     y2_bounds=(0.6, 2.2),
    #     num_points= 30,
    #     metric_type="decay_rate"
    # )

    # # Unpack results
    # y1_grid, y2_grid, pop_grid, opt_y1, opt_y2, max_pop = results

    # fig = plotter.plot_6emitter_optimization(results, params_6em)



    # === === PLOT SYSTEM FUNCTIONS === ===

    # sim_params = Sim_params(
    #     num_emitters=4,
    #     d_real=4.0,
    #     using_traps=True,
    #     using_photons=True,
    #     using_phonons=True,
    #     using_L_decay_rate = True
    # )

    # optimal_pos, max_eff, eff_map = gen.optimize_full_grid_geometry(
    #     params=sim_params,
    #     x_bounds=(0.25, sim_params.d_real - 0.25),
    #     y_bounds=(-1.5, 1.5),
    #     grid_density=(10, 10),
    #     metric_type='decay_rate'
    # )

    # print(f'All efficiencies are: {eff_map}')
    # print()
    # print(f'Optimal positions are: {optimal_pos}')
    # print()
    # print(f'Optimal efficiency is: {max_eff}')
    # print()


    # === === PLOT SYSTEM FUNCTIONS === ===
    #

    phonon_data = SystemFunctionsData(params, 'phonon')
    fig_phonon = plotter.plot_system_functions(phonon_data, dpi=180)


    # dark_light_fig = plotter.plot_dark_light_optimization(
    #     y_spacings=y_spacings,
    #     dark_pops=dark_pops,
    #     light_pops=light_pops,
    #     params=params,
    #     opt_y=opt_y,
    #     title='Dark/Light State Population Optimization'
    # )


    plt.show()



if __name__ == "__main__":
    main()
