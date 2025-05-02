import numpy as np
import random
from deap import algorithms, base, creator, tools
import math as math
import copy
import itertools
from core.geometry import make_symm_ring_geom, make_geometry_from_positions
from core.hamiltonian import Hamiltonian
from core.simulator import (SimulationData, initial_excited_state,
                          compute_steady_state, set_initial_state,
                          get_final_trap_population_steady_state,
                          simulate_time_evolution, get_final_trap_population_time_evolution,
                          compute_decay_rate, get_decay_rate, compute_dark_light_pop)
from configs.parameters import Sim_params
from core.geometry import Geometry
import analysis as analysis


def optimize_emitter_omegas(
    params: Sim_params,
    geometry: Geometry,
    omega_ranges: list[tuple[float, float, int]],
    metric_type: str = 'steady_state',
    t: float = None
) -> tuple:
    """
    Optimize emitter omega values by testing all combinations within specified ranges

    Args:
        params: Simulation parameters
        geometry: Fixed geometry to use
        omega_ranges: List of tuples specifying (min, max, num_points) for each emitter
        metric_type: Efficiency metric ('steady_state', 'time_evolution', 'decay_rate')
        t: Time parameter for time evolution metric (required if metric_type='time_evolution')

    Returns:
        tuple: (all_results, best_omegas, best_efficiency)
        - all_results: List of tuples (omega_values, efficiency)
        - best_omegas: Optimal omega values array (excluding ground state)
        - best_efficiency: Highest efficiency found
    """
    import copy
    from itertools import product

    # Validate metric type requirements
    if metric_type == 'time_evolution' and t is None:
        raise ValueError("Time t must be provided for time_evolution metric")

    # Validate omega ranges match emitter count
    num_emitters = params.num_emitters
    if len(omega_ranges) != num_emitters:
        raise ValueError(f"Must provide ranges for {num_emitters} emitters")

    # Generate omega values and combinations
    omega_values = [np.linspace(min_o, max_o, num_o) for (min_o, max_o, num_o) in omega_ranges]
    combinations = product(*omega_values)

    # Calculate total combinations
    total = 1
    for (_, _, num_o) in omega_ranges:
        total *= num_o

    best_efficiency = 0
    best_omegas = None
    efficiency = None
    results = []

    for i, combination in enumerate(combinations):
        # Create parameter copy with current omegas
        new_params = copy.deepcopy(params)
        new_params.emitter_omega_vals = np.array([new_params.ground_omega] + list(combination))
        new_params.dimension = len(new_params.emitter_omega_vals) + len(new_params.trap_omega_vals)

        try:
            # Build Hamiltonian and simulate
            hamiltonian = Hamiltonian.build(new_params, geometry)
            sim_data = SimulationData(hamiltonian=hamiltonian)

            # Set initial state (excite first emitter)
            sim_data = set_initial_state(
                sim_data,
                initial_excited_state(sim_data.num_levels, index=1)
            )

            # Calculate efficiency
            if metric_type == 'steady_state':
                sim_data = compute_steady_state(sim_data)
                efficiency = get_final_trap_population_steady_state(sim_data)
            elif metric_type == 'time_evolution':
                tlist = np.linspace(0, t, 10)
                sim_data = simulate_time_evolution(sim_data, tlist)
                efficiency = get_final_trap_population_time_evolution(sim_data)
            elif metric_type == 'decay_rate':
                sim_data = compute_decay_rate(sim_data)
                efficiency = get_decay_rate(sim_data)
            else:
                raise ValueError(f"Invalid metric_type: {metric_type}")

            # Track results
            results.append((combination, efficiency))

            if efficiency >= best_efficiency:
                best_efficiency = efficiency
                best_omegas = combination

        except Exception as e:
            print(f"Failed for combination {combination}: {str(e)}")
            results.append((combination, 0))

        print(f"Processing {i+1}/{total}: {combination}")
        print(f'Eff: {efficiency}')
        print()

    return results, best_omegas, best_efficiency

def optimize_4emitter_ring_y_spacing(
    params: Sim_params,
    y_spacing_min: float = 0.2,
    y_spacing_max: float = 1.8,
    num_points: int = 100,
    metric_type: str = 'steady_state',
    t: float = None,
    dark_light = False,
    emitter1 = None,
    emitter2 = None
) -> tuple:
    """
    Optimize y-spacing for 4-emitter ring geometry to maximize efficiency metric

    Args:
        params: Simulation parameters object
        y_spacing_min: Minimum y-spacing (relative to x_spacing)
        y_spacing_max: Maximum y-spacing (relative to x_spacing)
        num_points: Number of points to sample
        metric_type: Type of efficiency metric ('steady_state', 'time_evolution', 'decay_rate')
        t: Time for time evolution metric (required if metric_type='time_evolution')

    Returns:
        tuple: (y_spacing_values, efficiency_values, optimal_y, optimal_efficiency)
    """
    # Validate emitter count and metric type
    if params.num_emitters != 4:
        raise ValueError("This optimization is specific to 4-emitter ring systems")

    if metric_type not in ['steady_state', 'time_evolution', 'decay_rate']:
        raise ValueError("Invalid metric_type. Choose 'steady_state', 'time_evolution' or 'decay_rate'")

    if metric_type == 'time_evolution' and t is None:
        raise ValueError("Time t must be provided for time_evolution metric")
    if (dark_light and (emitter1 is None or emitter2 is None)):
        raise ValueError("If using dark light steady states then please supply the args of emitter1 and emitter2")

    # Generate spacing values
    y_spacing_vals = np.linspace(y_spacing_min, y_spacing_max, num_points)
    efficiencies = []
    dark_steady_pops = []
    light_steady_pops = []
    optimal = (0, 0)  # (y_spacing, efficiency)

    current_sim_num = 0

    # Main optimization loop
    for y_spacing in y_spacing_vals:
        current_sim_num += 1

        print(f"{current_sim_num}/{num_points}")

        # Create geometry with current y-spacing
        geometry = make_symm_ring_geom(
            p=params,
            y_params={'y1': y_spacing}
        )

        # Build Hamiltonian and initialize simulation
        hamiltonian = Hamiltonian.build(params, geometry)
        sim_data = SimulationData(hamiltonian=hamiltonian)

        # Set initial state (excite first emitter)
        sim_data = set_initial_state(
            sim_data,
            initial_excited_state(sim_data.num_levels, index=1)
        )

        # Calculate efficiency based on metric type
        if metric_type == 'steady_state':
            sim_data = compute_steady_state(sim_data)
            if dark_light:
                dark_light_pop = compute_dark_light_pop(params, sim_data,
                    rho_states = [sim_data.results_steady_state], emitter1 = emitter1,
                emitter2 = emitter2)

                dark_pop = dark_light_pop["dark_pop_steady"]
                light_pop = dark_light_pop["light_pop_steady"]


            efficiency = get_final_trap_population_steady_state(sim_data)
        elif metric_type == 'time_evolution':
            tlist = np.linspace(0, t, 20)
            sim_data = simulate_time_evolution(sim_data, tlist)
            efficiency = get_final_trap_population_time_evolution(sim_data)

            if dark_light:
                dark_light_pop = compute_dark_light_pop(params, sim_data,
                rho_states = sim_data.results_time_sim, emitter1 = emitter1,
                emitter2 = emitter2)

                dark_pop = dark_light_pop["dark_pop_steady"]
                light_pop = dark_light_pop["light_pop_steady"]

        elif metric_type == 'decay_rate':
            if params.using_L_decay_rate == False:
                raise Exception("If using metric = {metric_type} then please set using_L_decay_rate = True in params")
            sim_data = compute_decay_rate(sim_data)
            decay_rate = get_decay_rate(sim_data)
            efficiency = decay_rate

        efficiencies.append(efficiency)

        if dark_light:
            dark_steady_pops.append(dark_pop)
            light_steady_pops.append(light_pop)

        # Update optimal
        if efficiency > optimal[1]:
            optimal = (y_spacing, efficiency)

    return y_spacing_vals, np.array(efficiencies), optimal[0], optimal[1], [dark_steady_pops, light_steady_pops]


def optimize_6emitter_ring_yspacings(
    params: Sim_params,
    y1_bounds: tuple = (0.2, 1.8),
    y2_bounds: tuple = (0.2, 1.8),
    num_points: int = 20,
    metric_type: str = 'steady_state',
    t: float = None
) -> tuple:
    """
    Optimize both y1 and y2 spacings for 6-emitter ring geometry

    Args:
        params: Simulation parameters object
        y1_bounds: (min, max) for y1 spacing (relative to x_spacing)
        y2_bounds: (min, max) for y2 spacing (relative to x_spacing)
        num_points: Number of points per dimension (total runs = num_points²)
        metric_type: Type of efficiency metric ('steady_state' or 'time_evolution')
        t: Time for time evolution metric (required if metric_type='time_evolution')

    Returns:
        tuple: (y1_grid, y2_grid, population_grid, optimal_y1, optimal_y2, max_population)
    """
    # Validate emitter count
    if params.num_emitters != 6:
        raise ValueError("This optimization is specific to 6-emitter ring systems")

    if metric_type not in ['steady_state', 'time_evolution', 'decay_rate']:
        raise ValueError("Invalid metric_type. Choose 'steady_state' or 'time_evolution'")

    if metric_type == 'time_evolution' and t is None:
        raise ValueError("Time t must be provided for time_evolution metric")

    # Calculate x_spacing based on system geometry
    x_spacing = params.d_real / ((params.num_emitters // 2) + 1)

    # Generate spacing grids
    y1_vals = np.linspace(y1_bounds[0],
                         y1_bounds[1],
                         num_points)
    y2_vals = np.linspace(y2_bounds[0],
                         y2_bounds[1],
                         num_points)

    y1_grid, y2_grid = np.meshgrid(y1_vals, y2_vals)
    eff_grid = np.zeros_like(y1_grid)

    highest_eff = 0
    optimal_y1, optimal_y2 = 0, 0

    num_of_geoms = num_points ** 2
    current_geoms = 0

    # Main optimization loop
    for i in range(num_points):
        for j in range(num_points):
            y1 = y1_vals[i]
            y2 = y2_vals[j]

            # Create geometry with current spacings
            geometry = make_symm_ring_geom(
                p=params,
                y_params={'y1': y1, 'y2': y2}
            )

            # Build Hamiltonian and initialize simulation
            hamiltonian = Hamiltonian.build(params, geometry)
            sim_data = SimulationData(hamiltonian=hamiltonian)

            # Set initial state (excite first emitter)
            sim_data = set_initial_state(
                sim_data,
                initial_excited_state(sim_data.num_levels, index=1)
            )

            if metric_type == 'steady_state':
                # Compute steady state
                sim_data = compute_steady_state(sim_data)
                efficiency = get_final_trap_population_steady_state(sim_data)
            elif metric_type == 'time_evolution':
                # Run time evolution
                tlist = np.linspace(0, t, 4)
                sim_data = simulate_time_evolution(sim_data, tlist)
                efficiency = get_final_trap_population_time_evolution(sim_data)

            elif metric_type == 'decay_rate':
                if params.using_L_decay_rate == False:
                    raise Exception("If using metric = {metric_type} then please set using_L_decay_rate = True in params")
                sim_data = compute_decay_rate(sim_data)
                decay_rate = get_decay_rate(sim_data)
                efficiency = decay_rate

            else:
                raise Exception("Please specify a valid metric")

            eff_grid[j, i] = efficiency  # Note j,i for grid alignment

            current_geoms += 1
            print(f"{current_geoms} / {num_of_geoms}")

            # Update optimal
            if efficiency > highest_eff:
                highest_eff = efficiency
                optimal_y1, optimal_y2 = y1, y2

    return y1_grid, y2_grid, eff_grid, optimal_y1, optimal_y2, highest_eff



def optimize_full_grid_geometry(
    params: Sim_params,
    x_bounds: tuple = (0.0, 3.0),
    y_bounds: tuple = (-1.0, 1.5),
    grid_density: tuple = (3, 3),
    metric_type: str = 'steady_state',
    t: float = None,
    calc_dark_pop: bool = False
) -> tuple:
    """
    Optimizes emitter positions on a 2D grid to maximize trap population

    Args:
        params: Simulation parameters object
        x_bounds: (min, max) for x coordinates
        y_bounds: (min, max) for y coordinates
        grid_density: (num_x_points, num_y_points) for grid resolution
        metric_type: Efficiency metric type ('steady_state', 'time_evolution', 'decay_rate')
        t: Time parameter for time evolution metric (required if metric_type='time_evolution')
        calc_dark_pop: Whether to calculate dark state average time evolution (only for steady_state)

    Returns:
        tuple: (optimal_positions, max_efficiency, efficiency_map, optimal_geometry,
                optimal_sim_data, optimal_dark_pop, optimal_dark_pos, dark_pop_map)
    """
    # Validate input
    if params.num_emitters < 3:
        raise ValueError("Number of emitters must be at least 3")
    if metric_type not in ['steady_state', 'time_evolution', 'decay_rate']:
        raise ValueError("Invalid metric_type")
    if metric_type == 'time_evolution' and t is None:
        raise ValueError("Time t must be provided for time_evolution metric")
    if params.d_real is None:
        raise ValueError("d_real must be specified in parameters")
    if calc_dark_pop and metric_type != 'steady_state':
        raise ValueError("Dark population calculation only available for steady_state metric")

    # Generate grid points and validate combinations
    x_points = np.linspace(x_bounds[0], x_bounds[1], grid_density[0])
    y_points = np.linspace(y_bounds[0], y_bounds[1], grid_density[1])
    grid = list(itertools.product(x_points, y_points))

    fixed_emitter1_pos = (0.0, 0.0)
    fixed_emitter2_pos = (params.d_real - 1, 0.0)
    trap_pos = (params.d_real, 0.0)

    filtered_grid = [p for p in grid if p not in {fixed_emitter1_pos, fixed_emitter2_pos}]
    required_emitters = params.num_emitters - 2

    if len(filtered_grid) < required_emitters:
        raise ValueError(f"Not enough unique grid positions ({len(filtered_grid)}) "
                         f"for {required_emitters} variable emitters")

    combinations = itertools.combinations(filtered_grid, required_emitters)
    total_combinations = math.comb(len(filtered_grid), required_emitters)

    # Initialize tracking variables
    max_efficiency = 0.0
    optimal_dark_pop = 0.0 if calc_dark_pop else None
    optimal_positions = None
    optimal_geometry = None
    optimal_sim_data = None
    optimal_dark_pos = None
    efficiency_map = {}
    dark_pop_map = {} if calc_dark_pop else None

    def _process_combination(positions):
        """Process one position combination and return metrics"""
        nonlocal params, metric_type, t, calc_dark_pop

        pos_dict = {
            1: fixed_emitter1_pos,
            params.num_emitters: fixed_emitter2_pos,
            params.num_emitters + 1: trap_pos
        }
        for i, coord in enumerate(positions, 2):
            pos_dict[i] = coord

        geometry = make_geometry_from_positions(params, pos_dict)
        efficiency = 0.0
        dark_pop = 0.0 if calc_dark_pop else None
        sim_data = None

        try:
            hamiltonian = Hamiltonian.build(params, geometry)

            # Create base simulation data with initial state
            sim_data = SimulationData(hamiltonian=hamiltonian)
            sim_data = set_initial_state(
                sim_data,
                initial_excited_state(sim_data.num_levels, index=1)
            )

            if metric_type == 'steady_state':
                # Compute steady state efficiency
                sim_data = compute_steady_state(sim_data)
                efficiency = get_final_trap_population_steady_state(sim_data)

                if calc_dark_pop:
                    # Create fresh simulation data for dark population analysis
                    sim_data_dark = SimulationData(hamiltonian=hamiltonian)
                    sim_data_dark = set_initial_state(
                        sim_data_dark,
                        initial_excited_state(sim_data_dark.num_levels, index=1)
                    )

                    # Compute dark population dynamics
                    tlist_dark = np.linspace(0.0, 100000, 1000)
                    sim_data_dark = simulate_time_evolution(sim_data_dark, tlist=tlist_dark)
                    dark_light_results = analysis.analyze_dark_light_dynamics(
                        params, sim_data_dark, 2, 3, metric_type='time_evolution'
                    )
                    dark_pop = np.mean(dark_light_results['dark_pop_time'])

            elif metric_type == 'time_evolution':
                tlist = np.linspace(0, t, 100)
                sim_data = simulate_time_evolution(sim_data, tlist)
                efficiency = get_final_trap_population_time_evolution(sim_data)

            elif metric_type == 'decay_rate':
                if not params.using_L_decay_rate:
                    raise ValueError("Enable L_decay_rate for decay_rate metric")
                sim_data = compute_decay_rate(sim_data)
                efficiency = get_decay_rate(sim_data)

            return efficiency, dark_pop, geometry, sim_data

        except Exception as e:
            print(f"Error processing {positions}: {str(e)}")
            return (0.0, 0.0 if calc_dark_pop else None, None, None) if metric_type == 'steady_state' else (0.0, None, None, None)

    # Main optimization loop
    for current_combination, positions in enumerate(combinations, 1):
        # Process combination and get all metrics
        efficiency, dark_pop, geometry, sim_data = _process_combination(positions)

        # Store results in maps
        efficiency_map[positions] = efficiency
        if calc_dark_pop:
            dark_pop_map[positions] = dark_pop

        # Update optimal efficiency tracking
        if efficiency > max_efficiency and geometry is not None and sim_data is not None:
            max_efficiency = efficiency
            optimal_positions = positions
            optimal_geometry = geometry
            optimal_sim_data = sim_data

        # Update optimal dark population tracking
        if calc_dark_pop and dark_pop is not None and dark_pop > optimal_dark_pop:
            optimal_dark_pop = dark_pop
            optimal_dark_pos = positions

        print(f"Processed {current_combination}/{total_combinations} - "
              f"Efficiency: {efficiency}" + (f", Dark Pop: {dark_pop}" if calc_dark_pop else ""))

    return (
        optimal_positions,
        max_efficiency,
        efficiency_map,
        optimal_geometry,
        optimal_sim_data,
        optimal_dark_pop,
        optimal_dark_pos,
        dark_pop_map
    )




def optimize_ga_geometry(
    params: Sim_params,
    x_bounds: tuple = (0.0, 3.0),
    y_bounds: tuple = (-1.0, 1.5),
    population_size: int = 50,
    generations: int = 40,
    cxpb: float = 0.7,
    mutpb: float = 0.2,
    metric_type: str = 'steady_state',
    t: float = None,
    calc_dark_pop: bool = False
) -> tuple:
    """Optimize emitter positions using a genetic algorithm with strict bounds enforcement."""

    # Validate input
    if params.num_emitters < 3:
        raise ValueError("Number of emitters must be at least 3")
    if metric_type not in ['steady_state', 'time_evolution', 'decay_rate']:
        raise ValueError("Invalid metric_type")
    if metric_type == 'time_evolution' and t is None:
        raise ValueError("Time t must be provided for time_evolution metric")
    if params.d_real is None:
        raise ValueError("d_real must be specified in parameters")
    if calc_dark_pop and metric_type != 'steady_state':
        raise ValueError("Dark population calculation only available for steady_state metric")

    fixed_emitter1_pos = (0.0, 0.0)
    fixed_emitter2_pos = (params.d_real - 1, 0.0)
    trap_pos = (params.d_real, 0.0)

    def _process_combination(positions):
        """Process one position combination and return metrics"""
        nonlocal params, metric_type, t, calc_dark_pop

        pos_dict = {
            1: fixed_emitter1_pos,
            params.num_emitters: fixed_emitter2_pos,
            params.num_emitters + 1: trap_pos
        }
        for i, coord in enumerate(positions, 2):
            pos_dict[i] = coord

        geometry = make_geometry_from_positions(params, pos_dict)
        efficiency = 0.0
        dark_pop = 0.0 if calc_dark_pop else None
        sim_data = None

        try:
            hamiltonian = Hamiltonian.build(params, geometry)
            sim_data = SimulationData(hamiltonian=hamiltonian)
            sim_data = set_initial_state(
                sim_data,
                initial_excited_state(sim_data.num_levels, index=1)
            )

            if metric_type == 'steady_state':
                sim_data = compute_steady_state(sim_data)
                efficiency = float(get_final_trap_population_steady_state(sim_data))

                if calc_dark_pop:
                    sim_data_dark = SimulationData(hamiltonian=hamiltonian)
                    sim_data_dark = set_initial_state(
                        sim_data_dark,
                        initial_excited_state(sim_data_dark.num_levels, index=1))
                    tlist_dark = np.linspace(0.0, 100000, 1000)
                    sim_data_dark = simulate_time_evolution(sim_data_dark, tlist=tlist_dark)
                    dark_light_results = analysis.analyze_dark_light_dynamics(
                        params, sim_data_dark, 2, 3, metric_type='time_evolution')
                    dark_pop = float(np.mean(dark_light_results['dark_pop_time']))

            elif metric_type == 'time_evolution':
                tlist = np.linspace(0, t, 100)
                sim_data = simulate_time_evolution(sim_data, tlist)
                efficiency = float(get_final_trap_population_time_evolution(sim_data))

            elif metric_type == 'decay_rate':
                if not params.using_L_decay_rate:
                    raise ValueError("Enable L_decay_rate for decay_rate metric")
                sim_data = compute_decay_rate(sim_data)
                efficiency = float(get_decay_rate(sim_data))

            return efficiency, dark_pop, geometry, sim_data

        except Exception:
            return (0.0, 0.0 if calc_dark_pop else None, None, None) if metric_type == 'steady_state' else (0.0, None, None, None)

    # Genetic Algorithm Setup
    required_emitters = params.num_emitters - 2

    # Create DEAP types with full precision printing
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    # Attribute generators with bounds enforcement
    toolbox.register("attr_x", random.uniform, x_bounds[0], x_bounds[1])
    toolbox.register("attr_y", random.uniform, y_bounds[0], y_bounds[1])

    # Structure initializers
    toolbox.register("individual", tools.initCycle, creator.Individual,
                    (toolbox.attr_x, toolbox.attr_y), n=required_emitters)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Custom bounded mutation operator
    def mutBounded(individual, mu=0, sigma=0.1, indpb=0.2):
        for i in range(len(individual)):
            if random.random() < indpb:
                if i % 2 == 0:  # X-coordinate
                    individual[i] = np.clip(
                        individual[i] + random.gauss(mu, sigma),
                        x_bounds[0], x_bounds[1])
                else:  # Y-coordinate
                    individual[i] = np.clip(
                        individual[i] + random.gauss(mu, sigma),
                        y_bounds[0], y_bounds[1])
        return individual,

    # Genetic operators
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", mutBounded)
    toolbox.register("select", tools.selTournament, tournsize=3)

    def evaluate(individual):
        """Evaluate individual with strict bounds checking"""
        try:
            positions = [(individual[i], individual[i+1])
                        for i in range(0, len(individual), 2)]

            # Check bounds
            for x, y in positions:
                if not (x_bounds[0] <= x <= x_bounds[1]) or \
                   not (y_bounds[0] <= y <= y_bounds[1]):
                    return (0.0,)

            # Check fixed position collisions
            if any(np.allclose(pos, fixed_emitter1_pos, atol=1e-3) or
                   np.allclose(pos, fixed_emitter2_pos, atol=1e-3)
                   for pos in positions):
                return (0.0,)

            efficiency, _, _, _ = _process_combination(tuple(positions))
            return (float(efficiency),)  # Ensure full precision is maintained
        except Exception:
            return (0.0,)

    toolbox.register("evaluate", evaluate)

    # Configure statistics with full precision printing
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", lambda x: f"{np.mean(x):.20f}")
    stats.register("min", lambda x: f"{np.min(x):.20f}")
    stats.register("max", lambda x: f"{np.max(x):.20f}")

    population = toolbox.population(n=population_size)
    population, logbook = algorithms.eaSimple(
        population, toolbox, cxpb=cxpb, mutpb=mutpb, ngen=generations,
        stats=stats, verbose=True)

    # Extract and return results
    best_individual = tools.selBest(population, k=1)[0]
    best_positions = [(best_individual[i], best_individual[i+1])
                     for i in range(0, len(best_individual), 2)]
    max_efficiency, optimal_dark_pop, optimal_geometry, optimal_sim_data = \
        _process_combination(tuple(best_positions))

    return (
        tuple(best_positions),
        float(max_efficiency),  # Ensure full precision is maintained
        None,
        optimal_geometry,
        optimal_sim_data,
        float(optimal_dark_pop) if optimal_dark_pop is not None else None,
        None,
        None
    )


def optimize_emitter_omegas_ga(
    params: Sim_params,
    geometry: Geometry,
    omega_ranges: list[tuple[float, float]],
    metric_type: str = 'steady_state',
    t: float = None,
    population_size: int = 50,
    generations: int = 40,
    cxpb: float = 0.7,
    mutpb: float = 0.25
) -> tuple:
    """Optimize emitter omega values using a genetic algorithm for any number of emitters"""

    # Validate input
    num_emitters = params.num_emitters
    if len(omega_ranges) != num_emitters:
        raise ValueError(f"Must provide ranges for {num_emitters} emitters")

    # Genetic Algorithm Setup
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    # Attribute initialization for all emitters using provided ranges
    for i in range(num_emitters):
        min_o, max_o = omega_ranges[i]
        toolbox.register(f"attr_omega_{i}", random.uniform, min_o, max_o)

    # Individual creation (values for all emitters)
    toolbox.register("individual", tools.initCycle, creator.Individual,
                     (toolbox.__getattribute__(f"attr_omega_{i}") for i in range(num_emitters)),
                     n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Custom bounded mutation using input ranges
    def mutBounded(individual, mu=0, sigma=0.1, indpb=0.2):
        for i in range(len(individual)):
            if random.random() < indpb:
                min_o, max_o = omega_ranges[i]
                individual[i] = np.clip(
                    individual[i] + random.gauss(mu, sigma),
                    min_o, max_o
                )
        return individual,

    # Genetic operators
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", mutBounded)
    toolbox.register("select", tools.selTournament, tournsize=3)

    def evaluate(individual):
        """Evaluate fitness while preserving trap parameters"""
        try:
            # Verify individual stays within bounds
            for i, omega in enumerate(individual):
                min_o, max_o = omega_ranges[i]
                if not (min_o <= omega <= max_o):
                    return (0.0,)

            new_params = copy.deepcopy(params)
            new_params.emitter_omega_vals = np.array([new_params.ground_omega] + list(individual))
            new_params.dimension = len(new_params.emitter_omega_vals) + len(new_params.trap_omega_vals)

            if 1 >= new_params.dimension:
                return (0.0,)

            hamiltonian = Hamiltonian.build(new_params, geometry)
            sim_data = SimulationData(hamiltonian=hamiltonian)
            sim_data = set_initial_state(
                sim_data,
                initial_excited_state(sim_data.num_levels, index=1)
            )

            if metric_type == 'steady_state':
                sim_data = compute_steady_state(sim_data)
                efficiency = get_final_trap_population_steady_state(sim_data)
            elif metric_type == 'time_evolution':
                tlist = np.linspace(0, t, 10)
                sim_data = simulate_time_evolution(sim_data, tlist)
                efficiency = get_final_trap_population_time_evolution(sim_data)
            elif metric_type == 'decay_rate':
                sim_data = compute_decay_rate(sim_data)
                efficiency = get_decay_rate(sim_data)
            else:
                efficiency = 0.0

            return (efficiency,)

        except Exception:
            return (0.0,)

    toolbox.register("evaluate", evaluate)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)

    population = toolbox.population(n=population_size)
    population, logbook = algorithms.eaSimple(
        population, toolbox, cxpb=cxpb, mutpb=mutpb, ngen=generations,
        stats=stats, verbose=True
    )

    best_ind = tools.selBest(population, k=1)[0]
    best_omegas = np.array(best_ind)
    best_efficiency = best_ind.fitness.values[0]

    return best_omegas, best_efficiency, logbook


def gen_linear_chains(
    params: Sim_params,
    num_emitters_range: tuple[int, int],
    metric_type: str = 'steady_state',
    t: float = None
) -> list[tuple]:
    """
    Simulate linear chains of emitters with fixed spacing of 1 between emitters and trap
    for multiple emitter counts within specified range.

    Args:
        params: Simulation parameters object
        num_emitters_range: Tuple of (min_emitters, max_emitters) to test
        metric_type: Type of efficiency metric ('steady_state', 'time_evolution', 'decay_rate')
        t: Time for time evolution metric (required if metric_type='time_evolution')

    Returns:
        list: List of tuples containing results for each emitter count, where each tuple contains:
            - num_emitters: Number of emitters in this configuration
            - efficiency: The calculated efficiency metric
            - sim_data: Simulation data object
            - geometry: Geometry object used
            - pos_dict: Dictionary of positions for emitters and trap
    """
    # Validate input
    if metric_type not in ['steady_state', 'time_evolution', 'decay_rate']:
        raise ValueError("Invalid metric_type")
    if metric_type == 'time_evolution' and t is None:
        raise ValueError("Time t must be provided for time_evolution metric")
    if len(num_emitters_range) != 2 or num_emitters_range[0] < 1 or num_emitters_range[1] < num_emitters_range[0]:
        raise ValueError("num_emitters_range must be a tuple of (min, max) where min >= 1 and max >= min")

    results = []

    for num_emitters in range(num_emitters_range[0], num_emitters_range[1] + 1):
        # Create positions for linear chain
        pos_dict = {}
        for i in range(1, num_emitters + 1):
            pos_dict[i] = (float(i) - 1.0, 0.0)  # x positions: 0, 1, 2,... N-1

        # Add trap position (spaced by 1 from last emitter)
        pos_dict[num_emitters + 1] = (num_emitters, 0.0)

        new_params = Sim_params(
            num_emitters=num_emitters,
            trap_omega=params.trap_omega,
            d_real=float(num_emitters),
            using_traps=params.using_traps,
            using_photons=params.using_photons,
            using_phonons=params.using_phonons,
            using_L_decay_rate=params.using_L_decay_rate,
            temperatures=params.temperatures,
            spectral_photon_scaling=params.spectral_photon_scaling,
            spectral_phonon_scaling=params.spectral_phonon_scaling
        )

        # Create geometry
        geometry = make_geometry_from_positions(new_params, pos_dict)

        # Build Hamiltonian and simulate
        hamiltonian = Hamiltonian.build(new_params, geometry)
        sim_data = SimulationData(hamiltonian=hamiltonian)
        sim_data = set_initial_state(
            sim_data,
            initial_excited_state(sim_data.num_levels, index=1)
        )

        # Calculate efficiency based on metric type
        efficiency = 0.0
        if metric_type == 'steady_state':
            sim_data = compute_steady_state(sim_data)
            efficiency = get_final_trap_population_steady_state(sim_data)
        elif metric_type == 'time_evolution':
            tlist = np.linspace(0, t, 100)
            sim_data = simulate_time_evolution(sim_data, tlist)
            efficiency = get_final_trap_population_time_evolution(sim_data)
        elif metric_type == 'decay_rate':
            sim_data = compute_decay_rate(sim_data)
            efficiency = get_decay_rate(sim_data)

        # Store results for this emitter count
        results.append((num_emitters, efficiency, sim_data, geometry, pos_dict))

        print(f"Num emitters: {num_emitters}, Efficiency: {efficiency}")

    return results
