# type: ignore
import matplotlib.pyplot as plt
import numpy as np
import qutip as qt
from matplotlib.gridspec import GridSpec
import matplotlib.colors as mcolors
from typing import Optional
from core.geometry import Geometry
from core.simulator import SimulationData
from matplotlib.collections import LineCollection
from configs.parameters import Sim_params
from matplotlib.figure import Figure
from configs.system_functions import SystemFunctionsData
import configs.system_functions as sys_functs

# Color scheme constants
COLORS = {
    'emitter': '#CE4257',
    'first_emitter': '#00A896',
    'edge': '#2c3e50',
    'text': 'black',
    'background': '#ffffff',
    'coupling1': '#ffc1cf',
    'coupling2': '#ff001e',
    'level_0': '#00A896',  # First emitter
    'level_1': '#CE4257',  # Emitter 1
    'level_2': '#FF9B54',  # Emitter 2
    'level_3': '#7209B7',  # Emitter 3
    'trap': '#FFBF46',      # Trap state
    'grid': '#E0E0E0',
    'spines': '#808080',
    'legend_bg': '#ffffff30',  # Semi-transparent white
    'axis_bg': '#ffffff'
}

def plot_geometry(sim_data: SimulationData, geometry: Geometry, title: str = 'Network Geometry with Couplings',
                 dpi: int = 150, fontsize: int = 30) -> plt.Figure:
    """Plot network geometry with full styling from old version"""
    # Extract data from geometry
    positions = geometry.positions
    emitter_edges = geometry.emitter_edges_n_coupling
    trap_edges = geometry.trap_edges_n_coupling

    # Color configuration (preserving original scheme)
    colors = {
        'background': '#ffffff',
        'axis_bg': '#ffffff',
        'emitter': '#CE4257',
        'trap': '#FFBF46',      # Trap state
        'first_emitter': '#00A896',
        'text': 'white',
        'text_ax': 'black',
        'text_bg': 'black',
        'spines': '#BFBFBF',
        'edge': 'black',
        'legend_bg': 'black',
        'grid': '#E0E0E0'
    }

    # Create grayscale colormap with darker starting point
    line_cmap = mcolors.LinearSegmentedColormap.from_list(
        'custom_cmap', ['#808080', '#202020']  # Light gray to dark gray
    )

    # Figure setup
    fig, ax = plt.subplots(figsize=(14, 6), facecolor=colors['background'], dpi=dpi)
    ax.set_facecolor(colors['axis_bg'])

    # Prepare data for LineCollection
    nodes = sorted(positions.keys())
    coords = np.array([positions[node] for node in nodes])

    # ========================================================================
    # Key Fix: Proper node handling for first emitter and trap
    # ========================================================================

    # Plot main emitters (all except first and last)
    if len(coords) > 2:
        ax.scatter(coords[1:-1, 0], coords[1:-1, 1],
                 c=colors['emitter'], s=100, edgecolor=colors['edge'],
                 linewidth=1.2, zorder=3)

    # Plot first emitter (always index 0)
    if len(coords) > 0:
        ax.scatter(coords[0][0], coords[0][1],
                 c='red', s=100, edgecolor=colors['edge'],
                 linewidth=2.0, zorder=4)

    # Plot trap (always last node)
    if len(coords) > 1:
        ax.scatter(coords[-1][0], coords[-1][1],
                 c=colors['trap'], s=300, marker='D',
                 edgecolor=colors['edge'], linewidth=1.2,
                 zorder=4)

    # ========================================================================
    # Plot emitter-emitter couplings
    # ========================================================================
    lines = []
    strengths = []
    for edge in emitter_edges:
        node_i, node_j, strength = edge
        x1, y1 = positions[node_i]
        x2, y2 = positions[node_j]
        lines.append([(x1, y1), (x2, y2)])
        strengths.append(strength)

    # Create LineCollection for emitter couplings
    norm = mcolors.Normalize(vmin=np.min(strengths), vmax=np.max(strengths))
    lc = LineCollection(lines, cmap=line_cmap, norm=norm,
                       linewidths=2, alpha=0.7, zorder=1)
    lc.set_array(np.array(strengths))
    ax.add_collection(lc)

    # ========================================================================
    # Plot trap coupling (dotted line)
    # ========================================================================
    if trap_edges is not None:
        for edge in trap_edges:
            node_i, node_j, strength = edge
            x1, y1 = positions[node_i]
            x2, y2 = positions[node_j]
            ax.plot([x1, x2], [y1, y2],
                    linestyle=':', color='black',
                    linewidth=2, alpha=0.7, zorder=1)

    # Add colorbar
    cbar = plt.colorbar(lc, ax=ax, pad=0.01)
    cbar.set_label('(eV)', color=colors['text_ax'], fontsize=fontsize)
    cbar.ax.yaxis.set_tick_params(color=colors['text_ax'], labelsize=fontsize)
    plt.setp(cbar.ax.get_yticklabels(), color=colors['text_ax'], fontsize=fontsize)

    # Styling
    ax.set_xlabel('X (nm)', fontsize=fontsize+4, color=colors['text_ax'], labelpad=10)
    ax.set_ylabel('Y (nm)', fontsize=fontsize+4, color=colors['text_ax'], labelpad=10)
    #ax.set_title(title, fontsize=fontsize+4, color=colors['text_ax'], pad=15)

    ax.grid(True, color=colors['grid'], linestyle='--',
           linewidth=0.7, alpha=0.5)
    ax.spines['bottom'].set_color(colors['spines'])
    ax.spines['left'].set_color(colors['spines'])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.tick_params(axis='both', colors=colors['text_ax'], labelsize=fontsize)
    ax.ticklabel_format(style='sci', scilimits=(0,0), useMathText=True)
    ax.yaxis.get_offset_text().set_fontsize(fontsize)
    ax.xaxis.get_offset_text().set_fontsize(fontsize)

    plt.tight_layout()
    return fig

def plot_omega_optimization_2emitters(results_data, params: Sim_params,
                                     opt_point: tuple = None,
                                     title: str = 'Emitter omega Optimization',
                                     figsize: tuple = (16, 8)) -> plt.Figure:
    """
    Plot 2D optimization results for emitter omega values (2-emitter systems only)

    Args:
        results_data: Output from optimize_emitter_omegas
        params: Simulation parameters used for title info
        opt_point: (best_omega1, best_omega2, best_efficiency) to highlight
        title: Base title for the plot
        figsize: Figure dimensions

    Returns:
        matplotlib Figure object
    """
    # Extract and format data
    combinations, efficiencies = zip(*results_data[0])
    omega1_vals = np.array([c[0] for c in combinations])
    omega2_vals = np.array([c[1] for c in combinations])
    eff_vals = np.array(efficiencies)

    # Create grid for surface plots
    omega1_unique = np.unique(omega1_vals)
    omega2_unique = np.unique(omega2_vals)
    nx = len(omega1_unique)
    ny = len(omega2_unique)

    if nx * ny != len(eff_vals):
        raise ValueError("Data doesn't form regular grid - omega values must form full grid")

    # Create meshgrids for plotting
    omega1_grid, omega2_grid = np.meshgrid(omega1_unique, omega2_unique, indexing='ij')
    eff_grid = eff_vals.reshape(nx, ny)

    # Create figure
    fig = plt.figure(figsize=figsize, facecolor=COLORS['background'])
    fig.suptitle(f"{title}\nSystem Size: {params.d_real} nm",
                color=COLORS['text'], fontsize=14, y=1.02)

    # Set up grid layout
    gs = GridSpec(2, 2, width_ratios=[1.2, 1], height_ratios=[1, 0.05])
    ax1 = fig.add_subplot(gs[0, 0], projection='3d')
    ax2 = fig.add_subplot(gs[0, 1])
    cax = fig.add_subplot(gs[1, :])

    # 3D Surface plot
    surf = ax1.plot_surface(omega1_grid, omega2_grid, eff_grid,
                          cmap='viridis', edgecolor='none',
                          alpha=0.9, antialiased=True)

    ax1.set_xlabel(r'$\omega_1$ (eV)', labelpad=10, color=COLORS['text'])
    ax1.set_ylabel(r'$\omega_2$ (eV)', labelpad=10, color=COLORS['text'])
    ax1.set_zlabel('Efficiency', labelpad=10, color=COLORS['text'])
    ax1.tick_params(axis='both', colors=COLORS['text'])
    ax1.xaxis.pane.fill = False
    ax1.yaxis.pane.fill = False
    ax1.zaxis.pane.fill = False
    ax1.grid(color=COLORS['grid'], alpha=0.3)

    # Contour plot
    cont = ax2.contourf(omega1_grid, omega2_grid, eff_grid, levels=20, cmap='viridis')
    ax2.set_xlabel(r'$\omega_1$ (eV)', color=COLORS['text'])
    ax2.set_ylabel(r'$\omega_2$ (eV)', color=COLORS['text'])
    ax2.tick_params(colors=COLORS['text'])
    ax2.grid(color=COLORS['grid'], alpha=0.3)

    # Add optimal point markers
    if opt_point:
        ax1.scatter(opt_point[0], opt_point[1], opt_point[2],
                  color=COLORS['trap'], s=100, edgecolor=COLORS['edge'],
                  label=f'Optimal ({opt_point[0]:.2f}, {opt_point[1]:.2f})')

        ax2.scatter(opt_point[0], opt_point[1], color=COLORS['trap'], s=100,
                  edgecolor=COLORS['edge'], zorder=10)
        ax2.annotate(f'Optimal\n({opt_point[0]:.2f}, {opt_point[1]:.2f})',
                    (opt_point[0], opt_point[1]),
                    textcoords="offset points", xytext=(10,-10),
                    ha='left', color=COLORS['text'],
                    arrowprops=dict(arrowstyle="->", color=COLORS['text']))

    # Colorbar
    cbar = fig.colorbar(surf, cax=cax, orientation='horizontal')
    cbar.set_label('Trap Population Efficiency', color=COLORS['text'])
    cbar.ax.xaxis.set_tick_params(color=COLORS['text'])
    plt.setp(cbar.ax.get_xticklabels(), color=COLORS['text'])

    # Background colors
    for ax in [ax1, ax2]:
        ax.set_facecolor(COLORS['axis_bg'])
        for spine in ax.spines.values():
            spine.set_color(COLORS['spines'])

    # Add legend to 3D plot
    if opt_point:
        ax1.legend(facecolor=COLORS['legend_bg'], edgecolor=COLORS['edge'],
                  loc='upper right', fontsize=9)

    plt.tight_layout()
    return fig



def plot_populations(sim_data: SimulationData) -> plt.Figure:
    """
    Plot population dynamics in both level basis and eigenbasis.
    Adapted from original version with new data structure support.
    """
    # Extract data from SimulationData
    tlist = sim_data.times
    results = sim_data.results_time_sim
    H = sim_data.hamiltonian.H
    num_levels = sim_data.num_levels

    # Get eigenvalues and eigenvectors
    eigenvalues, eigenvectors = H.eigenstates()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # First plot: Population in original basis
    for i in range(num_levels):
        # Create projection operator for state i
        pop_op = qt.basis(num_levels, i) * qt.basis(num_levels, i).dag()
        pop_exp = [qt.expect(pop_op, state) for state in results]
        axes[0].plot(tlist, pop_exp, label=f'P_{i}',
                   color=COLORS[f'level_{i}'] if f'level_{i}' in COLORS else None)

    axes[0].set_xlabel('Time', color=COLORS['text'])
    axes[0].set_ylabel('Population', color=COLORS['text'])
    axes[0].legend()
    axes[0].set_title('Emitter Basis Populations', color=COLORS['text'])

    # Second plot: Population in eigenbasis
    for i in range(num_levels):
        # Create projection operator for eigenstate i
        psi = eigenvectors[i]
        pop_op = psi * psi.dag()
        pop_exp = [qt.expect(pop_op, state) for state in results]
        axes[1].plot(tlist, pop_exp, label=f'P_{i}',
                   color=COLORS[f'mode_{i}'] if f'mode_{i}' in COLORS else None)

    axes[1].set_xlabel('Time', color=COLORS['text'])
    axes[1].set_ylabel('Population', color=COLORS['text'])
    axes[1].legend()
    axes[1].set_title('Eigenbasis Populations', color=COLORS['text'])

    # Print population conservation check

    # Styling
    for ax in axes:
        ax.set_facecolor(COLORS['background'])
        ax.tick_params(colors=COLORS['text'])
        for spine in ax.spines.values():
            spine.set_color(COLORS['text'])
        ax.grid(True, color=COLORS['grid'], alpha=0.3)

    plt.tight_layout()
    return fig


def plot_population_subplots(sim_data_list: list[SimulationData], labels: list[str] = None,
                           colors: list[str] = None, linestyles: list[str] = None,
                           fontsize: int = 10, dpi: int = 130) -> plt.Figure:
    """
    Plot populations in a 2x2 grid structure, with multiple simulations plotted together.

    Args:
        sim_data_list: List of SimulationData objects containing time evolution data
        labels: List of string labels for each simulation (must match length of sim_data_list)
        colors: List of colors for each simulation (optional)
        linestyles: List of linestyles for each simulation (optional)
        fontsize: Base font size for text elements in the plot (default: 10)
        dpi: Resolution of the plot in dots per inch (default: 100)

    Returns:
        matplotlib Figure object
    """
    # Validate input
    if not sim_data_list:
        raise ValueError("No simulation data provided")

    if labels and len(labels) != len(sim_data_list):
        raise ValueError("Labels list must match length of sim_data_list")

    # Use default labels if none provided
    if not labels:
        labels = [f'Simulation {i+1}' for i in range(len(sim_data_list))]

    # Get number of states from first simulation (assume all have same number)
    num_states = sim_data_list[0].num_levels

    # Calculate grid dimensions based on number of states
    rows = int(np.ceil(num_states / 2))
    cols = 2 if num_states > 1 else 1

    # Create figure with appropriate grid
    fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows),
                           facecolor=COLORS['background'],
                           sharex=True, dpi=dpi)
    axes = axes.flatten()  # Flatten for easier iteration

    # Plot each state in its own subplot
    for state_idx in range(num_states):
        ax = axes[state_idx]

        # Plot all simulations for this state
        for sim_idx, sim_data in enumerate(sim_data_list):
            # Extract populations for this simulation
            tlist = sim_data.times
            populations = np.zeros((num_states, len(tlist)))

            for t_idx, state in enumerate(sim_data.results_time_sim):
                rho = state.full() if isinstance(state, qt.Qobj) else state
                populations[:, t_idx] = np.real(np.diag(rho))

            # Determine plot style
            color = colors[sim_idx] if colors and len(colors) > sim_idx else COLORS.get(f'level_{state_idx}', COLORS['emitter'])
            linestyle = linestyles[sim_idx] if linestyles and len(linestyles) > sim_idx else '-'

            # Plot this state's population
            ax.plot(tlist, populations[state_idx, :],
                   color=color,
                   linestyle=linestyle,
                   alpha=0.7 + 0.3*(1 - sim_idx/len(sim_data_list)),  # Vary alpha slightly
                   label=f'{labels[sim_idx]} ')

        # Subplot styling
        ax.set_facecolor(COLORS['axis_bg'])
        ax.set_ylabel(f'$\\rho_{{{state_idx}{state_idx}}}$',
                     color=COLORS['text'], fontsize=fontsize)
        ax.grid(True, color=COLORS['grid'], alpha=0.3)
        ax.tick_params(colors=COLORS['text'], labelsize=fontsize)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0), useMathText=True)
        ax.yaxis.get_offset_text().set_fontsize(fontsize)

        # Only show xlabel on bottom plots
        if state_idx >= (rows - 1) * cols:
            ax.set_xlabel(r'Time $\frac{1}{eV}$', color=COLORS['text'], fontsize=fontsize)

        # Spine coloring
        for spine in ax.spines.values():
            spine.set_color(COLORS['spines'])

    # Hide any unused axes
    for ax in axes[num_states:]:
        ax.set_visible(False)

    # Figure title
    # fig.suptitle('Population Dynamics Comparison',
    #             color=COLORS['text'],
    #             y=0.98, fontsize=fontsize+2)

    plt.tight_layout()
    return fig


def plot_dark_light_dynamics(sim_data_list: list[SimulationData], analysis_results_list: list[dict],
                            emitter1: int, emitter2: int,
                            title: str = 'Dark/Light State Dynamics',
                            figsize: tuple = (10, 6),
                            labels: list[str] = None,
                            colors: list[tuple[str, str]] = None,
                            linestyles: list[tuple[str, str]] = None,
                            fontsize: int = 10,
                            dpi: int = 100) -> plt.Figure:
    """
    Plot population dynamics of dark and light states for multiple simulations.

    Args:
        sim_data_list: List of SimulationData objects containing time evolution data
        analysis_results_list: List of dictionaries from analyze_dark_light_dynamics
        emitter1: Index of first emitter (for labeling)
        emitter2: Index of second emitter (for labeling)
        title: Plot title
        figsize: Figure size
        labels: List of labels for each simulation (optional)
        colors: List of (dark_color, light_color) tuples for each simulation (optional)
        linestyles: List of (dark_linestyle, light_linestyle) tuples for each simulation (optional)
        fontsize: Base font size for text elements
        dpi: Resolution of the plot in dots per inch

    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize, facecolor=COLORS['background'], dpi=dpi)
    ax.set_facecolor(COLORS['axis_bg'])

    # Use default labels if none provided
    if labels is None:
        labels = [f'Simulation {i+1}' for i in range(len(sim_data_list))]
    elif len(labels) != len(sim_data_list):
        raise ValueError("Labels list must match length of sim_data_list")

    # Use default colors if none provided
    if colors is None:
        colors = [COLORS.get('level_1', COLORS['emitter']),(COLORS.get('level_2', COLORS['trap']))] * len(sim_data_list)
    elif len(colors) != len(sim_data_list):
        raise ValueError("Colors list must match length of sim_data_list")

    # Use default linestyles if none provided
    if linestyles is None:
        linestyles = [('-', '--')] * len(sim_data_list)
    elif len(linestyles) != len(sim_data_list):
        raise ValueError("Linestyles list must match length of sim_data_list")

    # Plot each simulation's data
    for i, (sim_data, analysis_results) in enumerate(reversed(list(zip(sim_data_list, analysis_results_list)))):
        dark_color, light_color = colors[len(sim_data_list)-1-i]
        dark_linestyle, light_linestyle = linestyles[len(sim_data_list)-1-i]

        # Plot time evolution if available
        if analysis_results['dark_pop_time'] is not None:
            ax.plot(sim_data.times, analysis_results['dark_pop_time'],
                   color=dark_color, linestyle=dark_linestyle,
                   linewidth=2.5, alpha=0.7 + 0.3*(1 - (len(sim_data_list)-1-i)/len(sim_data_list)))

            ax.plot(sim_data.times, analysis_results['light_pop_time'],
                   color=light_color, linestyle=light_linestyle,
                   linewidth=2.5, alpha=0.7 + 0.3*(1 - (len(sim_data_list)-1-i)/len(sim_data_list)))

    # Add magnified inset
    ax_inset = ax.inset_axes([0.6, 0.6, 0.35, 0.35])  # x, y, width, height
    for i, (sim_data, analysis_results) in enumerate(reversed(list(zip(sim_data_list, analysis_results_list)))):
        dark_color, light_color = colors[len(sim_data_list)-1-i]
        dark_linestyle, light_linestyle = linestyles[len(sim_data_list)-1-i]

        if analysis_results['dark_pop_time'] is not None:
            ax_inset.plot(sim_data.times, analysis_results['dark_pop_time'],
                         color=dark_color, linestyle=dark_linestyle,
                         linewidth=1.5, alpha=0.7 + 0.3*(1 - (len(sim_data_list)-1-i)/len(sim_data_list)))
            ax_inset.plot(sim_data.times, analysis_results['light_pop_time'],
                         color=light_color, linestyle=light_linestyle,
                         linewidth=1.5, alpha=0.7 + 0.3*(1 - (len(sim_data_list)-1-i)/len(sim_data_list)))

    ax_inset.set_xlim(1600, 1750)
    ax_inset.set_ylim(0.41, 0.42)
    ax_inset.grid(True, color=COLORS['grid'], linestyle='--', alpha=0.5)
    ax_inset.set_xticks([])
    ax_inset.set_yticks([])
    ax.indicate_inset_zoom(ax_inset, edgecolor="black")

    # Styling
    ax.set_xlabel(r'Time $\frac{1}{eV}$', color=COLORS['text'], fontsize=fontsize)
    ax.set_ylabel('Population', color=COLORS['text'], fontsize=fontsize)
    # ax.set_title(f'{title}\nEmitters {emitter1} & {emitter2}',
    #             color=COLORS['text'], fontsize=fontsize+2, pad=15)

    # Grid and frame
    ax.grid(True, color=COLORS['grid'], linestyle='--', alpha=0.7)
    for spine in ax.spines.values():
        spine.set_color(COLORS['spines'])
        spine.set_linewidth(1.2)

    # Tick params
    ax.tick_params(axis='both', colors=COLORS['text'], labelsize=fontsize)

    plt.tight_layout()
    return fig

def plot_steady_state_results(sim_data: SimulationData) -> plt.Figure:
    """Plot bar chart comparing steady state vs final time population"""
    fig, ax = plt.subplots(figsize=(8, 5))

    labels = ['Steady State', 'Time Evolution']
    values = [sim_data.results_steady_state[-1,-1].real,
             sim_data.results_time_sim[-1][-1,-1].real]

    ax.bar(labels, values, color=[COLORS['trap'], COLORS['emitter']])
    ax.set_ylabel('Trap Population', color=COLORS['text'])
    ax.tick_params(axis='y', colors=COLORS['text'])
    plt.xticks(color=COLORS['text'])
    return fig

def plot_density_matrix_evolution(sim_data: SimulationData) -> plt.Figure:
    """Plot real and imaginary parts of density matrix elements"""
    fig, ax = plt.subplots(sim_data.num_levels, sim_data.num_levels,
                          figsize=(12, 12))

    for i in range(sim_data.num_levels):
        for j in range(sim_data.num_levels):
            # Extract matrix elements
            re = [rho[i,j].real for rho in sim_data.results_time_sim]
            im = [rho[i,j].imag for rho in sim_data.results_time_sim]

            ax[i,j].plot(sim_data.times, re, label='Real',
                        color=COLORS['coupling1'])
            ax[i,j].plot(sim_data.times, im, label='Imag',
                        color=COLORS['coupling2'], linestyle='--')

            ax[i,j].set_title(f'ρ[{i},{j}]', color=COLORS['text'], fontsize=8)

    plt.tight_layout()
    return fig

def plot_all(sim_data: SimulationData, geometry: Geometry) -> dict:
    """Generate all standard plots return dictionary of figures"""
    return {
        'geometry': plot_geometry(sim_data, geometry),
        'dynamics': plot_populations(sim_data),
        'steady_state': plot_steady_state_results(sim_data),
        'density_matrix': plot_density_matrix_evolution(sim_data),
        'population_subplots': plot_population_subplots(sim_data)
    }

def plot_dark_light_optimization(y_spacings: np.ndarray,
                                dark_pops: np.ndarray,
                                light_pops: np.ndarray,
                                params: Sim_params,
                                opt_y: float = None,
                                title: str = 'Dark/Light State Optimization',
                                figsize: tuple = (10, 6)) -> plt.Figure:
    """
    Plot steady state dark and light state populations vs y-spacing

    Args:
        y_spacings: Array of y-spacing values (nm)
        dark_pops: Array of dark state populations for each y-spacing
        light_pops: Array of light state populations for each y-spacing
        params: Simulation parameters used for title info
        opt_y: Optimal y-spacing to highlight (optional)
        title: Base title for the plot
        figsize: Figure dimensions

    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize, facecolor=COLORS['background'])
    ax.set_facecolor(COLORS['axis_bg'])

    # Plot main curves
    ax.plot(y_spacings, dark_pops,
           color=COLORS.get('level_1', COLORS['emitter']),
           linewidth=2.5, label='Dark State')

    ax.plot(y_spacings, light_pops,
           color=COLORS.get('level_2', COLORS['trap']),
           linewidth=2.5, linestyle='--', label='Light State')

    # Highlight optimal point if provided
    if opt_y is not None:
        # Find nearest index
        idx = np.argmin(np.abs(y_spacings - opt_y))

        # Add vertical line
        ax.axvline(opt_y, color=COLORS['edge'], linestyle=':',
                  alpha=0.8, label=f'Optimal y = {opt_y:.2f} nm')

        # Add markers
        ax.scatter(opt_y, dark_pops[idx], color=COLORS['level_1'],
                  s=100, edgecolor=COLORS['edge'], zorder=5)
        ax.scatter(opt_y, light_pops[idx], color=COLORS['level_2'],
                  s=100, edgecolor=COLORS['edge'], zorder=5)

    # Styling
    ax.set_xlabel('Y Spacing (nm)', color=COLORS['text'], fontsize=12)
    ax.set_ylabel('Steady State Population', color=COLORS['text'], fontsize=12)
    full_title = (f"{title}\n"
                 f"d_real = {params.d_real} nm, ω = {params.emitter_omega} eV")
    ax.set_title(full_title, color=COLORS['text'], fontsize=14, pad=15)

    # Grid and frame
    ax.grid(True, color=COLORS['grid'], linestyle='--', alpha=0.7)
    for spine in ax.spines.values():
        spine.set_color(COLORS['spines'])
        spine.set_linewidth(1.2)

    # Legend
    legend = ax.legend(facecolor=COLORS['legend_bg'],
                      edgecolor=COLORS['edge'],
                      loc='upper right', fontsize=10)
    plt.setp(legend.get_texts(), color=COLORS['text'])

    # Tick params
    ax.tick_params(axis='both', colors=COLORS['text'], labelsize=10)

    plt.tight_layout()
    return fig

def plot_4emitter_optimization(results: tuple, params: Sim_params) -> plt.Figure:
    """
    Plot y-spacing optimization results with consistent styling
    Args:
        results: Tuple from optimize_4emitter_ring_y_spacing
        params: Simulation parameters used for title info
    """
    y_spacings, efficiencies, opt_y, opt_eff, _ = results

    fig, ax = plt.subplots(figsize=(10, 5), dpi=300, facecolor=COLORS['background'])
    ax.set_facecolor(COLORS['axis_bg'])

    # Main plot line
    ax.plot(y_spacings, efficiencies,
           color=COLORS['emitter'], linewidth=2.5,
           alpha=0.9, label='Transfer Efficiency')

    # Optimal point
    ax.scatter(opt_y, opt_eff, color=COLORS['trap'], s=120,
              zorder=5, edgecolor=COLORS['edge'], linewidth=1.2,
              label=f'Optimal (y = {opt_y:.2f})')

    # Vertical line styling
    ax.axvline(opt_y, color=COLORS['edge'], linestyle='--', alpha=0.8, linewidth=1.2)

    # Text annotation
    ax.annotate(f'Max: {opt_eff:.3f}',
               xy=(opt_y, opt_eff),
               xytext=(opt_y*1.1, opt_eff*0.9),
               arrowprops=dict(facecolor=COLORS['edge'], shrink=0.05),
               fontsize=10, color=COLORS['text'])

    # Axis labels and titles
    ax.set_xlabel('Y Spacing (nm)', fontsize=12, color=COLORS['text'])
    ax.set_ylabel('Trap Population', fontsize=12, color=COLORS['text'])
    title = f'4-Emitter Ring Optimization (d_real={params.d_real} nm, omega={params.emitter_omega} eV)'
    ax.set_title(title, fontsize=13, color=COLORS['text'], pad=15)

    # Grid and frame
    ax.grid(True, color=COLORS['grid'], linestyle='--', alpha=0.8)
    for spine in ax.spines.values():
        spine.set_color(COLORS['spines'])
        spine.set_linewidth(1.2)

    # Legend and ticks
    legend = ax.legend(frameon=True, facecolor=COLORS['legend_bg'],
                      edgecolor=COLORS['edge'], fontsize=10)
    legend.get_frame().set_linewidth(1.2)
    ax.tick_params(axis='both', colors=COLORS['text'], labelsize=10)

    plt.tight_layout()
    return fig


def plot_6emitter_optimization(results: tuple, params: Sim_params) -> plt.Figure:
    """
    Plot 2D optimization results as a 3D surface + contour plot combination
    """
    y1_grid, y2_grid, pop_grid, opt_y1, opt_y2, max_pop = results

    # Create figure with subplots
    fig = plt.figure(figsize=(16, 8), dpi=300, facecolor=COLORS['background'])
    fig.suptitle(f'6-Emitter Ring Optimization (d_real={params.d_real} nm)',
                color=COLORS['text'], fontsize=14, y=1.02)

    # Create grid layout
    gs = GridSpec(2, 2, width_ratios=[1.2, 1], height_ratios=[1, 0.05])

    # 3D Surface plot
    ax1 = fig.add_subplot(gs[0, 0], projection='3d')
    surf = ax1.plot_surface(y1_grid, y2_grid, pop_grid, cmap='Greys',
                          rstride=1, cstride=1, alpha=1.0,  # Changed alpha from 0.8 to 1.0
                          linewidth=0, antialiased=True)

    # Contour plot
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.contourf(y1_grid, y2_grid, pop_grid, levels=20, cmap='Greys')  # Removed unused variable

    # Shared colorbar
    cax = fig.add_subplot(gs[1, :])
    cbar = fig.colorbar(surf, cax=cax, orientation='horizontal')
    cbar.set_label('Trap Population', color=COLORS['text'])

    # Optimal point markers
    for ax in [ax1, ax2]:
        if ax == ax1:
            ax.scatter(opt_y1, opt_y2, max_pop, color=COLORS['trap'], s=100,
                      edgecolor=COLORS['edge'], label='Optimal Point', zorder=10)
        else:
            ax.scatter(opt_y1, opt_y2, color=COLORS['trap'], s=100,
                      edgecolor=COLORS['edge'], label=f'Optimal\n({opt_y1:.2f}, {opt_y2:.2f})')

    # 3D plot styling
    ax1.set_xlabel('Y1 Spacing (nm)', labelpad=10, color=COLORS['text'])
    ax1.set_ylabel('Y2 Spacing (nm)', labelpad=10, color=COLORS['text'])
    ax1.set_zlabel('Trap Population', labelpad=10, color=COLORS['text'])
    ax1.tick_params(axis='both', colors=COLORS['text'])
    ax1.xaxis.pane.fill = False
    ax1.yaxis.pane.fill = False
    ax1.zaxis.pane.fill = False
    ax1.grid(color=COLORS['grid'], alpha=0.3)

    # Contour plot styling
    ax2.set_xlabel('Y1 Spacing (nm)', color=COLORS['text'])
    ax2.set_ylabel('Y2 Spacing (nm)', color=COLORS['text'])
    ax2.tick_params(colors=COLORS['text'])
    ax2.grid(color=COLORS['grid'], alpha=0.3)

    # Colorbar styling
    cbar.ax.xaxis.set_tick_params(color=COLORS['text'])
    cbar.ax.set_xlabel('Trap Population', color=COLORS['text'])
    plt.setp(cbar.ax.get_xticklabels(), color=COLORS['text'])

    # Background colors
    for ax in [ax1, ax2]:
        ax.set_facecolor(COLORS['axis_bg'])
        for spine in ax.spines.values():
            spine.set_color(COLORS['spines'])

    # Add legend to contour plot
    ax2.legend(facecolor=COLORS['legend_bg'], edgecolor=COLORS['edge'],
              loc='upper right', fontsize=9)

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)  # Make space for suptitle

    return fig



def plot_system_functions(
    data: SystemFunctionsData,
    dpi: int = 100,
    title_fontsize: int = 14,
    label_fontsize: int = 12,
    tick_fontsize: int = 10
) -> plt.Figure:
    """
    Plot photon and phonon system functions with separate plots for spectral density and noise power spectrum.
    Includes Bose-Einstein distribution for both particle types on same plot.

    Args:
        data: SystemFunctionsData containing parameters and particle kind
        dpi: Resolution of the figure in dots per inch
        title_fontsize: Font size for titles
        label_fontsize: Font size for axis labels
        tick_fontsize: Font size for tick labels
    """
    # Create frequency grids for both photon and phonon
    photon_spec_dens_omega = np.linspace(-1, 12, 500)
    photon_noise_power_omega = np.linspace(-4, 12, 500)
    phonon_spec_dens_omega = np.linspace(-0.1, 0.3, 200)
    phonon_noise_power_omega = np.linspace(-0.2, 0.3, 500)
    omega_vals = np.linspace(-0.3, 0.3, 500)


    # Create figure with subplots
    fig = plt.figure(figsize=(18, 10), facecolor='white', dpi=dpi)
    # fig.suptitle('System Functions Analysis',
    #             fontsize=title_fontsize, color=COLORS['text'], y=1.02)

    # Create grid layout with 2x3 subplots and adjust spacing
    gs = GridSpec(2, 3, figure=fig, wspace=0.4, hspace=0.4)
    ax1 = fig.add_subplot(gs[0, 0])  # Photon Spectral Density
    ax2 = fig.add_subplot(gs[0, 1])  # Phonon Spectral Density
    ax3 = fig.add_subplot(gs[0, 2])  # Bose-Einstein
    ax4 = fig.add_subplot(gs[1, 0])  # Photon Noise Power Spectrum
    ax5 = fig.add_subplot(gs[1, 1])  # Phonon Noise Power Spectrum

    # Plot photon spectral density
    photon_data = SystemFunctionsData(data.p, 'photon')
    J_photon = [sys_functs.spectral_density(photon_data, w) for w in photon_spec_dens_omega]
    ax1.plot(photon_spec_dens_omega, J_photon,
            color='blue',
            label=r'Photon $J(\omega)$')
    # ax1.set_title('Photon Spectral Density', color=COLORS['text'], fontsize=title_fontsize)
    ax1.set_xlabel(r'$\omega$ (eV)', color=COLORS['text'], fontsize=label_fontsize)
    ax1.set_ylabel(r'$J(\omega)$ (eV)', color=COLORS['text'], fontsize=label_fontsize)
    # ax1.legend(facecolor=COLORS['legend_bg'])
    ax1.tick_params(axis='both', colors=COLORS['text'], labelsize=tick_fontsize)

    # Plot phonon spectral density
    phonon_data = SystemFunctionsData(data.p, 'phonon')
    J_phonon = [sys_functs.spectral_density(phonon_data, w) for w in phonon_spec_dens_omega]
    ax2.plot(phonon_spec_dens_omega, J_phonon,
            color='black',
            label=r'Phonon $J(\omega)$')
    # ax2.set_title('Phonon Spectral Density', color=COLORS['text'], fontsize=title_fontsize)
    ax2.set_xlabel(r'$\omega$ (eV)', color=COLORS['text'], fontsize=label_fontsize)
    ax2.set_ylabel(r'$J(\omega)$ (eV)', color=COLORS['text'], fontsize=label_fontsize)
    # ax2.legend(facecolor=COLORS['legend_bg'])
    ax2.tick_params(axis='both', colors=COLORS['text'], labelsize=tick_fontsize)

    # Plot Bose-Einstein for both photon and phonon
    for kind in ['photon', 'phonon']:
        temp_data = SystemFunctionsData(data.p, kind)
        n_vals = [sys_functs.bose_einstein(temp_data, w) for w in omega_vals]
        ax3.plot(omega_vals, n_vals,
                color='blue' if kind == 'photon' else 'black',
                label=f'{kind}')

    # ax3.set_title('Bose-Einstein Distribution', color=COLORS['text'], fontsize=title_fontsize)
    ax3.set_xlabel(r'$\omega$ (eV)', color=COLORS['text'], fontsize=label_fontsize)
    ax3.set_ylabel(r'$N \; (\omega)$', color=COLORS['text'], fontsize=label_fontsize)
    ax3.set_ylim(-40, 40)
    ax3.legend(facecolor=COLORS['legend_bg'])
    ax3.tick_params(axis='both', colors=COLORS['text'], labelsize=tick_fontsize)

    # Plot photon noise power spectrum
    photon_data = SystemFunctionsData(data.p, 'photon')
    S_photon = [sys_functs.noise_power_spectrum(photon_data, w) for w in photon_noise_power_omega]
    ax4.plot(photon_noise_power_omega, S_photon,
            color='blue',
            label=r'Photon $S(\omega)$')
    # ax4.set_title('Photon Noise Power Spectrum', color=COLORS['text'], fontsize=title_fontsize)
    ax4.set_xlabel(r'$\omega$ (eV)', color=COLORS['text'], fontsize=label_fontsize)
    ax4.set_ylabel(r'$S(\omega)$ (eV)', color=COLORS['text'], fontsize=label_fontsize)
    #ax4.legend(facecolor=COLORS['legend_bg'])
    ax4.tick_params(axis='both', colors=COLORS['text'], labelsize=tick_fontsize)

    # Plot phonon noise power spectrum
    phonon_data = SystemFunctionsData(data.p, 'phonon')
    S_phonon = [sys_functs.noise_power_spectrum(phonon_data, w) for w in phonon_noise_power_omega]
    ax5.plot(phonon_noise_power_omega, S_phonon,
            color='black',
            label=r'Phonon $S(\omega)$')
    # ax5.set_title('Phonon Noise Power Spectrum', color=COLORS['text'], fontsize=title_fontsize)
    ax5.set_xlabel(r'$\omega$ (eV)', color=COLORS['text'], fontsize=label_fontsize)
    ax5.set_ylabel(r'$S(\omega)$ (eV)', color=COLORS['text'], fontsize=label_fontsize)
    #ax5.legend(facecolor=COLORS['legend_bg'])
    ax5.tick_params(axis='both', colors=COLORS['text'], labelsize=tick_fontsize)

    # Style adjustments for all axes
    for ax in [ax1, ax2, ax3, ax4, ax5]:
        ax.set_facecolor('white')
        ax.grid(True, color=COLORS['grid'], alpha=0.3)
        for spine in ax.spines.values():
            spine.set_color(COLORS['spines'])

    plt.tight_layout(pad=3.0)
    plt.savefig("sys_funcs.png", dpi=300, bbox_inches='tight')
    return fig


def _plot_spectral_density(ax, data, omega_vals):
    """Plot spectral density components"""
    J = [sys_functs.spectral_density(data, w) for w in omega_vals]

    ax.plot(omega_vals, J, color=COLORS['coupling1'],
             label=f'{data.particle_kind} J(omega)')

    ax.set_title('Spectral Density', color=COLORS['text'])
    ax.set_xlabel('Frequency omega (eV)', color=COLORS['text'])
    ax.set_ylabel('J(omega) (eV)', color=COLORS['text'])
    ax.legend(facecolor=COLORS['legend_bg'])

def _plot_bose_einstein(ax, data, omega_vals):
    """Plot Bose-Einstein distributions"""
    n_vals = [sys_functs.bose_einstein(data, w) for w in omega_vals]
    ax.plot(omega_vals, n_vals, color=COLORS['level_1'],
               label=f'{data.particle_kind} n(omega)')

    ax.set_title(f'Bose-Einstein Distribution ({data.particle_kind})',
                color=COLORS['text'])
    ax.set_xlabel('Frequency omega (eV)', color=COLORS['text'])
    ax.set_ylabel('Occupation Number', color=COLORS['text'])
    ax.legend(facecolor=COLORS['legend_bg'])

def _plot_noise_power(ax, data, omega_vals):
    """Plot noise power spectrum components"""
    S_total = [sys_functs.noise_power_spectrum(data, w) for w in omega_vals]
    S_emission = [sys_functs.noise_power_emission(data, w) for w in omega_vals]
    S_absorption = [sys_functs.noise_power_absorption(data, w) for w in omega_vals]

    ax.plot(omega_vals, S_total, color=COLORS['text'], label='Total S(omega)')
    ax.plot(omega_vals, S_emission, color=COLORS['emitter'],
       label='Emission', linestyle='--')
    ax.plot(omega_vals, S_absorption, color=COLORS['trap'],
             label='Absorption', linestyle=':')

    ax.set_title('Noise Power Spectrum', color=COLORS['text'])
    ax.set_xlabel('Frequency omega (eV)', color=COLORS['text'])
    ax.set_ylabel('S(omega) (eV⁻¹)', color=COLORS['text'])
    ax.legend(facecolor=COLORS['legend_bg'])

def _plot_ohmic_spectrum(ax, data, omega_vals):
    """Plot Ohmic spectrum components"""
    gamma_vals = [sys_functs.ohmic_spectrum(data, w) for w in omega_vals]

    ax.plot(omega_vals, gamma_vals, color=COLORS['edge'],
               label='Ohmic Spectrum')
    ax.set_title('Ohmic Spectral Density', color=COLORS['text'])
    ax.set_xlabel('Frequency omega (eV)', color=COLORS['text'])
    ax.set_ylabel('γ(omega) (eV)', color=COLORS['text'])
    ax.legend(facecolor=COLORS['legend_bg'])

def _plot_temperature_scaling(ax, data):
    """Plot temperature scaling relationships"""
    T_range = np.linspace(1, 7000, 500)
    kBT_vals = [sys_functs.kelvin_to_energy(T) for T in T_range]

    ax.plot(T_range, kBT_vals, color=COLORS['level_0'],
           label='Thermal Energy Scale')
    ax.axhline(data.p.spectral_omega_c_photon, color=COLORS['coupling1'],
              linestyle='--', label='Photon Cutoff')
    ax.axhline(data.p.spectral_omega_c_phonon, color=COLORS['coupling2'],
              linestyle=':', label='Phonon Cutoff')

    ax.set_title('Temperature Scaling', color=COLORS['text'])
    ax.set_xlabel('Temperature (K)', color=COLORS['text'])
    ax.set_ylabel('Energy (eV)', color=COLORS['text'])
    ax.legend(facecolor=COLORS['legend_bg'])
    ax.grid(True, color=COLORS['grid'], alpha=0.3)


def plot_dark_population_optimization(
    dark_pop_map: dict,
    params: Sim_params,
    figsize: tuple = (10, 6),
    dpi: int = 150,
    fontsize: int = 24
) -> plt.Figure:
    """
    Plot dark population as a function of emitter position in contour plot.

    Args:
        dark_pop_map: Dictionary mapping positions to dark populations (from optimize_full_grid_geometry)
        params: Simulation parameters used for title info
        figsize: Figure dimensions
        dpi: Resolution in dots per inch
        fontsize: Font size for labels and ticks

    Returns:
        matplotlib Figure object
    """
    # Extract and format data from dark_pop_map
    positions = list(dark_pop_map.keys())
    x_vals = np.array([pos[0][0] for pos in positions])  # First emitter's x position
    y_vals = np.array([pos[0][1] for pos in positions])  # First emitter's y position
    dark_pops = np.array(list(dark_pop_map.values()))

    # Create grid for surface plots
    x_unique = np.unique(x_vals)
    y_unique = np.unique(y_vals)
    nx = len(x_unique)
    ny = len(y_unique)

    if nx * ny != len(dark_pops):
        raise ValueError("Data doesn't form regular grid - position values must form full grid")

    # Create meshgrids for plotting
    x_grid, y_grid = np.meshgrid(x_unique, y_unique, indexing='ij')
    pop_grid = dark_pops.reshape(nx, ny)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi, facecolor=COLORS['background'])

    # Contour plot
    cont = ax.contourf(x_grid, y_grid, pop_grid, levels=20, cmap='Greys')
    ax.set_xlabel('X (nm)', color=COLORS['text'], fontsize=fontsize)
    ax.set_ylabel('Y (nm)', color=COLORS['text'], fontsize=fontsize)
    ax.tick_params(colors=COLORS['text'], labelsize=fontsize)
    ax.grid(color=COLORS['grid'], alpha=0.3)

    # Colorbar
    cbar = fig.colorbar(cont, ax=ax)
    cbar.set_label('Dark State Population', color=COLORS['text'], fontsize=fontsize)
    cbar.ax.tick_params(colors=COLORS['text'], labelsize=fontsize)
    plt.setp(cbar.ax.get_yticklabels(), color=COLORS['text'], fontsize=fontsize)

    # Style adjustments
    ax.set_facecolor(COLORS['axis_bg'])
    for spine in ax.spines.values():
        spine.set_color(COLORS['spines'])

    plt.tight_layout()
    return fig


def plot_efficiency_optimization_for_varying_one_emitter(
    eff_map: dict,
    params: Sim_params,
    figsize: tuple = (10, 6),
    dpi: int = 150,
    fontsize: int = 24
) -> plt.Figure:
    """
    Plot transfer efficiency as a function of the single varying emitter's position
    in a 3-emitter system (ignoring fixed emitter positions).

    Args:
        eff_map: Dictionary mapping position combinations to efficiency values
        params: Simulation parameters
        figsize: Figure size
        dpi: Resolution
        fontsize: Font size for labels and ticks

    Returns:
        matplotlib Figure object
    """
    # Validate input
    if params.num_emitters != 3:
        raise ValueError("This function is specifically for 3-emitter systems with one varying emitter")

    # Extract positions and efficiencies
    positions = list(eff_map.keys())
    efficiencies = np.array(list(eff_map.values()))

    # Extract x and y coordinates of the varying emitter
    x_vals = []
    y_vals = []

    for pos_tuple in positions:
        if len(pos_tuple) != 1:
            raise ValueError("Expected exactly one varying emitter position per combination")
        x, y = pos_tuple[0]  # Unpack the single position tuple
        x_vals.append(x)
        y_vals.append(y)

    x_vals = np.array(x_vals)
    y_vals = np.array(y_vals)

    # Get unique sorted values for grid
    x_unique = np.sort(np.unique(x_vals))
    y_unique = np.sort(np.unique(y_vals))

    # Create meshgrid with correct orientation
    x_grid, y_grid = np.meshgrid(x_unique, y_unique, indexing='ij')

    # Reshape efficiencies to match grid (no transpose needed)
    eff_grid = efficiencies.reshape(len(x_unique), len(y_unique))

    # Calculate dynamic z-axis scaling
    z_min = np.min(efficiencies)
    z_max = np.max(efficiencies)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi, facecolor=COLORS['background'])

    # Contour plot with grayscale colormap
    levels = np.linspace(z_min, z_max, 20)
    cont = ax.contourf(x_grid, y_grid, eff_grid, levels=levels, cmap='Greys')
    ax.set_xlabel('X (nm)', color=COLORS['text'], fontsize=fontsize)
    ax.set_ylabel('Y (nm)', color=COLORS['text'], fontsize=fontsize)

    # Add colorbar
    cbar = fig.colorbar(cont, ax=ax)
    cbar.set_label('Transfer Efficiency', color=COLORS['text'], fontsize=fontsize)
    cbar.ax.tick_params(colors=COLORS['text'], labelsize=fontsize)
    plt.setp(cbar.ax.get_yticklabels(), color=COLORS['text'], fontsize=fontsize)

    # Style adjustments
    ax.set_facecolor(COLORS['axis_bg'])
    ax.grid(color=COLORS['grid'], alpha=0.3)
    for spine in ax.spines.values():
        spine.set_color(COLORS['spines'])
    ax.tick_params(axis='both', colors=COLORS['text'], labelsize=fontsize)

    plt.tight_layout()
    return fig
