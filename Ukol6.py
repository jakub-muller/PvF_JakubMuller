import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
from alive_progress import alive_bar

# Parameters
domain = [0.0, 1.0, 0.0, 1.0]
Nx, Ny = 100, 100
dx = (domain[1] - domain[0]) / (Nx - 1)
dy = (domain[3] - domain[2]) / (Ny - 1)
phi1 = 0.0
phi2 = 100.0
epsilon0 = 8.854e-12

# Grid
x = np.linspace(domain[0], domain[1], Nx)
y = np.linspace(domain[2], domain[3], Ny)
X, Y = np.meshgrid(x, y, indexing='ij')

def solve_laplace_equation(phi, fixed_mask, tolerance=1e-4, max_iter=10000):
    phi_new = phi.copy()
    for it in range(max_iter):
        phi_old = phi_new.copy()
        phi_new[1:-1,1:-1] = 0.25 * (
            phi_old[:-2,1:-1] + phi_old[2:,1:-1] +
            phi_old[1:-1,:-2] + phi_old[1:-1,2:]
        )
        phi_new[fixed_mask] = phi[fixed_mask]
        diff = np.max(np.abs(phi_new - phi_old))
        if diff < tolerance:
            break
    return phi_new

def setup_condenser():
    phi = np.zeros((Nx, Ny))
    fixed = np.zeros_like(phi, dtype=bool)
    phi[:, 0] = phi1; fixed[:, 0] = True
    phi[:, -1] = phi2; fixed[:, -1] = True
    return solve_laplace_equation(phi, fixed), None

def setup_person_on_ground():
    phi = np.zeros((Nx, Ny))
    fixed = np.zeros_like(phi, dtype=bool)
    phi[:, 0] = phi1; fixed[:, 0] = True
    phi[:, -1] = phi2; fixed[:, -1] = True
    mask_person_conductor = (X - 0.5)**2 + (Y - 0.2)**2 < 0.1**2
    phi[mask_person_conductor] = phi1; fixed[mask_person_conductor] = True
    return solve_laplace_equation(phi, fixed), mask_person_conductor

def setup_person_and_lightning_rod(mask_person_conductor):
    phi = np.zeros((Nx, Ny))
    fixed = np.zeros_like(phi, dtype=bool)
    phi[:, 0] = phi1; fixed[:, 0] = True
    phi[:, -1] = phi2; fixed[:, -1] = True
    phi[mask_person_conductor] = phi1; fixed[mask_person_conductor] = True
    mask_lightning_rod = (np.abs(X - 0.8) < dx) & (Y >= 0.1)
    phi[mask_lightning_rod] = phi1; fixed[mask_lightning_rod] = True
    return solve_laplace_equation(phi, fixed), mask_lightning_rod

def compute_electric_field(phi):
    Ex = -(phi[2:,1:-1] - phi[:-2,1:-1]) / (2 * dx)
    Ey = -(phi[1:-1,2:] - phi[1:-1,:-2]) / (2 * dy)
    return Ex, Ey

def plot_potential_distribution(ax, X, Y, phi, title, vmin, vmax):
    cs = ax.contourf(X, Y, phi, 50, cmap='viridis', vmin=vmin, vmax=vmax)
    plt.colorbar(cs, ax=ax, label='Potential φ')
    ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')

def plot_charge_density_distribution(ax, X, Y, phi, mask, vmin, vmax):
    lap = np.zeros_like(phi)
    lap[1:-1,1:-1] = (
        (phi[:-2,1:-1] - 2*phi[1:-1,1:-1] + phi[2:,1:-1]) / dx**2 +
        (phi[1:-1,:-2] - 2*phi[1:-1,1:-1] + phi[1:-1,2:]) / dy**2
    )
    rho = -epsilon0 * lap
    pcm = ax.pcolormesh(X, Y, rho, shading='auto', cmap='RdBu', vmin=vmin, vmax=vmax)
    if mask is not None:
        ax.contour(X, Y, mask, levels=[0.5], colors='k')
    plt.colorbar(pcm, ax=ax, label='Charge density ρ')
    ax.set_title('Charge density')
    ax.set_xlabel('x')
    ax.set_ylabel('y')

def plot_electric_field(ax, X, Y, phi):
    Ex, Ey = compute_electric_field(phi)
    ax.quiver(x[1:-1], y[1:-1], Ex.T, Ey.T)
    ax.set_title('Electric Field')
    ax.set_xlabel('x')
    ax.set_ylabel('y')

def solve_case_setup(case_id, mask_person_conductor=None):
    if case_id == 1:
        return setup_condenser()
    elif case_id == 2:
        return setup_person_on_ground()
    elif case_id == 3 and mask_person_conductor is not None:
        return setup_person_and_lightning_rod(mask_person_conductor)
    else:
        raise ValueError("Invalid case_id or missing mask_person_conductor.")

def main():
    num_cores = mp.cpu_count()

    with alive_bar(3, title="Solving Laplace cases") as bar:
        phi_case1, _ = solve_case_setup(1)
        bar()

        phi_case2, mask_person_conductor = solve_case_setup(2)
        bar()

        with mp.Pool(processes=num_cores) as pool:
            result = pool.apply_async(solve_case_setup, args=(3, mask_person_conductor))
            phi_case3, mask_lightning_rod = result.get()
            bar()

    vmin_phi = min(np.min(phi_case1), np.min(phi_case2), np.min(phi_case3))
    vmax_phi = max(np.max(phi_case1), np.max(phi_case2), np.max(phi_case3))

    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    plot_potential_distribution(axs[0, 0], X, Y, phi_case1, 'Case 1: Condenser', vmin_phi, vmax_phi)
    plot_potential_distribution(axs[0, 1], X, Y, phi_case2, 'Case 2: Person Conductor', vmin_phi, vmax_phi)
    plot_potential_distribution(axs[0, 2], X, Y, phi_case3, 'Case 3: Person + Lightning Rod', vmin_phi, vmax_phi)

    vmin_rho, vmax_rho = None, None
    lap_case2 = (phi_case2[:-2,1:-1] - 2*phi_case2[1:-1,1:-1] + phi_case2[2:,1:-1]) / dx**2 + (phi_case2[1:-1,:-2] - 2*phi_case2[1:-1,1:-1] + phi_case2[1:-1,2:]) / dy**2
    lap_case3 = (phi_case3[:-2,1:-1] - 2*phi_case3[1:-1,1:-1] + phi_case3[2:,1:-1]) / dx**2 + (phi_case3[1:-1,:-2] - 2*phi_case3[1:-1,1:-1] + phi_case3[1:-1,2:]) / dy**2
    rho_case2 = -epsilon0 * lap_case2
    rho_case3 = -epsilon0 * lap_case3
    vmin_rho = min(np.min(rho_case2), np.min(rho_case3))
    vmax_rho = max(np.max(rho_case2), np.max(rho_case3))

    plot_charge_density_distribution(axs[1, 0], X, Y, phi_case2, mask_person_conductor, vmin_rho, vmax_rho)
    plot_charge_density_distribution(axs[1, 1], X, Y, phi_case3, mask_lightning_rod, vmin_rho, vmax_rho)
    plot_electric_field(axs[1, 2], X, Y, phi_case1)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()


