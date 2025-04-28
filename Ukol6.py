import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
from alive_progress import alive_bar

# Parameters
domain = [0.0, 1.0, 0.0, 1.0]
Nx, Ny = 100, 100
dx = (domain[1] - domain[0]) / (Nx - 1)
dy = (domain[3] - domain[2]) / (Ny - 1)

phi1 = 100.0
phi0 = 0.0
epsilon0 = 8.854e-12

# Grid
x = np.linspace(domain[0], domain[1], Nx)
y = np.linspace(domain[2], domain[3], Ny)
X, Y = np.meshgrid(x, y, indexing='ij')


def solve_laplace(phi, fixed_mask, tolerance=1e-4, max_iter=10000):
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

def case_parallel_plates():
    phi = np.zeros((Nx, Ny))
    fixed = np.zeros_like(phi, dtype=bool)
    phi[:, 0] = phi0; fixed[:, 0] = True
    phi[:, -1] = phi1; fixed[:, -1] = True
    return solve_laplace(phi, fixed), None

def case_person():
    phi = np.zeros((Nx, Ny))
    fixed = np.zeros_like(phi, dtype=bool)
    phi[:, 0] = phi0; fixed[:, 0] = True
    phi[:, -1] = phi1; fixed[:, -1] = True
    mask_person = (X - 0.5)**2 + (Y - 0.2)**2 < 0.1**2
    phi[mask_person] = phi0; fixed[mask_person] = True
    return solve_laplace(phi, fixed), mask_person

def case_person_and_rod(mask_person):
    phi = np.zeros((Nx, Ny))
    fixed = np.zeros_like(phi, dtype=bool)
    phi[:, 0] = phi0; fixed[:, 0] = True
    phi[:, -1] = phi1; fixed[:, -1] = True
    phi[mask_person] = phi0; fixed[mask_person] = True
    mask_rod = (np.abs(X - 0.8) < dx) & (Y >= 0.1)
    phi[mask_rod] = phi0; fixed[mask_rod] = True
    return solve_laplace(phi, fixed), mask_rod

def compute_field(phi):
    Ex = -(phi[2:,1:-1] - phi[:-2,1:-1]) / (2 * dx)
    Ey = -(phi[1:-1,2:] - phi[1:-1,:-2]) / (2 * dy)
    return Ex, Ey

def plot_potential(X, Y, phi, title):
    plt.figure()
    cs = plt.contourf(X, Y, phi, 50, cmap='viridis')
    plt.colorbar(cs, label='Potential φ')
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')

def plot_charge_density(X, Y, phi, mask):
    lap = np.zeros_like(phi)
    lap[1:-1,1:-1] = (
        (phi[:-2,1:-1] - 2*phi[1:-1,1:-1] + phi[2:,1:-1]) / dx**2 +
        (phi[1:-1,:-2] - 2*phi[1:-1,1:-1] + phi[1:-1,2:]) / dy**2
    )
    rho = -epsilon0 * lap
    plt.figure()
    pcm = plt.pcolormesh(X, Y, rho, shading='auto', cmap='RdBu')
    if mask is not None:
        plt.contour(X, Y, mask, levels=[0.5], colors='k')
    plt.colorbar(pcm, label='Charge density ρ')
    plt.title('Charge density')
    plt.xlabel('x')
    plt.ylabel('y')

def plot_field(X, Y, phi):
    Ex, Ey = compute_field(phi)
    plt.figure(figsize=(5,5))
    plt.quiver(x[1:-1], y[1:-1], Ex.T, Ey.T)
    plt.title('Electric Field')
    plt.xlabel('x')
    plt.ylabel('y')

def solve_case(case_id, mask_person=None):
    if case_id == 1:
        return case_parallel_plates()
    elif case_id == 2:
        return case_person()
    elif case_id == 3 and mask_person is not None:
        return case_person_and_rod(mask_person)
    else:
        raise ValueError("Invalid case_id or missing mask_person.")

def main():
    num_cores = mp.cpu_count()

    with alive_bar(3, title="Solving Laplace cases") as bar:
        with mp.Pool(processes=num_cores) as pool:
            # First two cases separately to pass dependencies
            phi_case1, _ = solve_case(1)
            bar()

            phi_case2, mask_person = solve_case(2)
            bar()

            result = pool.apply_async(solve_case, args=(3, mask_person))
            phi_case3, mask_rod = result.get()
            bar()

    # Plot results
    plot_potential(X, Y, phi_case1, 'Case 1: Parallel Plates')
    plot_potential(X, Y, phi_case2, 'Case 2: Person Conductor')
    plot_potential(X, Y, phi_case3, 'Case 3: Person + Lightning Rod')

    # Plot charge densities
    plot_charge_density(X, Y, phi_case2, mask_person)
    plot_charge_density(X, Y, phi_case3, mask_rod)

    # Plot E-field for parallel plates
    plot_field(X, Y, phi_case1)

    plt.show()

if __name__ == "__main__":
    main()
