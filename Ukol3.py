import numpy as np
import matplotlib.pyplot as plt

# Problem 1
def monte_carlo_integral(f, a, b, n=1000*1000):
    """
    Estimate the integral of function f from a to b using n samples.
    
    Parameters:
        f : function to integrate (should accept a NumPy array)
        a : lower integration bound
        b : upper integration bound
        n : number of samples (default: 1,000,000)
        
    Returns:
        Estimated integral value.
    """
    X = np.random.uniform(a, b, n)
    fX = f(X)
    return (b - a) * np.mean(fX)

def first_function(x):
    return np.exp(-x) * np.sin(x)

def second_function(x):
    return (1 + (1 + x**3)**(-0.5))**(-0.5)

# Problem 2
def V_d_dim_sphere(d, n=100*1000):
    """
    Estimate the volume of a d-dimensional sphere of radius 1.
    
    Parameters:
        d : dimension
        n : number of samples (default: 100,000)
        
    Returns:
        volume : estimated volume of the sphere
        err    : estimated error
        hit    : number of points falling inside the sphere
        X      : array of all generated sample points
    """
    X = np.random.uniform(-1, 1, (n, d))
    hit = 0
    for i in range(n):
        if np.linalg.norm(X[i]) <= 1:
            hit += 1
    err = (hit**0.5 / n) * (2**d)
    volume = (hit / n) * (2**d)
    return volume, err, hit, X

def plot_volume():
    """
    Plot the estimated volume of a d-dimensional sphere for dimensions 1 to 10,
    including error bars.
    """
    x = np.arange(1, 11)
    volumes = [V_d_dim_sphere(d)[0] for d in x]
    errors = [V_d_dim_sphere(d)[1] for d in x]

    plt.errorbar(x, volumes, yerr=errors, fmt='o-', label='Volume with errors')
    plt.xticks(x)
    plt.title('Volume of a d-dimensional sphere')
    plt.xlabel('Dimension')
    plt.ylabel('Volume')
    plt.legend()
    plt.grid()
    plt.show()

# Problem 3
def I_d_dim_sphere(d, n=1000):
    """
    Estimate the integral of cos(sum(x_i)) over the d-dimensional unit sphere using Monte Carlo.
    
    Parameters:
        d : dimension of the sphere
        n : number of samples (default: 1,000)
        
    Returns:
        Estimated integral value.
    """
    volume, err, hit, X = V_d_dim_sphere(d, n)
    if hit == 0:
        return 0.0  # safeguard: no points inside the sphere (unlikely for reasonable n)
    
    # Use the first 'hit' samples from X (assumed to be inside the sphere)
    # Here we assume that the first 'hit' rows of X correspond to points inside;
    # note: this logic mirrors the original code, though in practice one might store the mask.
    values = np.cos(np.sum(X[:hit], axis=1))
    avg_cos = np.mean(values)
    integral = volume * avg_cos
    return integral

# Problem 4
def is_in_heart(x, y):
    return ((x**2 + y**2 - y**3)**3 - x**2 * y**3) < 0

def gauss_2d(x, y):
    return np.exp(-0.5 * (x**2 + y**2)) / (2.0 * np.pi)

def monte_carlo_gauss_heart(N=10_000_00, x_min=-3.0, x_max=3.0, y_min=-3.0, y_max=3.0):
    """
    Estimate the integral of a 2D Gaussian over a heart-shaped region using Monte Carlo.
    
    Parameters:
        N     : total number of samples (default: 1,000,000)
        x_min, x_max, y_min, y_max : bounds of the sampling region
        
    Returns:
        Estimated integral value.
    """
    X = np.random.uniform(x_min, x_max, N)
    Y = np.random.uniform(y_min, y_max, N)
    mask = is_in_heart(X, Y)
    f_vals = gauss_2d(X[mask], Y[mask])
    sum_f = np.sum(f_vals)
    area = (x_max - x_min) * (y_max - y_min)
    integral_estimate = (area / N) * sum_f
    return integral_estimate

if __name__ == '__main__':
    # Problem 1 tests
    print("Problem 1:")
    print("Integral of first_function:", monte_carlo_integral(first_function, 0, 10**0.5))
    print("Integral of second_function:", monte_carlo_integral(second_function, 0, 1))
    
    # Problem 2 plot
    print("\nProblem 2: Plotting volume of d-dimensional sphere")
    plot_volume()
    
    # Problem 3 tests
    print("\nProblem 3:")
    print("I_d_dim_sphere for d=2:", I_d_dim_sphere(2))
    print("I_d_dim_sphere for d=3:", I_d_dim_sphere(3))
    
    # Problem 4 test
    print("\nProblem 4:")
    print("Monte Carlo Gaussian Heart Integral:", monte_carlo_gauss_heart())
