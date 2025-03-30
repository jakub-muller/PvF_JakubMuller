import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
import time
import Ukol3  # Module containing the original integration functions

def sphere_integral_parallel(dimension, total_samples, num_workers):
    """
    Parallel estimator for the integral over a d-dimensional unit sphere.
    
    Parameters:
        dimension (int): The dimension of the sphere.
        total_samples (int): The total number of samples.
        num_workers (int): The number of worker processes.
    
    Returns:
        float: The averaged integral estimate computed by the worker processes.
    """
    # Divide the total number of samples among the worker processes.
    samples_per_worker = total_samples // num_workers
    with mp.Pool(processes=num_workers) as pool:
        # Each process computes the integral using samples_per_worker samples.
        results = pool.starmap(Ukol3.I_d_dim_sphere, [[dimension, samples_per_worker]] * num_workers)
    return np.average(results)

def monte_carlo_gauss_heart_parallel(total_samples=10_000_00, x_min=-3.0, x_max=3.0,
                                       y_min=-3.0, y_max=3.0, num_workers=mp.cpu_count()):
    """
    Parallel estimator for the integral of a 2D Gaussian over a heart-shaped region
    using Monte Carlo sampling.
    
    Parameters:
        total_samples (int): Total number of samples (default: 1,000,000).
        x_min, x_max, y_min, y_max (float): Bounds of the sampling region.
        num_workers (int): Number of worker processes (default: number of available CPU cores).
        
    Returns:
        float: The averaged integral estimate computed by the worker processes.
    """
    # Divide the total number of samples among the worker processes.
    samples_per_worker = total_samples // num_workers
    with mp.Pool(processes=num_workers) as pool:
        # Each process computes the Monte Carlo estimate using samples_per_worker samples.
        results = pool.starmap(
            Ukol3.monte_carlo_gauss_heart,
            [(samples_per_worker, x_min, x_max, y_min, y_max)] * num_workers
        )
    return np.average(results)

def time_sphere_integration(dimension, max_workers=8, total_samples=10_000_000):
    """
    Measure and plot the computation time of the parallel sphere integral estimator
    using different numbers of worker processes.
    
    Parameters:
        dimension (int): The dimension for the sphere integral.
        max_workers (int): The maximum number of worker processes to test.
        total_samples (int): Total number of samples.
    """
    computation_times = []  # List to store elapsed times
    worker_range = range(1, max_workers + 1)

    for workers in worker_range:
        start_time = time.time()
        integral_value = sphere_integral_parallel(dimension, total_samples, workers)
        elapsed_time = time.time() - start_time

        computation_times.append(elapsed_time)
        print(f"Workers: {workers}, Sphere Integral = {integral_value}, Time taken: {elapsed_time:.4f} s.")

    plt.plot(worker_range, computation_times, marker='o')
    plt.xlabel("Number of Workers")
    plt.ylabel("Computation Time [s]")
    plt.xticks(worker_range)
    plt.title("Sphere Integral Computation Time vs. Number of Workers")
    plt.show()

def time_heart_integration(max_workers=8, total_samples=10_000_000, x_min=-3.0, x_max=3.0, y_min=-3.0, y_max=3.0):
    """
    Measure and plot the computation time of the parallel Monte Carlo estimator for the
    2D Gaussian over a heart-shaped region using different numbers of worker processes.
    
    Parameters:
        max_workers (int): The maximum number of worker processes to test.
        total_samples (int): Total number of samples.
        x_min, x_max, y_min, y_max (float): Bounds of the sampling region.
    """
    computation_times = []  # List to store elapsed times
    worker_range = range(1, max_workers + 1)

    for workers in worker_range:
        start_time = time.time()
        integral_value = monte_carlo_gauss_heart_parallel(total_samples, x_min, x_max, y_min, y_max, num_workers=workers)
        elapsed_time = time.time() - start_time

        computation_times.append(elapsed_time)
        print(f"Workers: {workers}, Heart Integral = {integral_value}, Time taken: {elapsed_time:.4f} s.")

    plt.plot(worker_range, computation_times, marker='o')
    plt.xlabel("Number of Workers")
    plt.ylabel("Computation Time [s]")
    plt.xticks(worker_range)
    plt.title("Heart Integral Computation Time vs. Number of Workers")
    plt.show()

if __name__ == "__main__":
    print("Number of logical cores:", mp.cpu_count())
    
    # Example: Compute the sphere integral for a 2-dimensional sphere.
    time_sphere_integration(dimension=2, max_workers=15, total_samples=5_000_000)
    
    # Example: Compute the Monte Carlo estimator for the heart-shaped integral.
    time_heart_integration(max_workers=15, total_samples=5_000_000)
