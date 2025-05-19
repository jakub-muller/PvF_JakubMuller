# TASK 1: DFT, IDFT and amplitude spectrum functions

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def discrete_fourier_transform(signal):
    N = len(signal)
    argument = -2j * np.pi / N * np.arange(N)
    H = np.array([sum(signal * np.exp(k * argument)) for k in range(N)]) / N
    return H

def inverse_discrete_fourier_transform(signal):
    N = len(signal)
    argument = 2j * np.pi / N * np.arange(N)
    h = np.array([sum(signal * np.exp(k * argument)) for k in range(N)])
    return h

def amplitude_spectrum(components, fs=1, log=False):
    N = len(components)
    result = 2 * np.abs(components)
    result[0] = 0.5 * result[0]
    if log:
        result = np.log(result + 1e-10)
    frequencies = np.linspace(0, fs // 2, N // 2)
    return frequencies, result[0:(N // 2)]

# TASK 2: Generate signal and compute spectrum

N = 2000
fs = 2000
t = np.arange(N) / fs

a = [0.1, 0.2, 0.3]
F = [440, 550, 660]

h = sum(a[n] * np.sin(2 * np.pi * F[n] * t) for n in range(3))

H = discrete_fourier_transform(h)
frequencies, S = amplitude_spectrum(H, fs=fs)

plt.figure(figsize=(10, 4))
plt.plot(frequencies, S)
plt.title("Power spectrum of the signal")
plt.xlabel("Frequency $f$ [Hz]")
plt.ylabel("Amplitude $S_k$")
plt.grid(True)
plt.tight_layout()
plt.show()

# TASK 3: Nonlinear damped-driven pendulum and its spectrum

eta = 0.5
omega = 1.0
A = 1.0
t_interval = (0, 1000)
fs = 2
t_eval = np.arange(t_interval[0], t_interval[1], 1/fs)

def solve_nonlinear_pendulum(omega_b):
    def equation(t, y):
        theta, theta_dot = y
        dtheta_dt = theta_dot
        dtheta_dot_dt = A * np.sin(omega_b * t) - eta * theta_dot - omega**2 * np.sin(theta)
        return [dtheta_dt, dtheta_dot_dt]
    y0 = [0.0, 0.0]
    sol = solve_ivp(equation, t_interval, y0, t_eval=t_eval, method='RK45')
    return sol.t, sol.y[0]

def compute_power_spectrum(signal, fs):
    N = len(signal)
    argument = -2j * np.pi / N * np.arange(N)
    H = np.array([sum(signal * np.exp(k * argument)) for k in range(N)]) / N
    spectrum = 2 * np.abs(H)
    spectrum[0] *= 0.5
    frequencies = np.linspace(0, fs // 2, N // 2)
    return frequencies, spectrum[0:(N // 2)]

omega_bs = [0.2, 0.5, 0.8]
results = {}

for omega_b in omega_bs:
    t, theta = solve_nonlinear_pendulum(omega_b)
    f, S = compute_power_spectrum(theta, fs)
    results[omega_b] = (f, S)

# Log-log graph for chaotic case (ω_b = 0.5 rad/s)
f, S = results[0.5]
mask = f < 200
fk = f[mask]
Sk = S[mask]

x = np.log(fk + 1e-10)
y = np.log(Sk + 1e-10)
a, b = np.polyfit(x, y, deg=1)

# Final plot: 2x2 layout
plt.figure(figsize=(12, 8))

# Spectra
for idx, omega_b in enumerate([0.2, 0.5, 0.8]):
    f, S = results[omega_b]
    plt.subplot(2, 2, idx + 1)
    plt.plot(f, S)
    plt.title(f"$\omega_b = {omega_b}\ \mathrm{{rad/s}}$")
    plt.xlabel("Frequency $f$ [Hz]")
    plt.ylabel("Amplitude $S_k$")
    plt.grid(True)

# Log-log graph
plt.subplot(2, 2, 4)
plt.plot(x, y, label='log($S_k$) vs log($f_k$)', color='orange')
plt.plot(x, a * x + b, '--', label=f'Fit: $y = {a:.2f}x + {b:.2f}$', color='orangered')
plt.title("Log-log for $\omega_b = 0.5$ rad/s")
plt.xlabel("log($f$)")
plt.ylabel("log($S$)")
plt.legend()
plt.grid(True)

plt.suptitle("Spectral analysis of nonlinear pendulum (Task 3)", fontsize=15)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# Print fit parameters
print("Fit parameters:")
print(f"Slope a = {a:.4f}")
print(f"Intercept b = {b:.4f}")
