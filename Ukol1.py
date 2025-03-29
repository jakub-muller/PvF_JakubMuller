import numpy as np
import matplotlib.pyplot as plt


#--------------------------------------------------------------------------
# Popis, jak to funguje:
# Pomocí několika funkcí je vypočítáno logistické zobrazení, bifurkační diagram, lyapunovův exponent a Feigenbaumova konstanta.
# Bifurkační diagram a Ljapunovův exponent jsou vykresleny na grafu, Bifurkační body a Feigenbaumova konstanta jsou vypsány.
#--------------------------------------------------------------------------
# Common logistic-map functions
def logistic_map(r, x):
    return r * x * (1 - x)

def logistic_map_derivative(r, x):
    return r * (1 - 2*x)

# Bifurcation diagram and period-doubling detection
def compute_period(r, tol, iterations=1000, last=100):
    """
    Computes the period (number of unique attractor points) for the logistic map at a given r.
    Two points are considered identical if they differ by less than tol.
    """
    x = 0.5
    xs = []
    for i in range(iterations):
        x = logistic_map(r, x)
        if i >= (iterations - last):
            xs.append(x)
    
    # Group points by tolerance
    unique_points = []
    for val in xs:
        if not any(abs(val - up) < tol for up in unique_points):
            unique_points.append(val)
    return len(unique_points)

def find_bifurcation_point(r_lower, r_upper, expected_period, tol, 
                           iterations=1000, last=100, search_tol=1e-6):
    """
    Uses a binary search to locate the r value where the period of the attractor
    increases (doubles) from the expected_period.
    """
    while (r_upper - r_lower) > search_tol:
        r_mid = (r_lower + r_upper) / 2.0
        period_mid = compute_period(r_mid, tol, iterations, last)
        if period_mid <= expected_period:
            # Not yet bifurcated
            r_lower = r_mid
        else:
            # Bifurcation has occurred
            r_upper = r_mid
    return (r_lower + r_upper) / 2.0

def calculate_bifurcation_points(n, tol, iterations=1000, last=100):
    """
    Calculates the first n bifurcation points (where the period doubles)
    for the logistic map.
    """
    bifurcation_points = []
    current_period = 1  # Start with period 1 (before the first bifurcation)
    r_lower = 2.5
    r_upper = 4.0

    for _ in range(n):
        bp = find_bifurcation_point(r_lower, r_upper, current_period, tol, 
                                    iterations, last)
        bifurcation_points.append(bp)
        # Next search starts just above the newly found point
        r_lower = bp + 1e-6
        current_period *= 2
    return bifurcation_points

def plot_bifurcation(ax, r_min=2.5, r_max=4.0, r_steps=10000, 
                     iterations=1000, last=100):
    """
    Plots the bifurcation diagram for the logistic map on the given Axes 'ax'.
    """
    r_values = np.linspace(r_min, r_max, r_steps)
    r_list = []
    x_list = []
    
    x = 0.5  # initial condition
    for r in r_values:
        x = 0.5  # reset for each r
        for i in range(iterations):
            x = logistic_map(r, x)
            if i >= (iterations - last):
                r_list.append(r)
                x_list.append(x)
    
    ax.plot(r_list, x_list, ',k', alpha=0.25)
    ax.set_xlabel('r')
    ax.set_ylabel('x')
    ax.set_title('Bifurcation Diagram')

# Lyapunov exponent
def compute_lyapunov_exponent(r, x0=0.5, n=1000, discard=200):
    """
    Computes the Lyapunov exponent for the logistic map at parameter r.
    lambda = (1/n) sum_{j=0 to n-1} ln |f'(x_j)|
    """
    x = x0
    for _ in range(discard):
        x = logistic_map(r, x)
    
    lyapunov_sum = 0.0
    for _ in range(n):
        deriv = logistic_map_derivative(r, x)
        lyapunov_sum += np.log(abs(deriv) + 1e-15)
        x = logistic_map(r, x)
    
    return lyapunov_sum / n

def plot_lyapunov_exponent(ax, r_min=0.0, r_max=4.0, steps=1000, 
                           x0=0.5, n=1000, discard=200):
    """
    Plots the Lyapunov exponent of the logistic map for r in [r_min, r_max]
    on the given Axes 'ax'.
    """
    r_values = np.linspace(r_min, r_max, steps)
    lyapunov_values = []
    
    for r in r_values:
        lam = compute_lyapunov_exponent(r, x0, n, discard)
        lyapunov_values.append(lam)
    
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.plot(r_values, lyapunov_values, 'r-', linewidth=1)
    ax.set_xlabel('r')
    ax.set_ylabel('Lyapunov Exponent, λ')
    ax.set_title('Lyapunov Exponent')
    ax.set_ylim(-3, 1)  # Show more negative range

# Feigenbaum constant approximation
def approximate_feigenbaum_constant(bif_points):
    """
    Given a list of bifurcation points a_1, a_2, ..., a_n,
    computes the average of the ratios:
        (a_{j+1} - a_j) / (a_{j+2} - a_{j+1})
    for all j where j+2 < len(bif_points).
    """
    ratios = []
    # For example, if we have 5 points: a1, a2, a3, a4, a5,
    # we can form ratios for j=0->(a2-a1)/(a3-a2), j=1->(a3-a2)/(a4-a3), j=2->(a4-a3)/(a5-a4)
    for j in range(len(bif_points) - 2):
        numerator = bif_points[j+1] - bif_points[j]
        denominator = bif_points[j+2] - bif_points[j+1]
        if abs(denominator) > 1e-15:  # Avoid division by zero
            ratio = numerator / denominator
            ratios.append(ratio)
    if not ratios:
        return None  # Not enough points to compute
    return sum(ratios) / len(ratios)

# ------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------
# 1. Calculate and print the first 5 bifurcation points
tol = 1e-3
bif_points = calculate_bifurcation_points(n=5, tol=tol, iterations=2000, last=500)
known_points = [3.0, 3.44949, 3.54409, 3.56440, 3.56876]  # approximate known values

print("First 5 Bifurcation Points:")
for i, (calc, known) in enumerate(zip(bif_points, known_points), start=1):
    print(f"  Bifurcation {i}: computed = {calc:.5f}, known ≈ {known}")

# 2. Compute and print the approximate Feigenbaum constant
feigenbaum_approx = approximate_feigenbaum_constant(bif_points)
if feigenbaum_approx is not None:
    print(f"\nApproximate Feigenbaum constant from first 5 points: {feigenbaum_approx:.6f}")
else:
    print("\nNot enough bifurcation points to compute the Feigenbaum constant.")

# 3. Create side-by-side plots for Bifurcation Diagram (left) and Lyapunov Exponent (right)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Bifurcation diagram on ax1
plot_bifurcation(ax1, r_min=2.5, r_max=4.0, r_steps=5000, iterations=1000, last=100)

# Lyapunov exponent on ax2
plot_lyapunov_exponent(ax2, r_min=0, r_max=4, steps=1000, x0=0.5, n=1000, discard=200)

plt.tight_layout()
plt.show()
# ------------------------------------------------------------------------
# Komentář ke kódu:
# Vážně dlouho jsem se snažil vše vyřešit sám, první tři úlohy mi nedělaly problém. 
# Dál jsem se ale trochu zasekl a musím se přiznat, že přesto, že jsem měl plán, jak bifurkační body najít, nebyl jsem schopen to realizovat.
# Radši jsem tedy zkusil velmi podrobně popsat mé plány ChatuGPT, který všechno po dlouhé době strávené úpravami a vylepšováním dokázal vyřešit.
# Rád bych všechno zvládl sám, ale python je pro mě letos nový a popravdě jsem byl na některé úkoly krátký.
# Nicméně jsem rád, že se mi podařilo dopracovat se s AI ke zdárnému konci a beru to jako pokrok, ještě před měsícem bych s ním takhle dobře pracovat nezvládl.
# Dlouhou dobu jsem taky věnoval hraním si s kódem a snažil jsem se vážně pochopit každý detail.
# Doufám že tenhle způsob řešení nějak moc nevadí. Děkuji za pochopení.
# ------------------------------------------------------------------------