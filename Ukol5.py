import numpy as np
import matplotlib.pyplot as plt
from alive_progress import alive_bar
import multiprocessing as mp
import functools

def diff_to_sys(d2=1, d1=1, d0=1, d0y=lambda y0: y0, rhs=1):
    def f(y, t):
        y0 = y[0]  # y
        y1 = y[1]  # y'
        r = rhs(t) if callable(rhs) else rhs
        dy0_dt = y1
        dy1_dt = (r - d0 * d0y(y0) - d1 * y1) / d2
        return [dy0_dt, dy1_dt]
    return f

def Euler(f, y0, t0, t1, dt):
    t = t0
    y = np.array(y0, dtype=float)
    ys = [y.copy()]
    ts = [t]

    while t < t1:
        k1 = np.array(f(y, t))
        k2 = np.array(f(y + k1 * dt, t + dt))
        y += dt * 0.5 * (k1 + k2)
        t += dt
        ys.append(y.copy())
        ts.append(t)

    return np.array(ys), np.array(ts)

def Runge_Kutta(f, y0, t0, t1, dt):
    t = t0
    y = np.array(y0, dtype=float)
    ys = [y.copy()]
    ts = [t]

    while t < t1:
        k1 = np.array(f(y, t))
        k2 = np.array(f(y + k1 * dt/2, t + dt/2))
        k3 = np.array(f(y + k2 * dt/2, t + dt/2))
        k4 = np.array(f(y + k3 * dt, t + dt))
        phi = (k1 + 2 * k2 + 2 * k3 + k4) / 6
        y += dt * phi
        t += dt
        ys.append(y.copy())
        ts.append(t)

    return np.array(ys), np.array(ts)

def plot_results(wb, dt, y0, t0, t1, d0, d1, d0y, A):
    for w in wb:
        f = diff_to_sys(d2=1, d1=d1, d0=d0, d0y=d0y, rhs=lambda t, w=w: A * np.sin(w * t))
        with alive_bar(len(dt) * 2, title=f"Solving for w={w}") as bar:
            for dt_ in dt:
                ys_rk, ts_rk = Runge_Kutta(f, y0, t0, t1, dt_)
                plt.plot(ts_rk, ys_rk[:, 0], label=f'Runge-Kutta, dt={dt_}, w={w}')
                bar()

                ys_euler, ts_euler = Euler(f, y0, t0, t1, dt_)
                plt.plot(ts_euler, ys_euler[:, 0], label=f'Euler, dt={dt_}, w={w}')
                bar()

        plt.title(f'Phase Space:')
        plt.xlabel('Displacement (y)')
        plt.ylabel("Velocity (v)")
        plt.grid()
        plt.legend()
        plt.show()

def process_poincare_task(w, v0, dt_fixed, t0, t1, d0, d1, A):
    Tb = 2 * np.pi / w
    f = diff_to_sys(d2=1, d1=d1, d0=d0, d0y=np.sin, rhs=lambda t: A * np.sin(w * t))
    y0_var = [0.0, v0]
    ys_rk, ts_rk = Runge_Kutta(f, y0_var, t0, t1, dt_fixed)
    ys_euler, ts_euler = Euler(f, y0_var, t0, t1, dt_fixed)
    return (ys_rk, ts_rk, Tb), (ys_euler, ts_euler, Tb)

def plot_poincare(wb, dt_fixed, y0, t0, t1, d0, d1, d0y, A, num_cores=2):
    tasks = [(w, v0, dt_fixed, t0, t1, d0, d1, A) for w in wb for v0 in np.linspace(0, 1, 10)]

    with mp.Pool(processes=num_cores) as pool:
        with alive_bar(len(tasks), title="Generating Poincaré plots") as bar:
            for result in pool.starmap(process_poincare_task, tasks):
                for ys, ts, Tb in result:
                    poincare(ys, ts, Tb)
                bar()

    plt.title('Poincaré Section')
    plt.xlabel('Displacement (y)')
    plt.ylabel("Velocity (v)")
    plt.grid()
    plt.show()

def poincare(ys, ts, T):
    points = []
    for t, y in zip(ts, ys):
        if np.isclose(t % T, 0, atol=0.01):
            points.append(y)
    points = np.array(points)
    if len(points):
        plt.plot(points[:, 0], points[:, 1], linestyle='-', linewidth=1, color=random_color())

def random_color():
    return "#{:06x}".format(np.random.randint(0, 0xFFFFFF))

def main():
    d1 = 0.5
    d0 = 1.0
    A = 1.0
    wb = [0.2, 0.4, 0.6]
    t0 = 0.0
    t1 = 500.0
    y0 = [0.0, 1.0]
    dt = [0.01, 0.005]
    d0y = np.sin

    plot_results(wb, dt, y0, t0, t1, d0, d1, d0y, A)

    dt_fixed = 0.005
    num_cores = 4  # Set desired number of cores here
    plot_poincare(wb, dt_fixed, y0, t0, t1, d0, d1, d0y, A, num_cores=num_cores)

if __name__ == "__main__":
    main()

# Bohužel jsem nemohl přijít na minulé cvičení, takže je tenhle kód silně inspirován prací kolegy. Každopádně jsem se snažil pochopit, jak funguje. Vše jsem testoval a zdá se mi, že program pracuje správně. Snad je to v pořádku...
