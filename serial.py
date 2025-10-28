# serial.py — УСТОЙЧИВАЯ ВЕРСИЯ
import numpy as np
import time

def u_init(x):
    return np.sin(3 * np.pi * (x - 1/6))

def u_left(t): return -1.0
def u_right(t): return 1.0

def solve_serial():
    start = time.time()

    a, b = 0.0, 1.0
    t0, T_target = 0.0, 6.0
    eps = 10**(-1.5)
    N = 400

    x = np.linspace(a, b, N+1)
    h = x[1] - x[0]

    # Условие устойчивости
    tau = 0.8 * h**2 / (2 * eps)
    M = int((T_target - t0) / tau) + 1
    T = (M-1) * tau
    t = np.linspace(t0, T, M)

    print(f"УСТОЙЧИВО: N={N}, M={M}, h={h:.2e}, tau={tau:.2e}, T={T:.3f}")

    u = np.zeros((2, N+1))
    u[0, :] = u_init(x)
    u[0, 0] = u_left(t0)
    u[0, -1] = u_right(t0)

    for m in range(M-1):
        curr = m % 2
        next_ = 1 - curr

        for n in range(1, N):
            d2 = (u[curr, n+1] - 2*u[curr, n] + u[curr, n-1]) / h**2
            d1 = (u[curr, n+1] - u[curr, n-1]) / (2*h)
            u[next_, n] = (u[curr, n] +
                           eps * tau * d2 +
                           tau * u[curr, n] * d1 +
                           tau * u[curr, n]**3)

        u[next_, 0] = u_left(t[m+1])
        u[next_, -1] = u_right(t[m+1])

        if m % max(1, (M-1)//10) == 0:
            print(f"  Прогресс: {m/(M-1)*100:5.1f}% | max|u| = {np.max(np.abs(u[curr])):.3f}")

    elapsed = time.time() - start
    print(f"ГОТОВО за {elapsed:.2f} сек | M={M} (tau={tau:.2e})")
    return u[(M-1) % 2], x, t[-1]

if __name__ == "__main__":
    u_final, x, T = solve_serial()
    print(f"Решение в t={T:.3f}: min={u_final.min():.3f}, max={u_final.max():.3f}")
