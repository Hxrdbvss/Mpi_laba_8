# cart_sendrecv.py — УСТОЙЧИВАЯ ВЕРСИЯ
from mpi4py import MPI
import numpy as np
import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# === УСТОЙЧИВЫЕ ПАРАМЕТРЫ ===
eps = 10**(-1.5)
N_global = 400
a, b = 0.0, 1.0
t0, T_target = 0.0, 6.0

if rank == 0:
    h = (b - a) / N_global
    tau = 0.8 * h**2 / (2 * eps)
    M = int((T_target - t0) / tau) + 1
    T = (M - 1) * tau
    print(f"УСТОЙЧИВО: N={N_global}, M={M}, h={h:.2e}, tau={tau:.2e}, T={T:.3f}")
else:
    h = tau = M = T = None

# Рассылка
N_global = comm.bcast(N_global, root=0)
M = comm.bcast(M, root=0)
h = comm.bcast(h, root=0)
tau = comm.bcast(tau, root=0)
T = comm.bcast(T, root=0)
eps = comm.bcast(eps, root=0)

# Виртуальная топология
comm_cart = comm.Create_cart([size], periods=[False], reorder=True)
rank_cart = comm_cart.Get_rank()
left, right = comm_cart.Shift(0, 1)

# Распределение точек
ave, res = divmod(N_global + 1, size)
counts = np.array([ave + 1 if i < res else ave for i in range(size)], dtype=np.int32)
displs = np.cumsum(np.hstack(([0], counts[:-1])), dtype=np.int32)

N_local = counts[rank]
x_local = np.linspace(a + displs[rank] * h, a + (displs[rank] + N_local - 1) * h, N_local)

# Локальный массив: 2 слоя + ореолы
u = np.zeros((2, N_local + 2))

# Инициализация
def u_init(x): return np.sin(3 * np.pi * (x - 1/6))
def u_left(t): return -1.0
def u_right(t): return 1.0

u[0, 1:-1] = u_init(x_local)
if rank == 0: u[0, 0] = u_left(t0)
if rank == size - 1: u[0, -1] = u_right(t0)

comm.Barrier()
if rank == 0:
    start_time = MPI.Wtime()

# Основной цикл
for m in range(M-1):
    curr = m % 2
    next_ = 1 - curr

    # Обмен ореолами
    if left != MPI.PROC_NULL:
        comm_cart.Sendrecv(u[curr, 1:2], dest=left, sendtag=0,
                           recvbuf=u[curr, 0:1], source=left, recvtag=1)
    if right != MPI.PROC_NULL:
        comm_cart.Sendrecv(u[curr, -2:-1], dest=right, sendtag=1,
                           recvbuf=u[curr, -1:], source=right, recvtag=0)

    # Вычисления
    for n in range(1, N_local + 1):
        d2 = (u[curr, n+1] - 2*u[curr, n] + u[curr, n-1]) / h**2
        d1 = (u[curr, n+1] - u[curr, n-1]) / (2*h)
        u[next_, n] = u[curr, n] + eps*tau*d2 + tau*u[curr, n]*d1 + tau*u[curr, n]**3

    # Граничные условия
    if rank == 0:
        u[next_, 0] = u_left(t0 + (m+1)*tau)
    if rank == size - 1:
        u[next_, -1] = u_right(t0 + (m+1)*tau)

    # Прогресс (только root, редко)
    if rank == 0 and m % max(1, (M-1)//10) == 0:
        print(f"  Прогресс: {m/(M-1)*100:5.1f}% | max|u| = {np.max(np.abs(u[curr])):.3f}")

# Сбор
last_local = u[(M-1) % 2, 1:-1]
u_final = np.empty(N_global + 1) if rank == 0 else None
comm.Gatherv(last_local, [u_final, counts, displs, MPI.DOUBLE], root=0)

if rank == 0:
    elapsed = MPI.Wtime() - start_time
    print(f"Cart+Sendrecv: {elapsed:.2f} сек | M={M}")
    np.save("solution_cart.npy", u_final)

MPI.Finalize()
