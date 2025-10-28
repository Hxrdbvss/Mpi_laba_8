# scatter_gather.py
from mpi4py import MPI
import numpy as np
import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# === УСТОЙЧИВЫЕ ПАРАМЕТРЫ ===
eps = 10**(-1.5)
N = 400
a, b = 0.0, 1.0
t0, T_target = 0.0, 0.6  # ← ТЕСТ: 0.6 сек

if rank == 0:
    h = (b - a) / N
    tau = 0.8 * h**2 / (2 * eps)
    M = int((T_target - t0) / tau) + 1
    T = (M - 1) * tau
    print(f"УСТОЙЧИВО: N={N}, M={M}, tau={tau:.2e}, T={T:.3f}")
else:
    h = tau = M = T = None

h = comm.bcast(h, root=0)
tau = comm.bcast(tau, root=0)
M = comm.bcast(M, root=0)
T = comm.bcast(T, root=0)

# Распределение
ave, res = divmod(N + 1, size)
rcounts = np.array([ave + 1 if i < res else ave for i in range(size)], dtype=np.int32)
displs = np.cumsum(np.hstack(([0], rcounts[:-1])), dtype=np.int32)
N_local = rcounts[rank]

# Локальные массивы
u = np.zeros((2, N_local + 2))
x_local = np.linspace(a + displs[rank]*h, a + (displs[rank] + N_local - 1)*h, N_local)

def u_init(x): return np.sin(3*np.pi*(x - 1/6))
def u_left(t): return -1.0
def u_right(t): return 1.0

u[0, 1:-1] = u_init(x_local)
if rank == 0: u[0, 0] = u_left(t0)
if rank == size - 1: u[0, -1] = u_right(t0)

if rank == 0:
    start_time = MPI.Wtime()

for m in range(M-1):
    curr = m % 2
    next_ = 1 - curr

    # Обмен ореолами
    if size > 1:
        if rank > 0:
            comm.Sendrecv(u[curr, 1:2], dest=rank-1, sendtag=0,
                          recvbuf=u[curr, 0:1], source=rank-1, recvtag=0)
        if rank < size - 1:
            comm.Sendrecv(u[curr, -2:-1], dest=rank+1, sendtag=0,
                          recvbuf=u[curr, -1:], source=rank+1, recvtag=0)

    for n in range(1, N_local + 1):
        d2 = (u[curr, n+1] - 2*u[curr, n] + u[curr, n-1]) / h**2
        d1 = (u[curr, n+1] - u[curr, n-1]) / (2*h)
        u[next_, n] = u[curr, n] + eps*tau*d2 + tau*u[curr, n]*d1 + tau*u[curr, n]**3

    if rank == 0: u[next_, 0] = u_left(t0 + (m+1)*tau)
    if rank == size - 1: u[next_, -1] = u_right(t0 + (m+1)*tau)

    if rank == 0 and m % max(1, (M-1)//10) == 0:
        print(f"  SG Прогресс: {m/(M-1)*100:5.1f}%")

last_local = u[(M-1) % 2, 1:-1]
u_final = np.empty(N + 1) if rank == 0 else None
comm.Gatherv([last_local, N_local, MPI.DOUBLE],
             [u_final, rcounts, displs, MPI.DOUBLE], root=0)

if rank == 0:
    elapsed = MPI.Wtime() - start_time
    print(f"Scatter/Gather: {elapsed:.2f} сек")
    np.save("solution_sg.npy", u_final)

MPI.Finalize()
