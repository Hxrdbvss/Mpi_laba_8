# plot_results.py
import matplotlib.pyplot as plt
import numpy as np
import os

# Создаём папку images/
os.makedirs("images", exist_ok=True)

# === ДАННЫЕ ИЗ ТВОИХ ЗАМЕРОВ ===
procs = [1, 2, 4, 8, 16]

# Последовательная: 65.39 сек (T=6.0, N=400)
T_serial = 65.39

# Cart+Sendrecv (T=6.0)
T_cart = [65.39, 43.40, 40.47, 26.67, 58.48]

# Scatter/Gather (T=0.6) — НЕ ИСПОЛЬЗУЕМ ДЛЯ СРАВНЕНИЯ
# T_sg = [65.39, 4.36, 4.04, 2.80, 3.54]

# === 1. График времени выполнения ===
plt.figure(figsize=(8, 5))
plt.plot(procs, T_cart, 's-', color='tab:blue', label='Cart+Sendrecv (T=6.0)', markersize=8)
plt.axhline(T_serial, color='tab:red', linestyle='--', label='Последовательная')
plt.xlabel('Число процессов')
plt.ylabel('Время выполнения, сек')
plt.title('Время выполнения (N=400, T=6.0)')
plt.grid(True, alpha=0.3)
plt.legend()
plt.xticks(procs)
plt.tight_layout()
plt.savefig('images/execution_time.png', dpi=200)
plt.close()

# === 2. График ускорения ===
S_cart = T_serial / np.array(T_cart)

plt.figure(figsize=(8, 5))
plt.plot(procs, S_cart, 's-', color='tab:green', label='Cart+Sendrecv', markersize=8)
plt.plot(procs, procs, '--', color='tab:red', label='Идеальное ускорение')
plt.xlabel('Число процессов')
plt.ylabel('Ускорение (Speedup)')
plt.title('Ускорение (N=400, T=6.0)')
plt.grid(True, alpha=0.3)
plt.legend()
plt.xticks(procs)
plt.ylim(0, max(S_cart.max(), 3) + 1)
plt.tight_layout()
plt.savefig('images/speedup.png', dpi=200)
plt.close()

# === 3. График эффективности ===
E_cart = S_cart / np.array(procs)

plt.figure(figsize=(8, 5))
plt.plot(procs, E_cart, 'o-', color='tab:purple', label='Cart+Sendrecv', markersize=8)
plt.axhline(1.0, color='tab:red', linestyle='--', label='Идеальная эффективность')
plt.xlabel('Число процессов')
plt.ylabel('Эффективность (E = S/p)')
plt.title('Эффективность параллелизма (N=400, T=6.0)')
plt.grid(True, alpha=0.3)
plt.legend()
plt.xticks(procs)
plt.ylim(0, 1.1)
plt.tight_layout()
plt.savefig('images/efficiency.png', dpi=200)
plt.close()

print("Графики сохранены в папку images/:")
print("   images/execution_time.png")
print("   images/speedup.png")
print("   images/efficiency.png")
