# -*- coding: utf-8 -*-
"""
Created on Mon May 25 20:38:41 2026

@author: valez
"""

# -*- coding: utf-8 -*-
"""
Created on Mon May 25 20:05:14 2026

@author: valez
"""

"""
Funciones de Neumann Nn(x) — Soluciones de la Ecuación de Bessel

"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.special import yn, yv          # yn = Neumann orden entero
from scipy.signal import argrelmin, argrelmax


output_dir = "outputs"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


plt.rcParams.update({
    "font.family": "serif", "font.size": 11,
    "axes.titlesize": 12, "axes.labelsize": 11,
    "figure.dpi": 150
})

COLORS  = ["#e03131", "#1971c2", "#2f9e44"]
LABELS  = [r"$N_0(x)$", r"$N_1(x)$", r"$N_2(x)$"]
LSTYLES = ["-", "--", "-."]

x_full = np.linspace(0.01, 20, 8000)   
x_fine = np.linspace(0.01, 20, 10000)  


fig1, ax1 = plt.subplots(figsize=(13, 6))
fig1.suptitle(
    r"Funciones de Neumann $N_n(x)$ — Soluciones de la ecuación de Bessel"
    "\n"
    r"$x^2 y'' + x y' + (x^2 - n^2)\,y = 0$",
    fontsize=13, fontweight="bold"
)

for n in range(3):
    vals = yn(n, x_full)
    
    vals_plot = np.where(np.abs(vals) > 5, np.nan, vals)
    ax1.plot(x_full, vals_plot, color=COLORS[n], lw=2,
             linestyle=LSTYLES[n], label=LABELS[n])

ax1.axhline(0, color="black", lw=0.8, ls=":")
ax1.set_xlim(0.01, 20)
ax1.set_ylim(-1.5, 1.0)
ax1.set_xlabel("$x$", fontsize=13)
ax1.set_ylabel("$N_n(x)$", fontsize=13)
ax1.set_title("Comportamiento global en $0.01 \\leq x \\leq 20$", fontsize=12)
ax1.legend(fontsize=12, loc="lower right")
ax1.grid(True, alpha=0.35)

# Anotaciones de divergencia en x→0+
for n, yann in zip([0, 1, 2], [0.8, 0.6, 0.4]):
    ax1.annotate(
        f"$N_{n}(x) \\to -\\infty$\ncuando $x \\to 0^+$",
        xy=(0.3, -1.4), xytext=(1.5 + n*1.8, -1.1 + n*0.1),
        arrowprops=dict(arrowstyle="->", color=COLORS[n], lw=1.2),
        color=COLORS[n], fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=COLORS[n], alpha=0.8)
    )

fig1.tight_layout()

fig1.savefig(os.path.join(output_dir, "neumann_fig1_global.png"), bbox_inches="tight", dpi=150)




print("\n" + "═"*60)
print(" PRIMEROS CEROS DE N₀(x), N₁(x), N₂(x)")
print("═"*60)
for n in range(3):
    v   = yn(n, x_fine)
    idx = np.where(np.diff(np.sign(v)))[0][:4] 
    zeros_n = []
    for i in idx:
        xz = x_fine[i] + (x_fine[i+1]-x_fine[i]) * \
             abs(v[i]) / (abs(v[i]) + abs(v[i+1]))
        zeros_n.append(round(xz, 4))
    print(f"  N_{n}(x) = 0  en: {zeros_n}")
print("═"*60)
