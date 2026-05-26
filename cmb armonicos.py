# -*- coding: utf-8 -*-
"""
Created on Tue May 26 10:44:15 2026

@author: valez
"""

# -*- coding: utf-8 -*-


import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.special import sph_harm as scipy_sph_harm
from matplotlib.colors import TwoSlopeNorm

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def sph_harm(m, l, phi, theta):
    return scipy_sph_harm(m, l, phi, theta)

plt.rcParams.update({
    "font.family": "serif", "font.size": 11,
    "axes.titlesize": 13, "axes.labelsize": 11,
    "figure.dpi": 150
})

CMAP = "RdBu_r"          
NPHI, NTHETA = 500, 250  

phi   = np.linspace(0, 2*np.pi, NPHI)     
theta = np.linspace(0,   np.pi, NTHETA)  
PHI, THETA = np.meshgrid(phi, theta)

def combo_Ylm(terms):
    field = np.zeros_like(PHI, dtype=complex)
    for l, m, c in terms:
        field += c * sph_harm(m, l, PHI, THETA)
    return np.real(field)



# FIGURA 1 – Armónicos individuales representativos


cases_indiv = [
    (2,  0, "Cuadrupolo $Y_2^0$\n(dipolo en latitud)"),
    (2,  2, r"Cuadrupolo $Y_2^2$\n(patrón azimutal $\times$2)"), # CORREGIDO: r"..."
    (10, 0, r"Multipolo $\ell=10$, $m=0$\n(bandas latitudinales)"), # CORREGIDO: r"..."
    (10, 5, r"Multipolo $\ell=10$, $m=5$\n(estructura mixta)"), # CORREGIDO: r"..."
    (30, 0, r"Multipolo $\ell=30$, $m=0$\n(escalas angulares pequeñas)"), # CORREGIDO: r"..."
    (30, 15, r"Multipolo $\ell=30$, $m=15$\n(escalas angulares pequeñas)"), # CORREGIDO: r"..."
]

fig1, axes1 = plt.subplots(2, 3, figsize=(16, 9),
                           subplot_kw={"projection": "mollweide"})
fig1.suptitle(r"Armónicos Esféricos Individuales $Y_\ell^m(\theta,\phi)$" "\n"
              r"Proyección de Mollweide — Análogos a fluctuaciones del CMB",
              fontsize=14, fontweight="bold", y=1.02)

for ax, (l, m, title) in zip(axes1.flat, cases_indiv):
    data = combo_Ylm([(l, m, 1.0)])
    vmax = np.abs(data).max()
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    lon = phi - np.pi                
    lat = np.pi/2 - theta            
    LON, LAT = np.meshgrid(lon, lat)
    im = ax.pcolormesh(LON, LAT, data, cmap=CMAP, norm=norm, rasterized=True)
    ax.set_title(title, pad=6)
    ax.set_xticklabels([]); ax.set_yticklabels([])
    ax.grid(color="k", linewidth=0.3, alpha=0.4)
    plt.colorbar(im, ax=ax, orientation="horizontal", pad=0.05, shrink=0.7,
                 label=r"$\Delta T/T$") # CORREGIDO: r"..."

fig1.tight_layout()
fig1.savefig(os.path.join(OUTPUT_DIR, "fig1_individuales.png"), bbox_inches="tight", dpi=150)



# FIGURA 2 – Combinaciones: multipolos BAJOS vs ALTOS


low_terms = [
    (2,  0,  1.0), (2,  1,  0.6), (2,  2,  0.4),
    (3,  0,  0.8), (3,  1,  0.5), (3,  3,  0.3),
]
low_field = combo_Ylm(low_terms)

high_terms = []
np.random.seed(42)
for l in range(20, 31):
    for m in range(0, l+1, 3):
        c = np.random.uniform(0.3, 1.0) * (1 if np.random.rand() > 0.5 else -1)
        high_terms.append((l, m, c))
high_field = combo_Ylm(high_terms)

all_terms = low_terms + high_terms
all_field  = combo_Ylm(all_terms)

lon = phi - np.pi
lat = np.pi/2 - theta
LON, LAT = np.meshgrid(lon, lat)

fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5),
                           subplot_kw={"projection": "mollweide"})
fig2.suptitle("Comparación: Multipolos Bajos vs Altos\n"
              "Analogía con el Mapa de Temperatura del CMB",
              fontsize=14, fontweight="bold")

panels = [
    (low_field,  r"Multipolos Bajos ($\ell=2,3$)" "\nEstructura a gran escala angular"), # CORREGIDO: r"..."
    (high_field, r"Multipolos Altos ($\ell=20$–$30$)" "\nEstructura a pequeña escala angular"), # CORREGIDO: r"..."
    (all_field,  r"Suma Total (bajos + altos)" "\nAnisotropía compuesta del CMB"), # CORREGIDO: r"..."
]

for ax, (data, title) in zip(axes2, panels):
    vmax = np.abs(data).max()
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    im = ax.pcolormesh(LON, LAT, data, cmap=CMAP, norm=norm, rasterized=True)
    ax.set_title(title, pad=6)
    ax.set_xticklabels([]); ax.set_yticklabels([])
    ax.grid(color="k", linewidth=0.3, alpha=0.4)
    cb = plt.colorbar(im, ax=ax, orientation="horizontal", pad=0.05, shrink=0.75)
    cb.set_label(r"$\Delta T/T$ (u.a.)") # CORREGIDO: r"..."

fig2.tight_layout()
fig2.savefig(os.path.join(OUTPUT_DIR, "fig2_bajo_vs_alto.png"), bbox_inches="tight", dpi=150)



# FIGURA 3 – Perfiles ecuatoriales


theta_eq = np.pi / 2
phi_vals  = np.linspace(0, 2*np.pi, 1000)

low_eq  = np.real(sum(sph_harm(m, l, phi_vals, theta_eq) * c for l, m, c in low_terms))
high_eq = np.real(sum(sph_harm(m, l, phi_vals, theta_eq) * c for l, m, c in high_terms))

fig3, axes3 = plt.subplots(1, 2, figsize=(16, 6))
fig3.suptitle(r"Perfiles Térmicos Ecuatoriales ($\theta = \pi/2$)" "\n"
              r"Simulación de Variaciones Angulares del CMB", fontsize=14, fontweight="bold") # CORREGIDO: r"..." y saltos de línea \n estándar

ax_b = axes3[0]
ax_b.plot(np.degrees(phi_vals), low_eq, color="darkorange", lw=2)
ax_b.axhline(0, color="k", lw=0.8, ls="--")
ax_b.fill_between(np.degrees(phi_vals), low_eq, 0, where=(low_eq > 0), alpha=0.25, color="red",   label="Caliente")
ax_b.fill_between(np.degrees(phi_vals), low_eq, 0, where=(low_eq < 0), alpha=0.25, color="blue",  label="Frío")
ax_b.set_xlabel(r"Longitud $\phi$ (grados)") # CORREGIDO: r"..."
ax_b.set_ylabel(r"$\Delta T/T$ (u.a.)")     # CORREGIDO: r"..."
ax_b.set_title(r"Multipolos Bajos ($\ell=2,3$)" "\nVariaciones lentas / gran escala angular", fontweight="bold") # CORREGIDO: r"..."
ax_b.legend(fontsize=9)
ax_b.grid(alpha=0.3)
ax_b.set_xlim(0, 360)

ax_h = axes3[1]
ax_h.plot(np.degrees(phi_vals), high_eq, color="steelblue", lw=1.5)
ax_h.axhline(0, color="k", lw=0.8, ls="--")
ax_h.fill_between(np.degrees(phi_vals), high_eq, 0, where=(high_eq > 0), alpha=0.25, color="red",  label="Caliente")
ax_h.fill_between(np.degrees(phi_vals), high_eq, 0, where=(high_eq < 0), alpha=0.25, color="blue", label="Frío")
ax_h.set_xlabel(r"Longitud $\phi$ (grados)") # CORREGIDO: r"..."
ax_h.set_ylabel(r"$\Delta T/T$ (u.a.)")     # CORREGIDO: r"..."
ax_h.set_title(r"Multipolos Altos ($\ell=20$–$30$)" "\nVariaciones rápidas / pequeña escala angular", fontweight="bold") # CORREGIDO: r"..."
ax_h.legend(fontsize=9)
ax_h.grid(alpha=0.3)
ax_h.set_xlim(0, 360)

fig3.tight_layout()
fig3.savefig(os.path.join(OUTPUT_DIR, "fig3_espectro_perfiles.png"), bbox_inches="tight", dpi=150)
