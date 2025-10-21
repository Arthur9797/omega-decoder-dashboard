import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# --- Constants ---
hbar = 1.0545718e-34  # Planck constant (J·s)
c = 3e8               # Speed of light (m/s)
G = 6.67430e-11       # Gravitational constant (m³/kg/s²)
M = 5.972e24          # Mass of Earth (kg)
m0 = 9.11e-31         # Electron mass (kg)

# --- Schwarzschild radius ---
# Spatial curvature approximation from general relativity
r_s = 2 * G * M / c**2  # Schwarzschild radius (m)

# --- Spatial grid ---
x = np.linspace(-1e-6, 1e-6, 1000)  # Position space (m)
dx = x[1] - x[0]

# --- Initial wave packet ---
# Gaussian wave packet centered at x0 with momentum k0
x0 = 0
sigma = 1e-7
k0 = 1e10
psi0 = np.exp(-(x - x0)**2 / (2 * sigma**2)) * np.exp(1j * k0 * x)
psi0 /= np.sqrt(np.sum(np.abs(psi0)**2))  # Normalize wavefunction

# --- Gravitational potential φ(x) from Schwarzschild metric ---
# Approximates spatial curvature using Newtonian potential
def phi(x):
    r = np.abs(x) + r_s  # Avoid singularity at r = 0
    return -G * M / r

# --- Time evolution parameters ---
dt = 1e-18  # Time step (s)
steps = 500  # Number of time steps

# --- Gravitational redshift effect on ω ---
# ω' = ω * sqrt(1 + 2φ/c²), slowing quantum oscillations near massive bodies
V_grav = m0 * phi(x)
omega_grav = np.sqrt((hbar * k0)**2 / m0**2 + (m0 * c**2)**2) / hbar * np.sqrt(1 + 2 * V_grav / c**2)

# --- Quantum tunneling barrier ---
# Gaussian-shaped potential barrier simulating tunneling
barrier = np.exp(-((x - 2e-7)**2) / (2 * (2e-8)**2)) * 1e-18
V_total = V_grav + barrier  # Total potential includes gravity + barrier

# --- Time evolution loop ---
# ψ(x, t) = ψ(x, 0) * exp(-i V(x) t / ħ), simplified Schrödinger evolution
psi = psi0.copy()
psi_t = [psi0.copy()]

for _ in range(steps):
    phase = np.exp(-1j * V_total * dt / hbar)
    psi = psi * phase
    psi_t.append(psi.copy())

# --- Streamlit UI ---
st.title("Quantum Wave Packet in Curved Spacetime")

frame = st.slider("Select time frame", 0, len(psi_t) - 1, 0)

# --- Gravitational lensing effect ---
# Simulates magnification of probability density near massive object
lensing = 1 + 0.5 * np.exp(-((x - 2e-7)**2) / (2 * (1e-7)**2))
y_plot = np.abs(psi_t[frame])**2 * lensing

fig, ax = plt.subplots()
ax.plot(x, y_plot)
ax.set_xlabel("Position (m)")
ax.set_ylabel("Probability Density")
ax.set_title(f"Frame {frame}")
st.pyplot(fig)
