import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# --- Constants ---
hbar = 1.0545718e-34  # Planck constant (J·s)
c = 3e8               # Speed of light (m/s)
G = 6.67430e-11       # Gravitational constant (m³/kg/s²)
M = 5.972e24          # Mass of Earth (kg)
m0 = 9.11e-31         # Electron mass (kg)

# --- Schwarzschild radius ---
r_s = 2 * G * M / c**2

# --- Spatial grid ---
x = np.linspace(-1e-6, 1e-6, 1000)
dx = x[1] - x[0]

# --- Initial wave packet ---
x0 = 0
sigma = 1e-7
k0 = 1e10
psi0 = np.exp(-(x - x0)**2 / (2 * sigma**2)) * np.exp(1j * k0 * x)
psi0 /= np.sqrt(np.sum(np.abs(psi0)**2))  # Normalize

# --- Gravitational potential φ(x) from Schwarzschild metric ---
def phi(x):
    r = np.abs(x) + r_s  # Avoid singularity
    return -G * M / r

# --- Time evolution parameters ---
dt = 1e-18
steps = 500

# --- Precompute potential and curvature effects ---
V_grav = m0 * phi(x)
omega_grav = np.sqrt((hbar * k0)**2 / m0**2 + (m0 * c**2)**2) / hbar * np.sqrt(1 + 2 * V_grav / c**2)

# --- Quantum tunneling barrier ---
barrier = np.exp(-((x - 2e-7)**2) / (2 * (2e-8)**2)) * 1e-18
V_total = V_grav + barrier

# --- Time evolution loop ---
psi = psi0.copy()
psi_t = [psi0.copy()]

for _ in range(steps):
    phase = np.exp(-1j * V_total * dt / hbar)
    psi = psi * phase
    psi_t.append(psi.copy())

# --- Animation: Gravitational Lensing Effect ---
fig, ax = plt.subplots()
line, = ax.plot(x, np.abs(psi_t[0])**2)
ax.set_ylim(0, np.max(np.abs(psi0)**2)*1.2)
ax.set_title("Quantum Wave Packet in Curved Spacetime")
ax.set_xlabel("Position (m)")
ax.set_ylabel("Probability Density")

def update(frame):
    lensing = 1 + 0.5 * np.exp(-((x - 2e-7)**2) / (2 * (1e-7)**2))  # Simulated lensing magnification
    line.set_ydata(np.abs(psi_t[frame])**2 * lensing)
    return line,

ani = animation.FuncAnimation(fig, update, frames=len(psi_t), interval=30, blit=True)
plt.show()
