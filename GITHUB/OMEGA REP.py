# === Required packages for deployment ===
# streamlit
# numpy
# matplotlib
# plotly
# scipy

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.constants import G, hbar, c, k
from scipy.integrate import simps

# === Particle Masses (GeV) ===
masses = {
    "Higgs": 125.1,
    "Top Quark": 173.0,
    "Neutrino": 0.0001,
    "Electron": 0.511,
    "Photon": 0.0,
    "Gluon": 0.0
}

gev_to_joule_m3 = 1.7827e-36 * (1e9)**4

# === Schwarzschild Geometry ===
def schwarzschild_radius(M):
    return 2 * G * M / c**2

def gravitational_potential(r, M):
    rs = schwarzschild_radius(M)
    return -G * M / r * (1 - rs / r)

# === Quantum–Relativity Bridge via Ω ===
def decode_universe(Omega):
    Lambda = (G * hbar * c**5) / (k**2 * Omega)
    T_Lambda = c * np.sqrt(Lambda) / k
    R_horizon = np.sqrt(3 / Lambda)
    A_horizon = 4 * np.pi * R_horizon**2
    S_horizon = (k * A_horizon) / (4 * G * hbar / c**3)
    rho_vac = (Lambda * c**2) / (8 * np.pi * G)
    I_bits = np.log2(Omega)
    curvature_radius = 1 / np.sqrt(Lambda)

    particle_energy = {name: (mass**4 * gev_to_joule_m3) for name, mass in masses.items()}
    rho_vac_particles = sum(particle_energy.values())
    Lambda_QFT = (8 * np.pi * G * rho_vac_particles) / c**2
    Delta_Lambda = Lambda_QFT - Lambda

    if Omega < 1e10:
        epoch = "Inflationary origin — quantum coherent, low entropy"
    elif Omega < 1e60:
        epoch = "Matter-dominated era — structure formation"
    elif Omega < 1e120:
        epoch = "Dark energy era — accelerated expansion"
    else:
        epoch = "Thermodynamic saturation — cosmic heat death"

    return {
        "Ω": Omega,
        "Λ": Lambda,
        "TΛ": T_Lambda,
        "R_horizon": R_horizon,
        "A_horizon": A_horizon,
        "S_horizon": S_horizon,
        "ρ_vac": rho_vac,
        "I_bits": I_bits,
        "Curvature Radius": curvature_radius,
        "Λ_QFT": Lambda_QFT,
        "ΔΛ": Delta_Lambda,
        "Epoch": epoch,
        "Particle Energy": particle_energy
    }

# === Wave Packet Dispersion in Curved Spacetime ===
def simulate_wave_packet(curvature_radius):
    x = np.linspace(-1e3, 1e3, 1000)
    sigma = curvature_radius / 10
    psi0 = np.exp(-x**2 / (2 * sigma**2)) * np.exp(1j * x / curvature_radius)
    dispersion = np.exp(-x**2 / (2 * (sigma * 1.5)**2))
    psi_t = psi0 * dispersion
    prob_density = np.abs(psi_t)**2
    return x, prob_density

# === Quantum Tunneling Near Gravitational Wells ===
def tunneling_probability(M):
    rs = schwarzschild_radius(M)
    barrier_height = G * M / rs
    energy = 0.9 * barrier_height
    prob = np.exp(-2 * np.sqrt(2 * M * (barrier_height - energy)) * rs / hbar)
    return prob

# === Gravitational Lensing Animation ===
def lensing_field():
    theta = np.linspace(-2, 2, 400)
    mass = 1e30
    alpha = 4 * G * mass / (c**2 * theta)
    field = np.exp(-theta**2) * np.cos(10 * theta - alpha)
    return theta, field

# === Quantum Decoherence Near Event Horizon ===
def decoherence_profile(M):
    rs = schwarzschild_radius(M)
    r = np.linspace(rs * 1.01, rs * 10, 500)
    coherence = np.exp(-((r - rs) / rs)**2)
    return r, coherence

# === Entanglement Degradation in Curved Spacetime ===
def entanglement_degradation(curvature_radius):
    t = np.linspace(0, 1e3, 500)
    degradation = np.exp(-t / curvature_radius)
    return t, degradation

# === Hawking Radiation Simulation ===
def hawking_radiation(M):
    T_hawking = hbar * c**3 / (8 * np.pi * G * M * k)
    freq = np.linspace(1e11, 1e14, 500)
    spectrum = (freq**3) / (np.exp(hbar * freq / (k * T_hawking)) - 1)
    return freq, spectrum, T_hawking

# === Streamlit UI ===
st.set_page_config(page_title="Ω Quantum–Relativity Dashboard", layout="wide")
st.title("Ω Quantum–Relativity Dashboard")
st.markdown("Explore the quantum structure of spacetime through Ω, Schwarzschild curvature, and wave dynamics.")

Omega = st.slider("Select Ω", min_value=1e0, max_value=1e130, value=1e120, format="%.1e")
mass_slider = st.slider("Mass of Gravitational Well (kg)", min_value=1e24, max_value=1e32, value=1e30, format="%.1e")

results = decode_universe(Omega)

# === Display Parameters ===
st.subheader("Decoded Cosmological Parameters")
for key, value in results.items():
    if key != "Particle Energy":
        st.write(f"**{key}**: {value:.3e}" if isinstance(value, float) else f"**{key}**: {value}")

# === Particle Vacuum Energy Plot ===
st.subheader("🔬 Particle Vacuum Energy Contributions")
fig = go.Figure()
fig.add_trace(go.Bar(
    x=list(results["Particle Energy"].keys()),
    y=list(results["Particle Energy"].values()),
    marker_color='indigo'
))
fig.update_layout(
    yaxis_title="Vacuum Energy Density (J/m³)",
    xaxis_title="Particle",
    title="Quantum Field Contributions to Vacuum Energy",
    template="plotly_white"
)
st.plotly_chart(fig, use_container_width=True)

# === Wave Packet Dispersion Plot ===
st.subheader("🌊 Wave Packet Dispersion in Curved Spacetime")
x, prob_density = simulate_wave_packet(results["Curvature Radius"])
fig2, ax2 = plt.subplots()
ax2.plot(x, prob_density, color='darkgreen')
ax2.set_title("Wave Packet Dispersion")
ax2.set_xlabel("Position (m)")
ax2.set_ylabel("Probability Density")
st.pyplot(fig2)

# === Quantum Tunneling Probability ===
st.subheader("🕳️ Quantum Tunneling Near Gravitational Well")
tunnel_prob = tunneling_probability(mass_slider)
st.write(f"**Tunneling Probability** near Schwarzschild radius: {tunnel_prob:.3e}")

# === Gravitational Lensing Animation ===
st.subheader("🔭 Gravitational Lensing of Quantum Field")
theta, field = lensing_field()
fig3, ax3 = plt.subplots()
ax3.plot(theta, field, color='purple')
ax3.set_title("Gravitational Lensing Effect on Quantum Field")
ax3.set_xlabel("Angular Position θ")
ax3.set_ylabel("Field Intensity")
st.pyplot(fig3)

# === Quantum Decoherence Near Horizon ===
st.subheader("🧨 Quantum Decoherence Near Event Horizon")
r, coherence = decoherence_profile(mass_slider)
fig4, ax4 = plt.subplots()
ax4.plot(r, coherence, color='orange')
ax4.set_title("Decoherence Profile Near Schwarzschild Radius")
ax4.set_xlabel("Radial Distance (m)")
ax4.set_ylabel("Coherence")
st.pyplot(fig4)

# === Entanglement Degradation ===
st.subheader("🔗 Entanglement Degradation in Curved Spacetime")
t, degradation = entanglement_degradation(results["Curvature Radius"])
fig5, ax5 = plt.subplots()
ax5.plot(t, degradation, color='red')
ax5.set_title("Entanglement Degradation Over Time")
ax5.set_xlabel("Time (s

