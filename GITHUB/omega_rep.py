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
from scipy.integrate import simpson

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

def schwarzschild_radius(M):
    return 2 * G * M / c**2

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

def simulate_wave_packet(curvature_radius):
    x = np.linspace(-1e3, 1e3, 1000)
    sigma = curvature_radius / 10
    psi0 = np.exp(-x**2 / (2 * sigma**2)) * np.exp(1j * x / curvature_radius)
    dispersion = np.exp(-x**2 / (2 * (sigma * 1.5)**2))
    psi_t = psi0 * dispersion
    prob_density = np.abs(psi_t)**2
    return x, prob_density

def tunneling_probability(M):
    rs = schwarzschild_radius(M)
    barrier_height = G * M / rs
    energy = 0.9 * barrier_height
    prob = np.exp(-2 * np.sqrt(2 * M * (barrier_height - energy)) * rs / hbar)
    return prob

def lensing_field(mass):
    theta = np.linspace(-2, 2, 400)
    alpha = 4 * G * mass / (c**2 * theta)
    field = np.exp(-theta**2) * np.cos(10 * theta - alpha)
    return theta, field

def decoherence_profile(M):
    rs = schwarzschild_radius(M)
    r = np.linspace(rs * 1.01, rs * 10, 500)
    coherence = np.exp(-((r - rs) / rs)**2)
    return r, coherence

def entanglement_degradation(curvature_radius):
    t = np.linspace(0, 1e3, 500)
    degradation = np.exp(-t / curvature_radius)
    return t, degradation

def hawking_radiation(M):
    T_hawking = hbar * c**3 / (8 * np.pi * G * M * k)
    freq = np.linspace(1e11, 1e14, 500)
    spectrum = (freq**3) / (np.exp(hbar * freq / (k * T_hawking)) - 1)
    return freq, spectrum, T_hawking

# === Streamlit UI ===
st.set_page_config(page_title="Ω Quantum–Relativity Dashboard", layout="wide")
st.title("Ω Quantum–Relativity Dashboard")

Omega = st.slider("Select Ω", min_value=1e0, max_value=1e130, value=1e120, format="%.1e")
mass_slider = st.slider("Mass of Gravitational Well (kg)", min_value=1e24, max_value=1e32, value=1e30, format="%.1e")
lensing_mass = st.slider("Lensing Mass (kg)", min_value=1e28, max_value=1e32, value=1e30, format="%.1e")
curvature_radius_override = st.slider("Override Curvature Radius (m)", min_value=1e1, max_value=1e6, value=1e3, format="%.1e")

results = decode_universe(Omega)

st.subheader("Decoded Cosmological Parameters")
for key, value in results.items():
    if key != "Particle Energy":
        st.write(f"**{key}**: {value:.3e}" if isinstance(value, float) else f"**{key}**: {value}")

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

st.subheader("🌊 Wave Packet Dispersion in Curved Spacetime")
x, prob_density = simulate_wave_packet(curvature_radius_override)
fig2, ax2 = plt.subplots(figsize=(3, 1.8))
ax2.plot(x, prob_density, color='darkgreen')
ax2.set_title("Wave Packet Dispersion")
ax2.set_xlabel("Position (m)")
ax2.set_ylabel("Probability Density")
st.pyplot(fig2)

st.subheader("🕳️ Quantum Tunneling Near Gravitational Well")
tunnel_prob = tunneling_probability(mass_slider)
st.write(f"**Tunneling Probability** near Schwarzschild radius: {tunnel_prob:.3e}")

st.subheader("🔭 Gravitational Lensing of Quantum Field")
theta, field = lensing_field(lensing_mass)
fig3, ax3 = plt.subplots(figsize=(3, 1.8))
ax3.plot(theta, field, color='purple')
ax3.set_title("Gravitational Lensing Effect")
ax3.set_xlabel("Angular Position θ")
ax3.set_ylabel("Field Intensity")
st.pyplot(fig3)

st.subheader("🧨 Quantum Decoherence Near Event Horizon")
r, coherence = decoherence_profile(mass_slider)
fig4, ax4 = plt.subplots(figsize=(3, 1.8))
ax4.plot(r, coherence, color='orange')
ax4.set_title("Decoherence Profile")
ax4.set_xlabel("Radial Distance (m)")
ax4.set_ylabel("Coherence")
st.pyplot(fig4)

st.subheader("🔗 Entanglement Degradation in Curved Spacetime")
t, degradation = entanglement_degradation(curvature_radius_override)
fig5, ax5 = plt.subplots(figsize=(3, 1.8))
ax5.plot(t, degradation, color='red')
ax5.set_title("Entanglement Degradation")
ax5.set_xlabel("Time (s)")
ax5.set_ylabel("Entanglement Strength")
st.pyplot(fig5)

st.subheader("🔥 Hawking Radiation Spectrum")
freq, spectrum, T_hawking = hawking_radiation(mass_slider)
fig6, ax6 = plt.subplots(figsize=(3, 1.8))
ax6.plot(freq, spectrum, color='black')
ax6.set_title(f"Hawking Radiation (T = {T_hawking:.2e} K)")
ax6.set_xlabel("Frequency (Hz)")
ax6.set_ylabel("Spectral Intensity")
st.pyplot(fig6)

st.markdown("---")
st.markdown("### 🧠 Scientific Notes")
st.markdown("""
- **Ω** links quantum entropy to spacetime curvature via Λ  
- **Schwarzschild radius** defines gravitational wells and tunneling barriers  
- **Wave packet dispersion** shows how quantum fields stretch in curved spacetime  
- **Tunneling probability** estimates quantum escape near black holes  
- **Gravitational lensing** distorts quantum fields around massive objects  
- **Decoherence** increases near event horizons due to curvature-induced phase loss  
- **Entanglement** weakens over time in curved spacetime due to information leakage  
- **Hawking radiation** emits particles from black holes, with temperature inversely proportional to mass  
""")





