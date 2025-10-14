# === Required packages for deployment ===
# streamlit
# numpy
# matplotlib
# plotly

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# === Fundamental Constants ===
G = 6.67430e-11
hbar = 1.054571817e-34
c = 3e8
kB = 1.380649e-23
pi = np.pi

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

# === Cosmological Models ===
def lambda_model(t, model):
    if model == "ΛCDM":
        return 1e-12 * np.exp(-t / 1e17) + 1e-120
    elif model == "Inflationary":
        return 1e-6 * np.exp(-t / 1e15) + 1e-120
    elif model == "Bouncing":
        return 1e-12 * np.exp(-np.abs(t - 1e17) / 1e17) + 1e-120
    elif model == "Cyclic":
        return 1e-12 * (1 + 0.1 * np.sin(t / 1e17)) + 1e-120
    elif model == "Phantom Energy":
        return 1e-120 * np.exp(t / 1e17)
    else:
        return 1e-12 * np.exp(-t / 1e17) + 1e-120

# === Decoder Function ===
def decode_universe(Omega):
    Lambda = (G * hbar * c**5) / (kB**2 * Omega)
    T_Lambda = c * np.sqrt(Lambda) / kB
    R_horizon = np.sqrt(3 / Lambda)
    A_horizon = 4 * pi * R_horizon**2
    S_horizon = (kB * A_horizon) / (4 * G * hbar / c**3)
    rho_vac = (Lambda * c**2) / (8 * pi * G)
    I_bits = np.log2(Omega)
    curvature_radius = 1 / np.sqrt(Lambda)

    particle_energy = {name: (mass**4 * gev_to_joule_m3) for name, mass in masses.items()}
    rho_vac_particles = sum(particle_energy.values())
    Lambda_QFT = (8 * pi * G * rho_vac_particles) / c**2
    Delta_Lambda = Lambda_QFT - Lambda

    if Omega < 1e10:
        epoch = "Inflationary origin — quantum coherent, low entropy"
        fate = "Rapid expansion, symmetry breaking"
        dominant_particles = "Inflaton field, quantum fluctuations"
    elif Omega < 1e60:
        epoch = "Matter-dominated era — structure formation"
        fate = "Entropy accumulation, galaxy evolution"
        dominant_particles = "Photons, baryons, neutrinos"
    elif Omega < 1e120:
        epoch = "Dark energy era — accelerated expansion"
        fate = "Dilution of matter, horizon isolation"
        dominant_particles = "Dark energy, relic neutrinos"
    else:
        epoch = "Thermodynamic saturation — cosmic heat death"
        fate = "Entropy maximum, quantum freeze-out"
        dominant_particles = "Vacuum modes, horizon states"

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
        "Dominant Particles": dominant_particles,
        "Cosmic Fate": fate,
        "Particle Energy": particle_energy
    }

# === Streamlit UI ===
st.set_page_config(page_title="Ω Decoder Dashboard", layout="wide")
st.title("Ω Decoder Dashboard")
st.markdown("Explore the thermodynamic and quantum structure of the universe by adjusting the cosmic constant Ω and selecting cosmological models.")

# === Model Toggle ===
model = st.selectbox("Choose Cosmological Model", ["ΛCDM", "Inflationary", "Bouncing", "Cyclic", "Phantom Energy"])

# === Ω Slider ===
Omega = st.slider("Select Ω", min_value=1e0, max_value=1e130, value=1e120, format="%.1e")
results = decode_universe(Omega)

# === Display Parameters ===
st.subheader("Decoded Cosmological Parameters")
for key, value in results.items():
    if key != "Particle Energy":
        if isinstance(value, float):
            st.write(f"**{key}**: {value:.3e}")
        else:
            st.write(f"**{key}**: {value}")

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

# === Ω Evolution Simulation ===
st.subheader("📈 Ω Evolution Across Cosmic Time")
t = np.logspace(-35, 40, 500)
Lambda_t = lambda_model(t, model)
Omega_t = (G * hbar * c**5) / (kB**2 * Lambda_t)

fig2, ax = plt.subplots(figsize=(10, 5))
ax.loglog(t, Omega_t, color='blue')
ax.set_xlabel("Time since Big Bang (s)")
ax.set_ylabel("Ω(t)")
ax.set_title(f"Ω Evolution in {model} Model")
ax.grid(True, which="both", ls="--")
st.pyplot(fig2)

# === Scientific Notes ===
st.markdown("---")
st.markdown("### 🧠 Scientific Notes")
st.markdown("""
- **Λ**: Cosmological constant from Ω, governs expansion and vacuum energy  
- **TΛ**: Horizon temperature from de Sitter thermodynamics  
- **S_horizon**: Entropy encoded on the horizon surface  
- **ρ_vac**: Vacuum energy density from Λ  
- **Λ_QFT**: Predicted Λ from quantum fields (Higgs, top quark, neutrinos, etc.)  
- **ΔΛ**: Discrepancy between quantum field prediction and observed Λ — the cosmological constant problem  
- **Ω(t)**: Tracks entropy and information growth across cosmic epochs  
- **Model Toggle**: Switch between cosmological scenarios to explore different futures  
- **Particle Energy Plot**: Shows how each field contributes to vacuum energy  
""")
