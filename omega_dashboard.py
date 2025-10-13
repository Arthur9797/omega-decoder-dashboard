import streamlit as st
import numpy as np

# === Fundamental Constants ===
G = 6.67430e-11         # Gravitational constant (m^3/kg/s^2)
hbar = 1.054571817e-34  # Reduced Planck constant (J·s)
c = 3e8                 # Speed of light (m/s)
kB = 1.380649e-23       # Boltzmann constant (J/K)
pi = np.pi

# === Decoder Function ===
def decode_universe(Omega):
    # Cosmological constant
    Lambda = (G * hbar * c**5) / (kB**2 * Omega)

    # Horizon temperature
    T_Lambda = c * np.sqrt(Lambda) / kB

    # Horizon radius
    R_horizon = np.sqrt(3 / Lambda)

    # Horizon area
    A_horizon = 4 * pi * R_horizon**2

    # Horizon entropy (Bekenstein-Hawking)
    S_horizon = (kB * A_horizon) / (4 * G * hbar / c**3)

    # Energy density (vacuum)
    rho_vac = (Lambda * c**2) / (8 * pi * G)

    # Information capacity
    I_bits = np.log2(Omega)

    # Curvature scale
    curvature_radius = 1 / np.sqrt(Lambda)

    # Epoch classification
    if Omega < 1e10:
        epoch = "Inflationary origin — quantum coherent, low entropy"
        fate = "Rapid expansion, symmetry breaking"
    elif Omega < 1e60:
        epoch = "Matter-dominated era — structure formation"
        fate = "Entropy accumulation, galaxy evolution"
    elif Omega < 1e120:
        epoch = "Dark energy era — accelerated expansion"
        fate = "Dilution of matter, horizon isolation"
    else:
        epoch = "Thermodynamic saturation — cosmic heat death"
        fate = "Entropy maximum, quantum freeze-out"

    return {
        "Ω (Input)": Omega,
        "Λ (Cosmological Constant)": Lambda,
        "TΛ (Horizon Temperature)": T_Lambda,
        "R_horizon (Horizon Radius)": R_horizon,
        "A_horizon (Horizon Area)": A_horizon,
        "S_horizon (Horizon Entropy)": S_horizon,
        "ρ_vac (Vacuum Energy Density)": rho_vac,
        "I_bits (Information Capacity)": I_bits,
        "Curvature Radius": curvature_radius,
        "Epoch": epoch,
        "Predicted Cosmic Fate": fate
    }

# === Streamlit UI ===
st.set_page_config(page_title="Ω Decoder Dashboard", layout="wide")
st.title("Ω Decoder Dashboard")
st.markdown("""
Explore the thermodynamic architecture of the universe by adjusting the cosmic constant Ω.  
This dashboard decodes Ω into physical parameters, entropy, curvature, and cosmic fate.
""")

Omega = st.slider("Select Ω", min_value=1e0, max_value=1e130, value=1e120, format="%.1e")

results = decode_universe(Omega)

st.subheader("Decoded Cosmological Parameters")
for key, value in results.items():
    if isinstance(value, float):
        st.write(f"**{key}**: {value:.3e}")
    else:
        st.write(f"**{key}**: {value}")

# === Scientific Notes ===
st.markdown("---")
st.markdown("### 🔬 Scientific Notes")
st.markdown("""
- **Λ (Cosmological Constant)**: Governs the vacuum energy density and expansion rate of spacetime.  
- **TΛ (Horizon Temperature)**: The temperature associated with the cosmological horizon, derived from de Sitter thermodynamics.  
- **S_horizon**: Entropy stored on the horizon surface, proportional to its area — a key insight from black hole thermodynamics.  
- **ρ_vac**: Energy density of empty space, driving accelerated expansion.  
- **Curvature Radius**: Sets the scale of spacetime curvature — smaller values imply stronger curvature.  
- **I_bits**: Total information capacity of the observable universe, in bits.  
- **Epoch & Fate**: Interprets Ω to classify the cosmic era and predict long-term evolution.
""")

# Particle masses (GeV)
m_H = 125.1       # Higgs
m_t = 173.0       # Top quark
m_nu = 0.0001     # Neutrino (approx)

# Vacuum energy density from Higgs field (simplified)
rho_vac_particles = (m_H**4 + m_t**4 + m_nu**4) * 1e9  # in J/m³

# Predicted Λ from QFT
Lambda_QFT = (8 * pi * G * rho_vac_particles) / c**2

# Discrepancy
Delta_Lambda = Lambda_QFT - Lambda

# Epoch particles
if Omega < 1e10:
    dominant_particles = "Inflaton field, quantum fluctuations"
elif Omega < 1e60:
    dominant_particles = "Photons, baryons, neutrinos"
elif Omega < 1e120:
    dominant_particles = "Dark energy, relic neutrinos"
else:
    dominant_particles = "Vacuum modes, horizon states"

