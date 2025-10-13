Python 3.9.5 (tags/v3.9.5:0a7dcbd, May  3 2021, 17:27:52) [MSC v.1928 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
>>> import streamlit as st
import numpy as np

# === Constants ===
G = 6.67430e-11         # Gravitational constant
hbar = 1.054571817e-34  # Reduced Planck constant
c = 3e8                 # Speed of light
kB = 1.380649e-23       # Boltzmann constant

# === Decoder Function ===
def decode_universe(Omega):
    Lambda = (G * hbar * c**5) / (kB**2 * Omega)
    T_Lambda = c * (Lambda**0.5) / kB
    S = Omega * kB
    I = np.log2(Omega)

    if Omega < 1e10:
        outcome = "Inflationary origin — low entropy, quantum coherent"
    elif Omega < 1e60:
        outcome = "Matter-dominated era — entropy growing"
    elif Omega < 1e120:
        outcome = "Dark energy era — accelerating expansion"
    else:
        outcome = "Thermodynamic saturation — cosmic heat death"

    return Lambda, T_Lambda, S, I, outcome

# === Streamlit UI ===
st.title("Ω Decoder Dashboard")
st.markdown("Explore the thermodynamic evolution of the universe by adjusting the cosmic constant Ω.")

Omega = st.slider("Select Ω", min_value=1e0, max_value=1e130, value=1e120, format="%.1e")

Lambda, T_Lambda, S, I, outcome = decode_universe(Omega)

st.subheader("Decoded Parameters")
st.write(f"Λ (Cosmological Constant): {Lambda:.2e} 1/m²")
st.write(f"TΛ (Horizon Temperature): {T_Lambda:.2e} K")
st.write(f"Entropy (S): {S:.2e} kB units")
st.write(f"Information Capacity: {I:.2f} bits")
st.write(f"Predicted Outcome: **{outcome}**")