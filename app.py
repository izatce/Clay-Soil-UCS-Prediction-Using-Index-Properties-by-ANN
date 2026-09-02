import streamlit as st

# Page config
st.set_page_config(page_title="Pump Power Calculator", layout="centered")

# Title and short description
st.title("⚙️ Pump Power Calculator")
st.markdown(
    "This app calculates the **hydraulic power (kW)** required to pump a fluid.\n\n"
    "Formula:  P = ρ · g · Q · H / η  (where η is decimal efficiency)"
)

# Input area
st.subheader("📥 Input Parameters")
col1, col2 = st.columns(2)

with col1:
    flow_unit = st.selectbox("Flow unit", ("m³/s", "L/s", "m³/h"))
    if flow_unit == "m³/s":
        flow_rate = st.number_input("Flow Rate Q (m³/s)", min_value=0.0, value=0.05, format="%.4f", key="flow_m3s")
    elif flow_unit == "L/s":
        flow_lps = st.number_input("Flow Rate Q (L/s)", min_value=0.0, value=50.0, format="%.3f", key="flow_lps")
        flow_rate = flow_lps / 1000.0  # convert L/s -> m³/s
    else:  # m³/h
        flow_m3h = st.number_input("Flow Rate Q (m³/h)", min_value=0.0, value=180.0, format="%.3f", key="flow_m3h")
        flow_rate = flow_m3h / 3600.0  # convert m³/h -> m³/s

    head = st.number_input("Head H (m)", min_value=0.0, value=10.0, format="%.2f", key="head")

with col2:
    density = st.number_input("Fluid Density ρ (kg/m³)", min_value=0.1, value=1000.0, format="%.1f", key="density")
    efficiency_percent = st.number_input(
        "Pump Efficiency η (%)", min_value=0.1, max_value=100.0, value=75.0, format="%.2f", key="eff"
    )

g = 9.81  # gravity (m/s^2)

# Compute on button press
if st.button("Calculate"):
    # Basic validation
    if efficiency_percent <= 0:
        st.error("Efficiency must be greater than 0%.")
    elif flow_rate <= 0:
        st.error("Flow rate must be greater than 0.")
    elif head < 0:
        st.error("Head cannot be negative.")
    elif density <= 0:
        st.error("Density must be greater than 0.")
    else:
        try:
            efficiency = efficiency_percent / 100.0
            power_watts = (density * g * flow_rate * head) / efficiency
            power_kw = power_watts / 1000.0
            bhp = power_watts / 745.699872  # 1 HP = 745.699872 W

            st.subheader("📊 Output")
            st.success(f"Hydraulic Power: {power_kw:.3f} kW  ({power_watts:.2f} W)")
            st.write(f"Brake horsepower (HP): {bhp:.3f} hp")
            st.write(f"Inputs used: Q = {flow_rate:.6f} m³/s, H = {head:.3f} m, ρ = {density:.1f} kg/m³, η = {efficiency_percent:.2f}%")
        except Exception as e:
            st.error(f"Calculation error: {e}")

st.markdown("---")
st.caption("Developed for Civil/Mechanical Engineering applications.")
