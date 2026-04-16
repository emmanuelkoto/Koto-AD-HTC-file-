# Written by KOTO EMMANUEL
# MSc MECHANICAL ENGINEERING STUDENT, UNIVERSITY OF LAGOS
# Integration: Thermal Barrier Coating (TBC) Optimization + Methane Tracking + Thermodynamic Analysis

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# --- Page Config & Styling ---
st.set_page_config(page_title="KOTO HTC Reactor: TBC + Methane Optimizer", layout="wide")
st.title("🌿 HydroThermal Carbonization Reactor: KOTO 259044028 (TBC + Methane Suite)")

# --- Constants & Physics ---
nx, ny = 100, 30
L, R = 5.0, 0.7
dx, dy = L/nx, R/ny
dt = 0.005  
rho_slurry = 1030.0  
area = np.pi * (R**2) 
k_steel = 50.0  # W/mK (Standard Reactor Steel)

# Initialize state
if 'T' not in st.session_state:
    st.session_state.T = np.ones((nx, ny)) * 40.0
    st.session_state.conv = np.zeros((nx, ny))
    st.session_state.ch4 = np.zeros((nx, ny)) 
    st.session_state.history = []

def vectorized_step(T, conv, ch4, u, a, t_ext, tin, r_barrier):
    """
    Vectorized simulation step with TBC boundary condition
    """
    T_new = T.copy()
    conv_new = conv.copy()
    ch4_new = ch4.copy()
    
    # Physics calculations (Advection-Diffusion)
    adv_x = u * (T[1:-1, 1:-1] - T[:-2, 1:-1]) / dx
    diff_x = a * (T[2:, 1:-1] - 2*T[1:-1, 1:-1] + T[:-2, 1:-1]) / dx**2
    diff_y = a * (T[1:-1, 2:] - 2*T[1:-1, 1:-1] + T[1:-1, :-2]) / dy**2
    
    # --- ACTIVE KINETICS ---
    rxn_mask = (T[1:-1, 1:-1] > 180.0) & (conv[1:-1, 1:-1] < 1.0)
    
    # Hydrochar Conversion (Solid phase)
    conv_rate = 0.01 * (T[1:-1, 1:-1] / 200.0)
    conv_new[1:-1, 1:-1] += np.where(rxn_mask, conv_rate * dt * 10, 0.0)
    
    # Methane Yield (Gas phase byproduct)
    gas_mask = (T[1:-1, 1:-1] > 200.0) & (ch4[1:-1, 1:-1] < 0.25)
    
    # --- VARYING METHANE YIELD LOGIC ---
    thermal_yield_rate = 0.005 * (T[1:-1, 1:-1] / 200.0)**2 
    active_conversion_state = conv[1:-1, 1:-1]
    dynamic_yield = 0.025 * (active_conversion_state * (1.0 - active_conversion_state))
    
    total_yield_rate = thermal_yield_rate + dynamic_yield
    ch4_new[1:-1, 1:-1] += np.where(gas_mask, total_yield_rate * dt * 10, 0.0)

    # Heat balancing
    s_rxn = np.where(rxn_mask, 6.0, 0.0)
    T_new[1:-1, 1:-1] = T[1:-1, 1:-1] + dt * (-adv_x + diff_x + diff_y + s_rxn)
    
    # --- THERMAL BARRIER BOUNDARY CONDITION (FIXED) ---
    # Calculate insulation factor (scalar)
    insulation_factor = 1.0 / (1.0 + r_barrier)
    
    # Calculate effective wall temperature for ALL wall nodes (vectorized)
    # This creates arrays of shape (nx,) for the left and right walls
    t_wall_eff_full = T[:, 0] + (t_ext - T[:, 0]) * insulation_factor

    # Apply Boundary Conditions
    T_new[0, :] = tin  # Inlet (x=0)
    T_new[-1, :] = T_new[-2, :]  # Outlet (x=L) - zero gradient
    T_new[:, 0] = t_wall_eff_full  # Left wall (r=0) - with TBC
    T_new[:, -1] = t_wall_eff_full  # Right wall (r=R) - with TBC
    
    return T_new, conv_new, ch4_new

def render_plots(T_data, conv_data, ch4_data, container):
    """Render the three main simulation plots"""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
    
    im1 = ax1.imshow(T_data.T, extent=[0, L, 0, R], origin='lower', cmap='magma', aspect='auto', vmin=40, vmax=350)
    ax1.set_title("Temperature Profile (°C)")
    plt.colorbar(im1, ax=ax1)
    
    im2 = ax2.imshow(conv_data.T, extent=[0, L, 0, R], origin='lower', cmap='Greens', aspect='auto', vmin=0, vmax=1)
    ax2.set_title("Hydrochar Conversion Yield")
    plt.colorbar(im2, ax=ax2)

    im3 = ax3.imshow(ch4_data.T, extent=[0, L, 0, R], origin='lower', cmap='Blues', aspect='auto', vmin=0, vmax=0.20)
    ax3.set_title("Active Methane (CH₄) Gaseous Yield")
    plt.colorbar(im3, ax=ax3)
    
    plt.tight_layout()
    container.pyplot(fig)
    plt.close(fig)

# --- Sidebar Controls ---
st.sidebar.header("🕹️ Reactor Controls")
u_vel = st.sidebar.slider("Slurry Velocity [m/s]", 0.01, 2.0, 0.10)
t_external = st.sidebar.slider("External Heat Source [°C]", 180.0, 450.0, 250.0)
alpha = st.sidebar.slider("Thermal Diffusivity", 0.002, 0.06, 0.01)
t_inlet = st.sidebar.number_input("Slurry Inlet Temp [°C]", value=40.0)

st.sidebar.header("🛡️ Thermal Barrier Optimization")
tbc_thickness = st.sidebar.slider("TBC Thickness [mm]", 0.0, 50.0, 5.0) / 1000.0  # Convert to meters
tbc_k = st.sidebar.slider("TBC Conductivity [W/mK]", 0.01, 2.0, 0.2)  # Low k = better insulation

# Calculate Resistance: R = thickness / k
r_total = (0.02 / k_steel) + (tbc_thickness / tbc_k)  # 20mm steel + TBC
st.sidebar.metric("Total Thermal Resistance", f"{r_total:.4f} m²K/W")

sim_steps = st.sidebar.number_input("Simulation Iterations", min_value=100, max_value=5000, value=1000, step=50)
mass_flow_rate = rho_slurry * area * u_vel
st.sidebar.metric("Mass Flow Rate", f"{mass_flow_rate:.2f} kg/s")

col_btn1, col_btn2 = st.sidebar.columns(2)

# --- Main Display Area ---
plot_spot = st.empty() 
metrics_spot = st.empty()

# Run Simulation logic (shared for both buttons)
def run_sim(is_animated, run_type):
    st.session_state.T = np.ones((nx, ny)) * t_inlet
    st.session_state.conv = np.zeros((nx, ny))
    st.session_state.ch4 = np.zeros((nx, ny))
    
    progress_bar = st.progress(0) if is_animated else None
    
    for i in range(sim_steps):
        st.session_state.T, st.session_state.conv, st.session_state.ch4 = vectorized_step(
            st.session_state.T, st.session_state.conv, st.session_state.ch4, 
            u_vel, alpha, t_external, t_inlet, r_total
        )
        if is_animated and i % 100 == 0:
            render_plots(st.session_state.T, st.session_state.conv, st.session_state.ch4, plot_spot)
            if progress_bar:
                progress_bar.progress(i / sim_steps)
    
    if progress_bar:
        progress_bar.empty()
    
    # Log the run
    avg_ch4_yield = np.mean(st.session_state.ch4) * 100
    st.session_state.history.append({
        "Run Type": run_type,
        "TBC Thickness (mm)": tbc_thickness * 1000,
        "TBC Conductivity": tbc_k,
        "Wall Temp (°C)": t_external,
        "Velocity (m/s)": u_vel,
        "Thermal Diffusivity": alpha,
        "Inlet Temp (°C)": t_inlet,
        "Avg CH₄ Yield (%)": round(avg_ch4_yield, 4)
    })

if col_btn1.button("▶️ Live Simulation"):
    with st.spinner("Processing thermochemical reactions with TBC (Animated)..."):
        run_sim(True, "Animated")
    render_plots(st.session_state.T, st.session_state.conv, st.session_state.ch4, plot_spot)

if col_btn2.button("⚡ RUN"):
    with st.spinner("Calculating final state with TBC optimization..."):
        run_sim(False, "Instant")
    render_plots(st.session_state.T, st.session_state.conv, st.session_state.ch4, plot_spot)

# --- Current Metrics Display ---
peak_t = np.max(st.session_state.T)
heat_loss_reduction = (1 - (1/(1+r_total))) * 100  # Stylized efficiency metric
current_avg_ch4 = np.mean(st.session_state.ch4) * 100

with metrics_spot.container():
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Thermal Resistance", f"{r_total:.3f} m²K/W")
    m2.metric("Peak Temp", f"{peak_t:.1f} °C")
    m3.metric("Thermal Efficiency", f"{heat_loss_reduction:.1f} %")
    m4.metric("Avg Conversion", f"{np.mean(st.session_state.conv)*100:.1f} %")
    m5.metric("Avg CH₄ Yield", f"{current_avg_ch4:.3f} %")

# --- Simulation History ---
st.markdown("---")
st.subheader("📋 Methane Yield & TBC Performance Log")
if st.session_state.history:
    st.table(st.session_state.history[-10:])  # Show last 10 runs
else:
    st.info("No runs recorded yet. Start a simulation to populate the log.")

# --- Comprehensive Analysis Report ---
st.markdown("---")
if st.button("📊 Comprehensive Optimization & Thermodynamic Analysis"):
    st.markdown("## 📑 Optimization & Thermodynamic Report: KOTO 259044028")
    
    # Calculate energy saved (Theoretical)
    q_uninsulated = (t_external - t_inlet) / (0.02 / k_steel)
    q_insulated = (t_external - t_inlet) / r_total
    energy_saved = q_uninsulated - q_insulated
    
    # Methane mass flow calculation
    avg_ch4 = current_avg_ch4
    gas_kg_s = (avg_ch4/100) * mass_flow_rate
    
    # Two-column layout for TBC and Methane analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("#### 🛡️ Thermal Barrier Impact")
        st.write(f"""
        - **TBC Thickness:** {tbc_thickness*1000:.1f} mm
        - **TBC Conductivity:** {tbc_k:.3f} W/mK
        - **Total Thermal Resistance:** {r_total:.4f} m²K/W
        - **Heat Flux Reduction:** {energy_saved:.1f} W/m²
        - **Thermal Efficiency Gain:** {heat_loss_reduction:.1f}%
        """)
        
        st.success("#### Economic Insight")
        st.write(f"""
        Higher thermal resistance allows for {((q_uninsulated - q_insulated)/q_uninsulated*100):.1f}% 
        reduction in external heating power to achieve the same hydrothermal conversion.
        """)
    
    with col2:
        st.info("#### 💨 Methane Generation Analysis")
        st.write(f"""
        - **Average CH₄ Yield:** {current_avg_ch4:.3f}%
        - **Methane Mass Flow:** {gas_kg_s:.5f} kg/s
        - **Peak Temperature:** {peak_t:.1f}°C
        - **Heat Required:** {(mass_flow_rate * 4.18 * (peak_t-t_inlet)):.1f} W
        """)
        
        st.success("#### Process Efficiency")
        st.write(f"""
        At external wall temperature of {t_external}°C with TBC, 
        the methane generation is maximized in the optimal conversion zone.
        """)
    
    # --- Thermodynamic Diagrams ---
    st.markdown("### 🔥 Thermodynamic Cycle Analysis")
    
    T_min_k = t_inlet + 273.15
    T_max_k = peak_t + 273.15
    
    fig_thermo, (ax_ts, ax_th) = plt.subplots(1, 2, figsize=(14, 5))

    # T-S Diagram (Steam Cycle approximation)
    s_points = [1.0, 2.5, 6.0, 7.5, 1.0] 
    t_points = [T_min_k, 373.15, 373.15, T_max_k, T_min_k] 
    
    ax_ts.plot(s_points, t_points, 'o-', color='crimson', linewidth=2, label='Ideal Steam Cycle')
    ax_ts.fill(s_points, t_points, color='crimson', alpha=0.1)
    ax_ts.set_xlabel("Entropy (s) [kJ/kg·K]")
    ax_ts.set_ylabel("Temperature (T) [K]")
    ax_ts.set_title("T-S Diagram (Steam Phase)")
    ax_ts.grid(True, linestyle='--', alpha=0.6)
    ax_ts.legend()

    # T-H Diagram (Gas Phase / Methane Heating approximation)
    cp_ch4 = 2.22 
    temps_th = np.linspace(T_min_k, T_max_k, 50)
    h_th = cp_ch4 * (temps_th - T_min_k)
    
    ax_th.plot(temps_th, h_th, '-', color='teal', linewidth=3, label='$CH_4$ Enthalpy Curve')
    ax_th.fill_between(temps_th, 0, h_th, color='teal', alpha=0.1)
    ax_th.set_xlabel("Temperature (T) [K]")
    ax_th.set_ylabel("Enthalpy Change (Δh) [kJ/kg]")
    ax_th.set_title("T-H Diagram (Gas Cycle)")
    ax_th.grid(True, linestyle='--', alpha=0.6)
    ax_th.legend()

    st.pyplot(fig_thermo)
    
    # --- Detailed Written Report ---
    st.markdown("## 📋 Detailed Engineering Analysis")
    
    rep_col1, rep_col2 = st.columns(2)
    
    with rep_col1:
        st.markdown("#### 1. T-S (Temperature-Entropy) Analysis")
        st.write(f"""
        The T-S diagram represents the **Steam Rankine Cycle** coupled with the HTC reactor.
        - **Isentropic Compression:** Effective pumping into the pressure vessel
        - **Heat Addition Phase:** Phase transition at constant temperature
        - **Process Insight:** At peak temperature of **{peak_t:.1f}°C**, significant energy availability for heat recovery
        - **TBC Effect:** Insulation maintains higher entropy generation for same heat input
        """)
        
        st.markdown("#### 2. Thermal Barrier Performance")
        st.write(f"""
        - **Without TBC:** Heat flux = {q_uninsulated:.1f} W/m²
        - **With TBC:** Heat flux = {q_insulated:.1f} W/m²
        - **Savings:** {energy_saved:.1f} W/m² ({((q_uninsulated - q_insulated)/q_uninsulated*100):.1f}%)
        """)
    
    with rep_col2:
        st.markdown("#### 3. T-H (Temperature-Enthalpy) Analysis")
        st.write(f"""
        The T-H diagram tracks the **Total Energy Content** of process gases.
        - **Linear Correlation:** Enthalpy ($h$) is a direct function of $T$ where $h = c_p T$
        - **Energy Density:** Slope represents specific heat capacity
        - **TBC Advantage:** Retained enthalpy enables downstream energy recovery
        """)
        
        st.markdown("#### 4. Methane Yield Optimization")
        st.write(f"""
        - **Current Yield:** {current_avg_ch4:.3f}%
        - **Mass Flow Rate:** {mass_flow_rate:.2f} kg/s
        - **CH₄ Production:** {gas_kg_s:.5f} kg/s
        """)
    
    # --- Safety and Recommendations ---
    st.markdown("### ⚠️ Safety & Design Recommendations")
    
    warning_color = "🟢" if peak_t < 250 else "🟡" if peak_t < 300 else "🔴"
    stability = "STABLE" if peak_t < 250 else "CAUTION" if peak_t < 300 else "CRITICAL"
    
    col_rec1, col_rec2, col_rec3 = st.columns(3)
    
    with col_rec1:
        st.metric("Temperature Status", f"{warning_color} {stability}", 
                 delta=f"{peak_t:.0f}°C", 
                 delta_color="inverse" if peak_t > 250 else "normal")
    
    with col_rec2:
        st.metric("TBC Effectiveness", f"{heat_loss_reduction:.1f}% reduction",
                 delta="Good" if heat_loss_reduction > 30 else "Moderate")
    
    with col_rec3:
        st.metric("Reaction Zone", "Active" if peak_t > 180 else "Inactive",
                 delta="Optimal" if 200 < peak_t < 280 else "Check")
    
    st.markdown("---")
    st.markdown("#### 🛠️ Optimization Recommendations")
    
    if tbc_thickness > 0.03:
        st.warning("⚠️ **High TBC Thickness Detected:** Consider reducing thickness to avoid excessive core temperatures and potential hot spots near inlet.")
    elif tbc_thickness < 0.002:
        st.info("ℹ️ **Low TBC Thickness:** Increasing TBC thickness could improve thermal efficiency by 20-40%.")
    else:
        st.success("✅ **TBC Thickness Optimal:** Current configuration provides good insulation without risking thermal runaway.")
    
    if u_vel < 0.05:
        st.warning("⚠️ **Low Slurry Velocity:** May cause uneven temperature distribution. Consider increasing velocity to improve mixing.")
    elif u_vel > 1.0:
        st.info("ℹ️ **High Slurry Velocity:** Residence time may be insufficient for complete conversion. Consider reducing velocity.")
    
    if peak_t > 300:
        st.error("🔥 **CRITICAL:** Peak temperature exceeds 300°C. Risk of material degradation and excessive pressure. Reduce external temperature or increase TBC thickness.")
    elif peak_t < 180:
        st.error("❄️ **INSUFFICIENT:** Peak temperature below 180°C. Hydrothermal reactions will not initiate. Increase external temperature.")
    else:
        st.success(f"✓ **OPERATIONAL:** Temperature range ({peak_t:.0f}°C) suitable for HTC process.")
    
    # --- Energy Balance Summary ---
    st.markdown("### 📊 Energy Balance Summary")
    
    energy_input = mass_flow_rate * 4.18 * (peak_t - t_inlet)
    energy_saved_total = energy_saved * area
    
    energy_col1, energy_col2, energy_col3 = st.columns(3)
    with energy_col1:
        st.metric("Total Heat Input", f"{energy_input/1000:.2f} kW")
    with energy_col2:
        st.metric("Energy Saved via TBC", f"{energy_saved_total/1000:.2f} kW")
    with energy_col3:
        st.metric("CO₂ Reduction Estimate", f"{energy_saved_total * 0.0002:.3f} kg/s", 
                 help="Estimated CO₂ savings based on natural gas displacement")

# --- Footer ---
st.markdown("---")
st.markdown("*Developed by KOTO EMMANUEL | MSc Mechanical Engineering | University of Lagos*")
st.markdown("*Integration: TBC Optimization + Methane Tracking + Thermodynamic Analysis*")