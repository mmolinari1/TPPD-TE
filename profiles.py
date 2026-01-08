import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ==========================================
# 1. Total Demand & Baseload Calculation
# ==========================================

HOURS_PER_YEAR = 8000

# Annual Demand
annual_demand = {
    "Electric_kWh": 15_500_000,
    "Thermal_smc": 2_700_000
}

# Thermal conversion factor
KWH_PER_SMC = 10.6

# Calculate Thermal Annual kWh
annual_demand["Thermal_kWh"] = annual_demand["Thermal_smc"] * KWH_PER_SMC

# Baseload Power Calculation (Average Power)
# Power (kW) = Energy (kWh) / Hours (h)
baseload_power = {
    "Electric_kW": annual_demand["Electric_kWh"] / HOURS_PER_YEAR,
    "Thermal_kW": annual_demand["Thermal_kWh"] / HOURS_PER_YEAR
}

print("--- Total Annual Demand ---")
print(f"Electric: {annual_demand['Electric_kWh']:,.0f} kWh/y")
print(f"Thermal:  {annual_demand['Thermal_smc']:,.0f} smc/y ({annual_demand['Thermal_kWh']:,.0f} kWh/y)")
print("\n--- Calculated Baseload Power (on 8000h) ---")
print(f"Electric: {baseload_power['Electric_kW']:.2f} kW")
print(f"Thermal:  {baseload_power['Thermal_kW']:.2f} kW")


# ==========================================
# 2. Process Data Definitions
# ==========================================

data = [
    {
        "Process": "Drying Raw Material",
        "Electric_pct": 2,
        "Thermal_pct": 15,
        "Time_Window": "00:00-24:00",
        "Note": "Continuous hot air flow"
    },
    {
        "Process": "Cutting",
        "Electric_pct": 10,
        "Thermal_pct": 0,
        "Time_Window": "06:00-14:00",
        "Note": "Mechanical grinders, 1st shift"
    },
    {
        "Process": "Solid-Liquid Extraction",
        "Electric_pct": 15,
        "Thermal_pct": 20,
        "Time_Window": "06:00-22:00",
        "Note": "Heating solvent tanks"
    },
    {
        "Process": "Concentration (Evap.)",
        "Electric_pct": 15,
        "Thermal_pct": 45,
        "Time_Window": "00:00-24:00",
        "Note": "Thin-film evaporators"
    },
    {
        "Process": "Freeze-Drying (Lyoph.)",
        "Electric_pct": 45,
        "Thermal_pct": 10,
        "Time_Window": "00:00-24:00",
        "Note": "High electric demand (refrigeration)"
    },
    {
        "Process": "Sludge & Depuration",
        "Electric_pct": 5,
        "Thermal_pct": 5,
        "Time_Window": "06:00-22:00",
        "Note": "Treatment of machine sludge"
    },
    {
        "Process": "Facility OPEX & HVAC",
        "Electric_pct": 8,
        "Thermal_pct": 5,
        "Time_Window": "06:00-22:00",
        "Note": "Lighting, HVAC"
    }
]

df_processes = pd.DataFrame(data)

print("\n--- Daily Process Subdivision ---")
print(df_processes[["Process", "Electric_pct", "Thermal_pct", "Time_Window"]])

# Verify Percentages Sum
print(f"\nTotal Electric %: {df_processes['Electric_pct'].sum()}%")
print(f"Total Thermal %:  {df_processes['Thermal_pct'].sum()}%")

# ==========================================
# 3. Daily Energy & Hourly Distribution
# ==========================================

# Total Daily Energy (kWh/day) based on Baseload Power * 24h
# This assumes the "Baseload" provided is the average power over the active days
total_daily_electric_kWh = baseload_power["Electric_kW"] * 24
total_daily_thermal_kWh = baseload_power["Thermal_kW"] * 24

print("\n--- Daily Energy Calculation ---")
print(f"Total Daily Electric Energy: {total_daily_electric_kWh:,.0f} kWh")
print(f"Total Daily Thermal Energy:  {total_daily_thermal_kWh:,.0f} kWh")

# Prepare hourly arrays (0 to 23)
hours = np.arange(24)
hourly_profiles_el = pd.DataFrame(index=hours)
hourly_profiles_th = pd.DataFrame(index=hours)

def parse_window(window_str):
    """Parses 'HH:MM-HH:MM' into start and end hour integers."""
    start_str, end_str = window_str.split('-')
    start = int(start_str.split(':')[0])
    end = int(end_str.split(':')[0])
    if end == 0 and end_str != '00:00': # Handle 24:00 case if written as such, though usually 00:00
         end = 24
    if end == 0 and start == 0: # 00:00-24:00 convention
         end = 24
    return start, end

# Calculate Power Profiles
for _, row in df_processes.iterrows():
    proc_name = row["Process"]
    
    # Parse Time Window
    start, end = parse_window(row["Time_Window"])
    duration = end - start
    if duration <= 0: duration += 24 # Handle overnight if needed, primarily 0-24
    
    # Electric Calculation
    el_energy = total_daily_electric_kWh * (row["Electric_pct"] / 100)
    el_power = el_energy / duration if duration > 0 else 0
    
    # Thermal Calculation
    th_energy = total_daily_thermal_kWh * (row["Thermal_pct"] / 100)
    th_power = th_energy / duration if duration > 0 else 0
    
    # Create profile arrays
    profile_el = np.zeros(24)
    profile_th = np.zeros(24)
    
    # Fill active hours
    # Note: simple integer hour handling. 
    # If 06:00-14:00, it means hours 6,7,8,9,10,11,12,13 (Total 8 hours)
    if start < end:
        profile_el[start:end] = el_power
        profile_th[start:end] = th_power
    else: # Crosses midnight
        profile_el[start:] = el_power
        profile_el[:end] = el_power
        profile_th[start:] = th_power
        profile_th[:end] = th_power
        
    hourly_profiles_el[proc_name] = profile_el
    hourly_profiles_th[proc_name] = profile_th

# Add Total Column
hourly_profiles_el["Total_kW"] = hourly_profiles_el.sum(axis=1)
hourly_profiles_th["Total_kW"] = hourly_profiles_th.sum(axis=1)

print("\n--- Hourly Profiles Calculated ---")
print("Electric Peak (kW):", hourly_profiles_el["Total_kW"].max())
print("Thermal Peak (kW): ", hourly_profiles_th["Total_kW"].max())

# Preview Distribution
print("\nSample Electric Profile (Process Power in kW):")
print(hourly_profiles_el.head())

# ==========================================
# 4. Verification
# ==========================================

print("\n--- Verification: Integral Check ---")

# Electric Verification
calc_total_el = hourly_profiles_el["Total_kW"].sum() * 1 # kW * 1h
diff_el = calc_total_el - total_daily_electric_kWh
print(f"Electric: Calculated {calc_total_el:,.2f} kWh vs Target {total_daily_electric_kWh:,.2f} kWh")
print(f"Diff: {diff_el:.2f} kWh ({diff_el/total_daily_electric_kWh*100:.4f}%) -> {'OK' if abs(diff_el) < 1 else 'ERROR'}")

# Thermal Verification
calc_total_th = hourly_profiles_th["Total_kW"].sum() * 1 # kW * 1h
diff_th = calc_total_th - total_daily_thermal_kWh
print(f"Thermal:  Calculated {calc_total_th:,.2f} kWh vs Target {total_daily_thermal_kWh:,.2f} kWh")
print(f"Diff: {diff_th:.2f} kWh ({diff_th/total_daily_thermal_kWh*100:.4f}%) -> {'OK' if abs(diff_th) < 1 else 'ERROR'}")


# ==========================================
# 5. Plotting
# ==========================================

def plot_profile(df_profiles, title, ylabel, total_daily_val, filename):
    # Dropping 'Total_kW' for plotting individual stack components
    plot_data = df_profiles.drop(columns=["Total_kW"])
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Stacked Area Plot
    ax.stackplot(plot_data.index, plot_data.T, labels=plot_data.columns, alpha=0.8)
    
    # Plot Total Line on top
    ax.plot(df_profiles.index, df_profiles["Total_kW"], color='black', linewidth=2, linestyle='--', label='Total Demand')
    
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Hour of Day", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xlim(0, 23)
    ax.set_xticks(np.arange(0, 24))
    ax.grid(True, alpha=0.3)
    
    # Move legend outside
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    
    plt.tight_layout()
    
    # Ensure output directory exists
    output_dir = "plots"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, dpi=300)
    print(f"Saved plot to: {save_path}")
    # plt.show() # Commented out to avoid blocking if running non-interactively, or generally just save. 

# Plot Electric
plot_profile(hourly_profiles_el, 
             f"Daily Electric Demand Profile\nTotal: {total_daily_electric_kWh:,.0f} kWh/day", 
             "Power (kW)", 
             total_daily_electric_kWh,
             "electric_profile.png")

# Plot Thermal
plot_profile(hourly_profiles_th, 
             f"Daily Thermal Demand Profile\nTotal: {total_daily_thermal_kWh:,.0f} kWh/day", 
             "Power (kW)", 
             total_daily_thermal_kWh,
             "thermal_profile.png")

