import cantera as ct
import numpy as np
import matplotlib.pyplot as plt

# ------------------------
# DATI INPUT
# ------------------------
# Composizione secca (frazione molare equivalente)
fuel = "C:1 H:1.67 O:0.51 N:0.048"

# Aria
air = "O2:1, N2:3.76"

# Umidità
moisture = 0.112  # 11.2%

# Portate annue (kg)
mass_secco = 800
mass_umido = 259
mass_fanghi = 70
mass_tot = mass_secco + mass_umido + mass_fanghi

# LHV (J/kg)
LHV_dry = 19.04e6

# Range equivalence ratio
phi_list = np.linspace(0.5, 1.2, 100)

# Inizializza array risultati
T_flame = []
NO_ppm = []
CO_ppm = []
O2_ppm = []
CO2_ppm = []

# ------------------------
# CICLO SUL PHI
# ------------------------
for phi in phi_list:
    gas = ct.Solution("gri30.yaml")
    gas.set_equivalence_ratio(phi, fuel, air)
    
    # aggiunta acqua inerente
    Y = gas.Y
    idx_h2o = gas.species_index("H2O")
    Y = Y * (1 - moisture)
    Y[idx_h2o] += moisture
    gas.Y = Y / Y.sum()
    
    gas.TP = 298.15, ct.one_atm
    gas.equilibrate("HP")
    
    T_flame.append(gas.T)
    NO_ppm.append(gas["NO"].X[0]*1e6)
    CO_ppm.append(gas["CO"].X[0]*1e6)
    O2_ppm.append(gas["O2"].X[0]*1e6)
    CO2_ppm.append(gas["CO2"].X[0]*1e6)

# ------------------------
# GRAFICI
# ------------------------
plt.figure(figsize=(10,6))
plt.plot(phi_list, T_flame, 'r-o', label="T fiamma [K]")
plt.xlabel("Equivalence ratio φ")
plt.ylabel("Temperatura [K]")
plt.grid(True)
plt.title("Temperatura di fiamma vs equivalence ratio")
plt.legend()
plt.show()

plt.figure(figsize=(10,6))
plt.plot(phi_list, NO_ppm, 'g-o', label="NO [ppm]")
plt.plot(phi_list, CO_ppm, 'b-o', label="CO [ppm]")
plt.plot(phi_list, O2_ppm, 'orange', label="O2 [ppm]")
plt.plot(phi_list, CO2_ppm, 'r-o', label="CO2 [ppm]")
plt.xlabel("Equivalence ratio φ")
plt.ylabel("Concentrazione [ppm]")
plt.grid(True)
plt.title("Andamento dei prodotti di combustione vs equivalence ratio")
plt.legend()
plt.show()

# ------------------------
# TROVA PHI OTTIMALE (minimizza NO + CO)
# ------------------------
pollutants = np.array(NO_ppm) + np.array(CO_ppm)
idx_opt = np.argmin(pollutants)
phi_opt = phi_list[idx_opt]
print(f"Equivalence ratio ottimale (min NO+CO): φ = {phi_opt:.2f}")

# ------------------------
# SIMULAZIONE CON φ OTTIMALE
# ------------------------
gas = ct.Solution("gri30.yaml")
gas.set_equivalence_ratio(phi_opt, fuel, air)

# aggiunta acqua
Y = gas.Y
Y = Y * (1 - moisture)
Y[idx_h2o] += moisture
gas.Y = Y / Y.sum()

gas.Y = Y / Y.sum()
gas.TP = 298.15, ct.one_atm
gas.equilibrate("HP")

# ------------------------
# CONVERSIONE IN kg/anno e g/kWh
# ------------------------
# frazioni molari → moli per kg combustibile
# assumiamo 1 kg di miscela secca
molar_mass_gas = gas.mean_molecular_weight  # kg/kmol
n_total = 1.0 / molar_mass_gas  # kmol per kg combustibile

# moli specie
n_NO = n_total * gas["NO"].X[0]
n_CO = n_total * gas["CO"].X[0]
n_CO2 = n_total * gas["CO2"].X[0]

# massa kg/1 kg combustibile
m_NO = n_NO * 30.006 / 1000  # kg
m_CO = n_CO * 28.01 / 1000
m_CO2 = n_CO2 * 44.01 / 1000

# totale per anno
m_NO_anno = m_NO * mass_secco
m_CO_anno = m_CO * mass_secco
m_CO2_anno = m_CO2 * mass_secco

# g/kWh (potenza termica)
E_tot_J = mass_secco * LHV_dry
E_tot_kWh = E_tot_J / 3.6e6
g_NO_kWh = m_NO_anno * 1e3 / E_tot_kWh
g_CO_kWh = m_CO_anno * 1e3 / E_tot_kWh
g_CO2_kWh = m_CO2_anno * 1e3 / E_tot_kWh

print("\n--- RISULTATI ANNUALI ---")
print(f"NO: {m_NO_anno:.2f} kg/anno, {g_NO_kWh:.2f} g/kWh")
print(f"CO: {m_CO_anno:.2f} kg/anno, {g_CO_kWh:.2f} g/kWh")
print(f"CO2: {m_CO2_anno:.2f} kg/anno, {g_CO2_kWh:.2f} g/kWh")
print(f"Temperatura fiamma φ ottimale: {gas.T:.1f} K")
