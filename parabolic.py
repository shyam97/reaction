import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import newton
from functions import *
import os
import cantera

# USER-DEFINED PARAMETERS ------------------------------------------------------

filename = 'input.txt'

with open(filename,'r') as file:
    data = []
    labels = []
    for lines in file:
        vals = lines.split()
        data.append(float(vals[2]))

[delta_0, dp_0, Tg_0, Tp_0, end, tstep, Re, pressure, evapflag, radflag] = data

# ==============================================================================

# MODEL CONSTANTS --------------------------------------------------------------

k0_FeO = 2.67e-4    # pre-exponential factor in m2/s
Ta_FeO = 20319      # activation temperature of FeO in K
H0_FeO = -266269.76 # enthalpy of formation of FeO in J/mol at 273.15 K
rho_Fe = 7874       # density of Fe in kg/m3
rho_FeO = 5745      # density of FeO in kg/m3
epsilon = 0.88      # emissivity of Fe3O4 
sigma = 5.67e-8     # Stefan-Boltzmann constant in W/m2K4
W_Fe = 55.845*1e-3  # molar weight of Fe in g/mol
W_FeO = 71.844*1e-3 # molar weight of FeO in g/mol
W_O = 15.999*1e-3   # molar weight of O in g/mol
W_O2 = 31.999*1e-3  # molar weight of O2 in g/mol
W_N2 = 28.0134*1e-3 # molar weight of N2 in g/mol
R = 8.3144598       # real gas constant in J/mol/K
# X_N2 = 0.7808       # mole fraction of N2 in air
X_O2 = 0.2095       # mole fraction of O2 in air
factor = 1/2        # boundary layer factor   

# ==============================================================================

# INITIALISING SYSTEM VARIABLES ------------------------------------------------

dp = dp_0
dp_Fe = dp_0 * (1-delta_0)
X_FeO = dp_0 * delta_0 / 2

m_Fe = np.pi * dp_Fe**3 / 6 * rho_Fe
m_FeO = np.pi / 6 * ( dp**3 - dp_Fe**3 ) * rho_FeO

Tp = Tp_0
Tg = Tg_0

nu_O2_FeO = 0.5 * W_O2 / W_FeO
nu_Fe_O2 = W_Fe / W_O2 / 0.5
nu_FeO_O2 = W_FeO / W_O2 / 0.5

Hp = m_Fe / (W_Fe) * Fe_data('h',Tp) + m_FeO / (W_FeO) * FeO_data('h',Tp)

gas_f = ct.Solution('air_iron3.yaml')
if evapflag: gas_p = ct.Solution('air_iron3.yaml')

# ==============================================================================

# MAIN TIME LOOP ---------------------------------------------------------------

time = tstep
iter = 1
times = []
temps, Fe_mass, FeO_mass, Hdots, FeO_X, Hps = ([] for list in range(6))
O2_mdot_max, O2_mdot_r, O2_mdot = ([] for list in range(3))

# LOOP STARTS HERE

while time <= end:

    # PRE-PROCESSING
    Tf = (1-factor)*Tp + factor*Tg
    X_O2_f = X_O2*factor

    gas_f.TPX = Tf, pressure, {'O2':X_O2_f, 'N2':1-X_O2_f}

    Mu_a_f = gas_f.viscosity
    Rho_a_f = gas_f.density
    Cp_a_f = gas_f.cp
    Lam_a_f = gas_f.thermal_conductivity
    Diff_O2 = gas_f.mix_diff_coeffs[gas.species_index('O2')]
    Rho_O2 = X_O2 * W_O2 * pressure / (R * Tf)

    h_Feg = h_Fe_g(Tp,pressure)
    h_FeOg = h_FeO_g(Tp,pressure)

    Sc = Mu_a_f / Rho_a_f / Diff_O2
    Sh = 2 + 0.552 * Re**0.5 * Sc**0.333
    Pr = Mu_a_f * Cp_a_f / Lam_a_f
    Nu = 2 + 0.552 * Re**0.5 * Pr**0.333

    if evapflag:

        gas_p.TPX = Tp, pressure, {'O2':X_O2_f, 'N2':1-X_O2_f}

        Mu_a_p = gas_p.viscosity
        Rho_a_p = gas_p.density
        Diff_Fe = gas_p.mix_diff_coeffs[gas.species_index('Fe')]
        Diff_FeO = gas_p.mix_diff_coeffs[gas.species_index('FeO')]

        Sc_Fe = Mu_a_p / Rho_a_p / Diff_Fe
        Sh_Fe = 2 + 0.6 * Re**0.5 * Sc_Fe**(1/3)
        Sc_FeO = Mu_a_p / Rho_a_p / Diff_FeO
        Sh_FeO = 2 + 0.6 * Re**0.5 * Sc_FeO**(1/3)
        evap_Fe = Sh_Fe * Diff_Fe / dp * pvap_Fe(Tp) / R / Tp
        evap_FeO = (Sh_Fe * Diff_Fe / dp * pvap_FeFeO(Tp) + Sh_FeO * Diff_FeO / dp * pvap_FeO(Tp)) / R / Tp

        mdot_Fe_evap = m_Fe * np.pi * dp_Fe**2 * evap_Fe
        mdot_FeO_evap = m_FeO * np.pi * dp**2 * evap_FeO
        evapterms = - mdot_Fe_evap * h_Feg - mdot_FeO_evap * h_FeOg
    else:
        mdot_Fe_evap = 0
        mdot_FeO_evap = 0
        evapterms = 0

    if radflag:
        radterms = -sigma * epsilon * (Tp**4 - Tg**4)
    else:
        radterms = 0

    # CALCULATE MASS FLOW RATE OF OXYGEN

    mdot_O2_r = -nu_O2_FeO * np.pi * dp**2 * rho_FeO * k0_FeO / X_FeO * np.exp(-Ta_FeO/Tp) # KINETIC RATE
    mdot_O2_dmax = - np.pi * dp * Sh * Diff_O2 * Rho_O2 # EXTERNAL DIFFUSION RATE

    # SWITCH BETWEEN DIFFUSION AND KINETICS

    if mdot_O2_r < mdot_O2_dmax:
        mdot_O2 = mdot_O2_dmax
    else:
        mdot_O2 = mdot_O2_r

    # COMPUTE MASS CHANGE IN FE AND FEO

    mdot_FeO = -nu_FeO_O2 * mdot_O2 - mdot_FeO_evap
    mdot_Fe = nu_Fe_O2 * mdot_O2 - mdot_FeO_evap

    if m_Fe + mdot_Fe * tstep < 0:
        mdot_Fe = 0
        mdot_O2 = 0
        if m_FeO + mdot_FeO * tstep < 0:
            mdot_FeO = 0
        else:
            mdot_FeO = -mdot_FeO_evap
    
    # COMPUTE ENTHALPY CHANGE OF PARTICLE

    Hdot = - O2_data('h',Tp) / (W_O2) * mdot_O2 - np.pi * dp * Nu * Lam_a_f *(Tp - Tg) \
            + evapterms + radterms
    
    # UPDATE MASS, DIAMETER AND ENTHALPY
  
    m_Fe += mdot_Fe * tstep
    m_FeO += mdot_FeO * tstep

    dp_Fe = (6*m_Fe/rho_Fe/np.pi)**(1/3)
    X = 0.5 * ( ((6*m_FeO/np.pi/rho_FeO) + dp_Fe**3)**(1/3) - dp_Fe )
    dp = dp_Fe + 2*X
    
    Hp += Hdot * tstep 

    # DETERMINE PARTICLE TEMPERATURE

    try:
        Tp = newton(Tp_eqn, x0=Tp, args=(m_Fe,m_FeO,Hp),maxiter=1000,tol=1)
    except:
        print('Error!')
        break

    # SAVE VARIABLES

    times.append(time*1000)
    temps.append(Tp)
    Fe_mass.append(m_Fe)
    FeO_mass.append(m_FeO)
    Hdots.append(Hdot)
    Hps.append(Hp)
    FeO_X.append(X_FeO)

    O2_mdot_max.append(mdot_O2_dmax)
    O2_mdot_r.append(mdot_O2_r)
    O2_mdot.append(mdot_O2)

    # WRAP UP AND CONTINUE

    time+=tstep
    iter+=1

    print(np.round(time*1000,3),'ms, T =',np.round(Tp, 3),'K,', int(time/end*100), '%% done.')#, end='\r') 

# PLOTS

plt.figure(num=4)
plt.plot(times,temps)
plt.xlabel('$t$ [ms]')
plt.ylabel('$T_\mathrm{p}$ [K]')
plt.xlim([0,times[-1]])
plt.savefig('temperature.png')

plt.figure(num=3,figsize=(10,5))
fig,axs = plt.subplots(nrows=1, ncols=2,figsize=(10,5))
axs[0].plot(times,O2_mdot,'-',color='gray',label='$\dot{m}_{O_2}$',linewidth=3)
axs[0].plot(times,O2_mdot_r,label='$\dot{m}_{O_2,r}$')
axs[0].plot(times,O2_mdot_max,label='$\dot{m}_{O_2,dmax}$')
axs[0].set_xlabel('$t$ [ms]')
axs[0].set_ylabel('$\dot{m}$ [kg/s]')
axs[0].set_xlim([0,times[-1]])
axs[0].legend()

axs[1].plot(times,Hdots)
axs[1].set_xlabel('$t$ [ms]')
axs[1].set_ylabel('$\dot{H}$ [J/s]')
axs[1].set_xlim([0,times[-1]])
fig.tight_layout()
fig.savefig('rates.png')

plt.figure(num=1, figsize=(10,10))
fig,axs = plt.subplots(nrows=2, ncols=2, figsize=(10,10))
axs[0,0].plot(times, temps)
axs[0,0].set_xlabel('$t$ [ms]')
axs[0,0].set_ylabel('$T_\mathrm{p}$ [K]')
axs[0,0].set_xlim([0,times[-1]])

axs[0,1].plot(times, Fe_mass,label="Fe_mass")
axs[0,1].plot(times, FeO_mass,label='FeO_mass')
axs[0,1].legend()
axs[0,1].set_xlabel('$t$ [ms]')
axs[0,1].set_ylabel('$m$ [kg]')
axs[0,1].set_xlim([0,times[-1]])

axs[1,0].plot(times, Hps)
axs[1,0].set_xlabel('$t$ [ms]')
axs[1,0].set_ylabel('$H_\mathrm{p}$ [J]')
axs[1,0].ticklabel_format(axis='y',style='sci',scilimits=(0,0))
axs[1,0].set_xlim([0,times[-1]])

axs[1,1].plot(times, FeO_X)
axs[1,1].set_xlabel('$t$ [ms]')
axs[1,1].set_ylabel('$X_\mathrm{FeO}$ [m]')
axs[1,1].set_xlim([0,times[-1]])

fig.tight_layout()
fig.savefig('properties.png')


