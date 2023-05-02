import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import newton
from functions import *
import os

# DERIVED VARIABLES ------------------------------------------------------------

# USER-DEFINED PARAMETERS ------------------------------------------------------

filename = 'input.txt'

with open(filename,'r') as file:
    data = []
    labels = []
    for lines in file:
        vals = lines.split()
        data.append(float(vals[2]))

[delta_0, dp_0, Tg_0, Tp_0, end, tstep, Re, pressure, debug] = data

# ==============================================================================

# MODEL CONSTANTS --------------------------------------------------------------

k0_FeO = 2.67e-4    # pre-exponential factor in m2/s
Ta_FeO = 20319      # activation temperature of FeO in K
H0_FeO = -266269.76 # enthalpy of formation of FeO in J/mol at 273.15 K
rho_Fe = 7874       # density of Fe in kg/m3
rho_FeO = 5745      # density of FeO in kg/m3
epsilon = 0.88      # emissivity of Fe3O4 
W_Fe = 55.845*1e-3  # molar weight of Fe in g/mol
W_FeO = 71.844*1e-3 # molar weight of FeO in g/mol
W_O = 15.999*1e-3   # molar weight of O in g/mol
W_O2 = 31.999*1e-3  # molar weight of O2 in g/mol
W_N2 = 28.0134*1e-3 # molar weight of N2 in g/mol
R = 8.3144598       # real gas constant in J/mol/K
X_N2 = 0.7808       # mole fraction of N2 in air
X_O2 = 0.2095       # mole fraction of O2 in air

# ==============================================================================

# INITIALISING SYSTEM VARIABLES ------------------------------------------------

dp = dp_0
dp_Fe = dp_0 * (1-delta_0)
X_FeO = dp_0 * delta_0 / 2

m_Fe = np.pi * dp_Fe**3 / 6 * rho_Fe
m_FeO = np.pi / 6 * ( dp**3 - dp_Fe**3 ) * rho_FeO

Tp = Tp_0
Tg = Tg_0

nu_O2_FeO = W_O2 / W_FeO
nu_Fe_O2 = W_Fe / W_O2
nu_FeO_O2 = W_FeO / W_O2

print(nu_O2_FeO, nu_FeO_O2, nu_Fe_O2)

Hp = m_Fe / (W_Fe) * Fe_data('h',Tp) + m_FeO / (W_FeO) * FeO_data('h',Tp)

# ==============================================================================

# MAIN TIME LOOP ---------------------------------------------------------------

time = tstep
iter = 1
times = []
temps, Fe_mass, FeO_mass, Hdots, FeO_X, Hps = ([] for list in range(6))
O2_mdot_max, O2_mdot_r, O2_mdot = ([] for list in range(3))

if debug:

    print(dp_Fe, X_FeO, m_Fe, m_FeO, Hp, Tp)

    debugfile = 'debug.txt'

    if os.path.exists(debugfile):
        os.remove(debugfile)

    log = open(debugfile,'w')
    log.write('[mdot_O2_r , mdot_O2_dmax , mdot_O2 , H1 , H2 , H3 , Hdot]\n')

# LOOP STARTS HERE

while time <= end:

    Tf = (2*Tp + Tg)/3

    mdot_O2_r = -nu_O2_FeO * area(dp) * rho_FeO * k0_FeO / X_FeO * np.exp(-Ta_FeO/Tp)
    mdot_O2_dmax = -area(dp) * beta_p(Re,Tf,dp) * rho_O2(Tg)

    if mdot_O2_r < mdot_O2_dmax:
        mdot_O2 = mdot_O2_dmax
    else:
        mdot_O2 = mdot_O2_r

    mdot_FeO = -nu_FeO_O2 * mdot_O2
    mdot_Fe = nu_Fe_O2 * mdot_O2

    if m_Fe + mdot_Fe * tstep < 0:
        mdot_Fe = 0
        mdot_O2 = 0
        mdot_FeO = 0
    
    Hdot = -H0_FeO / (W_FeO) * mdot_FeO - O2_data('h',Tf) / (W_O2) * mdot_O2 \
        - area(dp)*h_p(Re,Tf,dp)*(Tp - Tf)
    
    # UPDATE VARIABLES
    
    m_Fe += mdot_Fe * tstep
    m_FeO += mdot_FeO * tstep

    [dp_Fe,X_FeO,dp] = mass_to_diameter(m_Fe,m_FeO)
    
    Hp += Hdot * tstep 

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

    # DEBUG

    if debug:
        H1 = H0_FeO / (W_FeO) * mdot_FeO
        H2 = - O2_data('h',Tf) / (W_O2) * mdot_O2
        H3 = - area(dp)*h_p(Re,Tf,dp)*(Tp - Tf)
        logs = [mdot_O2_r,mdot_O2_dmax,mdot_O2,H1,H2,H3,Hdot]
        for s in logs:
            log.write('%.8e , ' %s)
        log.write('\n')

    if m_Fe<0:
        print('Holup bro!')
        break

    # NEWTON-RAPHSON

    Tp = newton(Tp_eqn, x0=Tp, args=(m_Fe,m_FeO,Hp),maxiter=1000,tol=1)

    # try:
    #     Tp = newton(Tp_eqn, x0=Tp, args=(m_Fe,m_FeO,Hp),maxiter=1000,tol=1)
    # except:
    #     print('Error!')
    #     break

    # WRAP UP AND CONTINUE

    time+=tstep
    iter+=1

    print(np.round(time*1000,3),'ms, T =',np.round(Tp, 3),'K,', int(time/end*100), '%% done.', end='\r') 

if debug:
    log.close()

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


