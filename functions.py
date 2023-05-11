import numpy as np
import os
import cantera as ct

gas = ct.Solution('air_iron3.yaml')

# DERIVED VARIABLES ------------------------------------------------------------

# USER-DEFINED PARAMETERS ------------------------------------------------------

filename = 'input.txt'

with open(filename,'r') as file:
    data = []
    labels = []
    for lines in file:
        vals = lines.split()
        data.append(float(vals[2]))

[delta_0, dp_0, Tg_0, Tp_0, end, tstep, Re, pressure, debug, evapflag, radflag] = data

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
# X_O2 = 0.2095       # mole fraction of O2 in air
factor = 1/2        # boundary layer factor 

# ==============================================================================

def h_p(Re,temp,dp,X_O2):
    nusselt = Nu(Re,temp,X_O2)
    lambda_g = lambda_air(temp,X_O2)
    hp = nusselt * lambda_g / dp
    return hp

def Nu(Re,temp,X_O2):
    Pr = Mu_air(temp,X_O2) * Cp_air(temp,X_O2) / lambda_air(temp,X_O2)
    nusselt = 2 + 0.552 * Re**0.5 * Pr**0.333
    return nusselt

def Sh(Re,temp,X_O2):
    Sc = Mu_air(temp,X_O2) / rho_air(temp,X_O2) / D_O2(temp,X_O2)
    sherwood = 2 + 0.552 * Re**0.5 * Sc**0.333
    return sherwood

def beta_p(Re,temp,d_p,X_O2):
    beta = Sh(Re,temp,X_O2) * D_O2(temp,X_O2) / d_p 
    return beta

def mass_to_diameter(mass_Fe, mass_FeO):
    vol_Fe = mass_Fe/rho_Fe
    d_Fe = (6*vol_Fe/np.pi)**(1/3)
    X = 0.5 * ( ((6*mass_FeO/np.pi/rho_FeO) + d_Fe**3)**(1/3) - d_Fe )
    dp = d_Fe + 2*X
    return np.array([d_Fe, X, dp])

def area(diameter):
    A = np.pi * diameter**2 
    return A

def Tp_eqn(temp, m_Fe, m_FeO, Hp):
    val_Fe = m_Fe / W_Fe * Fe_data('h',temp)
    val_FeO = m_FeO / W_FeO * FeO_data('h',temp) #- H0_FeO / W_FeO * m_FeO
    return val_Fe + val_FeO - Hp

def Tp_prime(temp, m_Fe, m_FeO, Hp):
    Cp_tot = m_Fe / W_Fe * Fe_data('cp',temp) + m_FeO / W_FeO * FeO_data('cp',temp)
    return Cp_tot

# ==============================================================================

def evap(species,Re,Tp,X_O2,dp):
    R = 8.3144598

    if species == 'Fe':
        kd = kd_Fe(Re,Tp,X_O2,dp)
        pvap = pvap_Fe(Tp)
        return kd * pvap / R / Tp

    elif species == 'FeO':
        k_FeFeO = kd_FeO(Re,Tp,X_O2,dp)
        pv_FeFeO = pvap_FeFeO(Tp)

        k_FeO = kd_FeO(Re,Tp,X_O2,dp)
        pv_FeO = pvap_FeO(Tp)

        return (k_FeO * pv_FeO + k_FeFeO * pv_FeFeO) / R / Tp

    return 0

# ==============================================================================

def kd_Fe(Re,temp,X_O2,dp):

    gas.TPX = temp, pressure, {'O2':X_O2, 'N2':1-X_O2}

    D_Fe = gas.mix_diff_coeffs[gas.species_index('Fe')]

    Sc = gas.viscosity/gas.density/D_Fe
    Sh = 2 + 0.6 * Re**0.5 * Sc**(1/3)

    return Sh * D_Fe / dp

def kd_FeO(Re,temp,X_O2,dp):

    gas.TPX = temp, pressure, {'O2':X_O2, 'N2':1-X_O2}

    D_FeO = gas.mix_diff_coeffs[gas.species_index('FeO')]

    Sc = gas.viscosity/gas.density/D_FeO
    Sh = 2 + 0.6 * Re**0.5 * Sc**(1/3)

    return Sh * D_FeO / dp

# ==============================================================================

def pvap_Fe(Tp):
    A1 = 35.4
    A2 = -4.963e4
    A3 = -2.433

    term1 = A1
    term2 = A2 / Tp
    term3 = A3 * np.log(Tp)

    return np.exp(term1 + term2 + term3)

def pvap_FeFeO(Tp):
    A1 = 62.08
    A2 = -6.412e4
    A3 = -5.399

    term1 = A1
    term2 = A2 / Tp
    term3 = A3 * np.log(Tp)

    return np.exp(term1 + term2 + term3)

def pvap_FeO(Tp):
    A1 = 52.93
    A2 = -6.48e4
    A3 = -4.37

    term1 = A1
    term2 = A2 / Tp
    term3 = A3 * np.log(Tp)

    return np.exp(term1 + term2 + term3)

# ==============================================================================

# NASA POLYNOMIALS -------------------------------------------------------------

def eta_N2(temp):
    intervals = [300,1000,5000]

    poly0 = [0.60443938,-0.43632704e+02, -0.88441949e+03, 0.18972150e+01]
    poly1 = [0.65060585, 0.28517449e+02, -0.16690236e+05, 0.15223271e+01]

    poly = np.array([poly0,poly1])

    for i in range(len(intervals)):
        if intervals[i] <= temp <= intervals[i+1]:
            coeff = poly[i]
            break
    
    log_eta = coeff[0]*np.log(temp) + coeff[1]/temp + coeff[2]/(temp**2) \
                + coeff[3]

    Nu = np.exp(log_eta)*1e-7
    return Nu

# ------------------------------------------------------------------------------
    
def lambda_N2(temp):
    intervals = [300,1000,5000]

    poly0 = [0.94306384, 0.12279898e+03, -0.11839435e+05, -0.10668773]
    poly1 = [0.65147781, -0.15059801e+03, -0.13746760e+05, 0.21801632e+01]

    poly = np.array([poly0,poly1])

    for i in range(len(intervals)):
        if intervals[i] <= temp <= intervals[i+1]:
            coeff = poly[i]
            break
    
    log_lambda = coeff[0]*np.log(temp) + coeff[1]/temp + coeff[2]/(temp**2) \
                + coeff[3]

    lambda_n2 = np.exp(log_lambda)*1e-4
    return lambda_n2

# ------------------------------------------------------------------------------

def eta_O2(temp):
    intervals = [300,1000,5000]

    poly0 = [0.61936357, -0.44608607e+02, -0.13460714e+04, 0.19597562e+01]
    poly1 = [0.63839563, -0.12344438e+01, -0.22885810e+05, 0.18056937e+01]

    poly = np.array([poly0,poly1])

    for i in range(len(intervals)):
        if intervals[i] <= temp <= intervals[i+1]:
            coeff = poly[i]
            break
    
    log_eta = coeff[0]*np.log(temp) + coeff[1]/temp + coeff[2]/(temp**2) \
                + coeff[3]

    Nu = np.exp(log_eta)*1e-7
    return Nu

# ------------------------------------------------------------------------------

def lambda_O2(temp):
    intervals = [300,1000,5000]

    poly0 = [0.81595343, -0.34366856e+02, 0.22785080e+04, 0.10050999e+01]
    poly1 = [0.80805788, 0.11982181e+03, -0.47335931e+05, 0.95189193]

    poly = np.array([poly0,poly1])

    for i in range(len(intervals)):
        if intervals[i] <= temp <= intervals[i+1]:
            coeff = poly[i]
            break
        
    log_lambda = coeff[0]*np.log(temp) + coeff[1]/temp + coeff[2]/(temp**2) \
                + coeff[3]

    lambda_n2 = np.exp(log_lambda)*1e-4
    return lambda_n2

# ------------------------------------------------------------------------------

def N2_data(param,temp):
    R = 8.3144598

    intervals = [200,1000,6000]

    poly0 = [2.210371497e+04, -3.818461820e+02, 6.082738360e+00, -8.530914410e-03,
             1.384646189e-05, -9.625793620e-09, 2.519705809e-12, 7.108460860e+02]
    poly1 = [5.877124060e+05, -2.239249073e+03, 6.066949220e+00,-6.139685500e-04,
             1.491806679e-07, -1.923105485e-11, 1.061954386e-15, 1.283210415e+04]
    
    poly = np.array([poly0, poly1])

    for i in range(len(intervals)):
        if intervals[i] <= temp <= intervals[i+1]:
            coeff = poly[i]
            break
    
    if param == 'h':
        H = -coeff[0]/temp + coeff[1]*np.log(temp) + ((((coeff[6]/5*temp + \
            coeff[5]/4)*temp + coeff[4]/3)*temp + coeff[3]/2)*temp + coeff[2])*temp + \
            coeff[7]
        
        H = H*R
        return H
    
    if param == 'cp':
        Cp = coeff[0]/temp**2 + coeff[1]/temp + (((coeff[6]*temp + \
            coeff[5])*temp + coeff[4])*temp + coeff[3])*temp + coeff[2]

        Cp = Cp*R
        return Cp
    
# ------------------------------------------------------------------------------

def O2_data(param,temp):
    R = 8.3144598

    intervals = [200,1000,6000]

    poly0 = [-3.425563420e+04, 4.847000970e+02, 1.119010961e+00, 4.293889240e-03,
             -6.836300520e-07, -2.023372700e-09, 1.039040018e-12, -3.391454870e+03]
    poly1 = [-1.037939022e+06, 2.344830282e+03, 1.819732036e+00, 1.267847582e-03,
             -2.188067988e-07, 2.053719572e-11, -8.193467050e-16, -1.689010929e+04]
    
    poly = np.array([poly0, poly1])

    for i in range(len(intervals)):
        if intervals[i] <= temp <= intervals[i+1]:
            coeff = poly[i]
            break
    
    if param == 'h':
        H = -coeff[0]/temp + coeff[1]*np.log(temp) + ((((coeff[6]/5*temp + \
            coeff[5]/4)*temp + coeff[4]/3)*temp + coeff[3]/2)*temp + coeff[2])*temp + \
            coeff[7]
        
        H = H*R
        return H
    
    if param == 'cp':
        Cp = coeff[0]/temp**2 + coeff[1]/temp + (((coeff[6]*temp + \
            coeff[5])*temp + coeff[4])*temp + coeff[3])*temp + coeff[2]

        Cp = Cp*R
        return Cp
    
# ------------------------------------------------------------------------------

def rho_O2(temp,X_O2):
    R = 8.3144598
    r_O2 = X_O2 * W_O2 * pressure / (R * temp)
    return r_O2

# ------------------------------------------------------------------------------

def rho_air(temp,X_O2):

    # R = 8.3144598
    # X_N2 = 1 - X_O2
    # r_air =  ((W_O2) * X_O2 + (W_N2) * X_N2) * pressure / (R * temp)

    gas.TPX = temp, pressure, {'O2':X_O2, 'N2':1 - X_O2}
    return gas.density

# ------------------------------------------------------------------------------

def lambda_air(temp,X_O2):

    # X_N2 = 1 - X_O2
    # sum1 = X_N2 * lambda_N2(temp) + X_O2 * lambda_O2(temp)
    # sum2 = X_N2 / lambda_N2(temp) + X_O2 / lambda_O2(temp)
    # k_air = 0.5 * (sum1 + 1/sum2)

    gas.TPX = temp, pressure, {'O2':X_O2, 'N2':1-X_O2}
    return gas.thermal_conductivity

# ------------------------------------------------------------------------------

def Mu_air(temp, X_O2):

    # X_N2 = 1 - X_O2
    # sum1 = X_N2 * eta_N2(temp) + X_O2 * eta_O2(temp)
    # sum2 = X_N2 / eta_N2(temp) + X_O2 / eta_O2(temp)
    # eta_air = 0.5 * (sum1 + 1/sum2)

    gas.TPX = temp, pressure, {'O2':X_O2, 'N2':1-X_O2}
    return gas.viscosity

# ------------------------------------------------------------------------------

def Cp_air(temp, X_O2):

    # X_N2 = 1 - X_O2
    # sum = X_N2 * N2_data("cp",temp) / (W_N2) + X_O2 * O2_data("cp",temp) / (W_O2)

    gas.TPX = temp, pressure, {'O2':X_O2, 'N2':1-X_O2}
    return gas.cp

# ------------------------------------------------------------------------------

def D_O2(temp,X_O2):

    # k_O2 = lambda_O2(temp)
    # k_air = lambda_air(temp,X_O2)
    # r_air = rho_air(temp,X_O2)
    # r_O2 = rho_O2(temp,1)
    # Cp_O2 = O2_data("cp",temp) / (W_O2) #* X_O2
    # Le = 1.11
    # D_old = k_O2 / r_O2 / Cp_O2 / Le

    gas.TPX = temp, pressure, {'O2':X_O2, 'N2':1-X_O2}
    D = gas.mix_diff_coeffs[gas.species_index('O2')]
    return D

# ------------------------------------------------------------------------------

def Fe_data(param,temp):
    R = 8.3144598

    intervals = [200,500,800,1042,1184,1665,1809,6000]
    
    poly0 = [1.350490931e+04, -7.803806250e+02, 9.440171470e+00, -2.521767704e-02,
             5.350170510e-05,-5.099094730e-08, 1.993862728e-11,  2.416521408e+03]
    
    poly1 = [3.543032740e+06, -2.447150531e+04, 6.561020930e+01, -7.043929680e-02,
             3.181052870e-05, 0.000000000e+00, 0.000000000e+00, 1.345059978e+05]
    
    poly2 = [2.661026334e+09, -7.846827970e+06, -7.289212280e+02, 2.613888297e+01,
             -3.494742140e-02, 1.763752622e-05, -2.907723254e-09, 5.234868470e+07]
    
    poly3 = [2.481923052e+08, 0.000000000e+00, -5.594349090e+02, 3.271704940e-01,
             0.000000000e+00, 0.000000000e+00, 0.000000000e+00, 6.467503430e+05]
    
    poly4 = [1.442428576e+09, -5.335491340e+06, 8.052828000e+03, -6.303089630e+00,
             2.677273007e-03, -5.750045530e-07, 4.718611960e-11, 3.264264250e+07]

    poly5 = [-3.450190030e+08, 0.000000000e+00, 7.057501520e+02, -5.442977890e-01,
             1.190040139e-04, 0.000000000e+00, 0.000000000e+00, -8.045725750e+05]
    
    poly6 = [0.000000000e+00, 0.000000000e+00, 5.535383324e+00, 0.000000000e+00, 
             0.000000000e+00, 0.000000000e+00, 0.000000000e+00, -1.270608703e+03]
    
    poly = np.array([poly0, poly1, poly2, poly3, poly4, poly5, poly6])

    for i in range(len(intervals)):
        if intervals[i] <= temp and temp < intervals[i+1]:

            if param=='h' and np.abs(intervals[i]-temp) < 5 and i>0:
                var1 = intervals[i] - 5
                var2 = intervals[i] + 5
                param1 = Fe_data(param,var1)
                param2 = Fe_data(param,var2)
                var3 = param1 + (param2 - param1)*(temp - var1)/(var2-var1)
                return var3

            if param=='h' and np.abs(intervals[i+1]-temp) < 5 and i+1 < len(intervals)-1:
                var1 = intervals[i+1] - 5
                var2 = intervals[i+1] + 5
                param1 = Fe_data(param,var1)
                param2 = Fe_data(param,var2)
                var3 = param1 + (param2 - param1)*(temp - var1)/(var2-var1)
                return var3

            coeff = poly[i]
            break
    
    if param == 'h':
        H = -coeff[0]/temp + coeff[1]*np.log(temp) + ((((coeff[6]/5*temp + \
            coeff[5]/4)*temp + coeff[4]/3)*temp + coeff[3]/2)*temp + coeff[2])*temp + \
            coeff[7]
        
        H = H*R
        return H
    
    if param == 'cp':
        Cp = coeff[0]/temp**2 + coeff[1]/temp + (((coeff[6]*temp + \
            coeff[5])*temp + coeff[4])*temp + coeff[3])*temp + coeff[2]

        Cp = Cp*R
        return Cp

def FeO_data(param,temp):
    R = 8.3144598

    intervals = [298.15,1652,6000]
    
    poly0 = [-1.179193966e+04, 1.388393372e+02, 2.999841854e+00, 1.274527210e-02,
             -1.883886065e-05, 1.274258345e-08, -3.042206479e-12, -3.417350500e+04]
    
    poly1 = [0.000000000e+00, 0.000000000e+00, 8.147077819e+00, 0.000000000e+00,
             0.000000000e+00, 0.000000000e+00, 0.000000000e+00, -3.255080650e+04]
    
    poly = np.array([poly0, poly1])

    for i in range(len(intervals)):
        if intervals[i] <= temp and temp < intervals[i+1]:

            if param=='h' and np.abs(intervals[i]-temp) < 5 and i>0:
                var1 = intervals[i] - 5
                var2 = intervals[i] + 5
                param1 = FeO_data(param,var1)
                param2 = FeO_data(param,var2)
                var3 = param1 + (param2 - param1)*(temp - var1)/(var2-var1)
                return var3

            if param=='h' and np.abs(intervals[i+1]-temp) < 5 and i+1<len(intervals)-1:
                var1 = intervals[i+1] - 5
                var2 = intervals[i+1] + 5
                param1 = FeO_data(param,var1)
                param2 = FeO_data(param,var2)
                var3 = param1 + (param2 - param1)*(temp - var1)/(var2-var1)
                return var3

            coeff = poly[i]
            break
    
    if param == 'h':
        H = -coeff[0]/temp + coeff[1]*np.log(temp) + ((((coeff[6]/5*temp + \
            coeff[5]/4)*temp + coeff[4]/3)*temp + coeff[3]/2)*temp + coeff[2])*temp + \
            coeff[7]
        
        H = H*R
        return H
    
    if param == 'cp':
        Cp = coeff[0]/temp**2 + coeff[1]/temp + (((coeff[6]*temp + \
            coeff[5])*temp + coeff[4])*temp + coeff[3])*temp + coeff[2]

        Cp = Cp*R
        return Cp

# ==============================================================================

def h_Fe_g(Tp):
    gas_Fe = ct.Solution('air_iron3.yaml')
    gas_Fe.TPY = Tp, pressure, {'Fe':1}

    return gas_Fe.enthalpy_mass

def h_FeO_g(Tp):
    gas_FeO = ct.Solution('air_iron3.yaml')
    gas_FeO.TPY = Tp, pressure, {'FeO':1}

    return gas_FeO.enthalpy_mass

# ===============================================================================

