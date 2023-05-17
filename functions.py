import numpy as np
import os
import cantera as ct

gas = ct.Solution('air_iron3.yaml')

# MODEL CONSTANTS --------------------------------------------------------------

W_Fe = 55.845*1e-3  # molar weight of Fe in g/mol
W_FeO = 71.844*1e-3 # molar weight of FeO in g/mol
W_O2 = 31.999*1e-3  # molar weight of O2 in g/mol

# ==============================================================================

def Tp_eqn(temp, m_Fe, m_FeO, Hp):
    val_Fe = m_Fe / W_Fe * Fe_data('h',temp)
    val_FeO = m_FeO / W_FeO * FeO_data('h',temp) #- H0_FeO / W_FeO * m_FeO
    return val_Fe + val_FeO - Hp

def Tp_prime(temp, m_Fe, m_FeO, Hp):
    Cp_tot = m_Fe / W_Fe * Fe_data('cp',temp) + m_FeO / W_FeO * FeO_data('cp',temp)
    return Cp_tot

# ==============================================================================

def pvap_Fe(Tp):
    A1 = 35.4
    A2 = -4.963e4
    A3 = -2.433

    term1 = A1
    term2 = A2 / Tp
    term3 = A3 * np.log(Tp)

    return np.exp(term1 + term2 + term3)

# ------------------------------------------------------------------------------

def pvap_FeFeO(Tp):
    A1 = 62.08
    A2 = -6.412e4
    A3 = -5.399

    term1 = A1
    term2 = A2 / Tp
    term3 = A3 * np.log(Tp)

    return np.exp(term1 + term2 + term3)

# ------------------------------------------------------------------------------

def pvap_FeO(Tp):
    A1 = 52.93
    A2 = -6.48e4
    A3 = -4.37

    term1 = A1
    term2 = A2 / Tp
    term3 = A3 * np.log(Tp)

    return np.exp(term1 + term2 + term3)

# ------------------------------------------------------------------------------

def h_Fe_g(Tp,pressure):
    gas_Fe = ct.Solution('air_iron3.yaml')
    gas_Fe.TPY = Tp, pressure, {'Fe':1}

    return gas_Fe.enthalpy_mass

# ------------------------------------------------------------------------------

def h_FeO_g(Tp,pressure):
    gas_FeO = ct.Solution('air_iron3.yaml')
    gas_FeO.TPY = Tp, pressure, {'FeO':1}

    return gas_FeO.enthalpy_mass

# ==============================================================================

# NASA POLYNOMIALS -------------------------------------------------------------

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
    
# ------------------------------------------------------------------------------

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

