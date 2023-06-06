import numpy as np
import os
import cantera as ct

gas = ct.Solution('air_iron3.yaml')

# MODEL CONSTANTS --------------------------------------------------------------

W_Fe = 55.845*1e-3  # molar weight of Fe in g/mol
W_FeO = 71.844*1e-3 # molar weight of FeO in g/mol
W_O2 = 31.999*1e-3  # molar weight of O2 in g/mol
Fe_intervals = [200,500,800,1042,1184,1665,1809,6000]
FeO_intervals = [298.15,1652,6000]

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

def Fe_data(index,param,temp):
    R = 8.3144598
    
    poly = [[1.350490931e+04, -7.803806250e+02, 9.440171470e+00, -2.521767704e-02,
             5.350170510e-05,-5.099094730e-08, 1.993862728e-11,  2.416521408e+03],  

            [3.543032740e+06, -2.447150531e+04, 6.561020930e+01, -7.043929680e-02,
             3.181052870e-05, 0.000000000e+00, 0.000000000e+00, 1.345059978e+05],    

            [2.661026334e+09, -7.846827970e+06, -7.289212280e+02, 2.613888297e+01,
             -3.494742140e-02, 1.763752622e-05, -2.907723254e-09, 5.234868470e+07],
    
            [2.481923052e+08, 0.000000000e+00, -5.594349090e+02, 3.271704940e-01,
             0.000000000e+00, 0.000000000e+00, 0.000000000e+00, 6.467503430e+05],
    
            [1.442428576e+09, -5.335491340e+06, 8.052828000e+03, -6.303089630e+00,
             2.677273007e-03, -5.750045530e-07, 4.718611960e-11, 3.264264250e+07],

            [-3.450190030e+08, 0.000000000e+00, 7.057501520e+02, -5.442977890e-01,
             1.190040139e-04, 0.000000000e+00, 0.000000000e+00, -8.045725750e+05],
    
            [0.000000000e+00, 0.000000000e+00, 5.535383324e+00, 0.000000000e+00, 
             0.000000000e+00, 0.000000000e+00, 0.000000000e+00, -1.270608703e+03]]

    coeff = poly[index]
    
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

def FeO_data(index,param,temp):
    R = 8.3144598

    poly = [[-1.179193966e+04, 1.388393372e+02, 2.999841854e+00, 1.274527210e-02,
             -1.883886065e-05, 1.274258345e-08, -3.042206479e-12, -3.417350500e+04],
    
            [0.000000000e+00, 0.000000000e+00, 8.147077819e+00, 0.000000000e+00,
             0.000000000e+00, 0.000000000e+00, 0.000000000e+00, -3.255080650e+04]]
    
    coeff = poly[index]
    
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

def newtonFeFeO(m_Fe,m_FeO,Hp,T0,tol,maxiter):

    Fe_range = 0
    FeO_range = 0
    for i in range(len(Fe_intervals)-1):
        if T0 <= Fe_intervals[i+1]:
            Fe_range = i
            break
    
    for i in range(len(FeO_intervals)-1):
        if T0 <= FeO_intervals[i+1]:
            FeO_range = i
            break

    Ts = np.max([Fe_intervals[Fe_range],FeO_intervals[FeO_range]])
    Te = np.min([Fe_intervals[Fe_range+1],FeO_intervals[FeO_range+1]])

    count = 0
    while count<maxiter:
        count+=1

        Ts = np.min([Fe_intervals[Fe_range],FeO_intervals[FeO_range]])
        Te = np.min([Fe_intervals[Fe_range+1],FeO_intervals[FeO_range+1]])

        H0 = m_Fe / W_Fe * Fe_data(Fe_range,'h',T0) + m_FeO / W_FeO * FeO_data(FeO_range,'h',T0) - Hp #- H0_FeO / W_FeO * m_FeO
        Cp0 = m_Fe / W_Fe * Fe_data(Fe_range,'cp',T0) + m_FeO / W_FeO * FeO_data(FeO_range,'cp',T0)

        T1 = T0 - H0/Cp0

        if T1 > Te:
            Hs = m_Fe / W_Fe * Fe_data(Fe_range,'h',Te) + m_FeO / W_FeO * FeO_data(FeO_range,'h',Te)

            if T1 > Fe_intervals[Fe_range+1]:
                He = m_Fe / W_Fe * Fe_data(Fe_range+1,'h',Te)
            else:
                He = m_Fe / W_Fe * Fe_data(Fe_range,'h',Te)

            if T1 > FeO_intervals[FeO_range+1]:
                He += m_FeO / W_FeO * FeO_data(FeO_range+1,'h',Te)
            else:
                He+= m_FeO / W_FeO * FeO_data(FeO_range,'h',Te)

            if Hp > Hs:
                if Hp < He:
                    return Te
                else:
                    if T1 >= Fe_intervals[Fe_range+1]:
                        Fe_range+=1
                    if T1 >= FeO_intervals[FeO_range+1]:
                        FeO_range+=1
                    T1 = Te
            else:
                T1 = Te

        elif T1 < Ts:
            He = m_Fe / W_Fe * Fe_data(Fe_range,'h',Ts) + m_FeO / W_FeO * FeO_data(FeO_range,'h',Ts)

            if T1 < Fe_intervals[Fe_range]:
                Hs = m_Fe / W_Fe * Fe_data(Fe_range-1,'h',Ts)
            else:
                Hs = m_Fe / W_Fe * Fe_data(Fe_range,'h',Ts)

            if T1 < FeO_intervals[FeO_range]:
                Hs += m_FeO / W_FeO * FeO_data(FeO_range-1,'h',Ts)
            else:
                Hs+= m_FeO / W_FeO * FeO_data(FeO_range,'h',Ts)

            if Hp < He:
                if Hp > Hs:
                    return Ts
                else:
                    if T1 < Fe_intervals[Fe_range]:
                        Fe_range-=1
                    if T1 < FeO_intervals[FeO_range]:
                        FeO_range-=1
                    T1 = Ts
                    # return T1
            else:
                T1 = Ts

        elif np.abs(T1-T0) < tol:
                return T1
        else:
                T0 = T1

