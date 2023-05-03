import numpy as np
import matplotlib.pyplot as plt
from functions import *
# from nasaPoly import *

# T = 1200
# nasaPoly.listSpecies()
# oxygen = nasaPoly.Species('O2')
# ironoxide = nasaPoly.Species('FeO')

print("Viscosity of air at 1000K =", Mu_air(1000))
print("Thermal conductivity of air at 1000K =", lambda_air(1000))
print("Cp of air at 1000K =", Cp_air(1000))
print("Density of air at 1000K =", rho_air(1696.7))
print("Density of O2 at 1696.7K =", rho_O2(1696.7))
print("Diffusivity of O2 at 1696.7K =", D_O2(1696.7))
print("Cp of O2 at 1000K =",O2_data('cp',1000)/(W_O2/1000))
print("Cp of N2 at 1000K =",N2_data('cp',1000)/(W_N2/1000))

T = np.linspace(800,1290,num=3000)
H = np.zeros_like(T)

for i in range(len(H)):
    H[i] = Fe_data('h',T[i])

plt.figure()
plt.plot(T,H)
plt.xlabel('T [K]')
plt.ylabel(' H [J/mol]')
plt.savefig('images/HvsT_corrected.png')

# print(Fe_data('h',1183.99997))
# print(Fe_data('h',1183.99999))
# print(Fe_data('h',1184.00001))
# print(Fe_data('h',1184.00003))
# print('\n')
# print(Fe_data('cp',1183.99997))
# print(Fe_data('cp',1183.99999))
# print(Fe_data('cp',1184.00001))
# print(Fe_data('cp',1184.00003))

# oxygen.printState(T)
# print(O2_data('cp',T),O2_data('h',T))

# ironoxide.printState(T)
# print(FeO_data('cp',T),FeO_data('h',T))

print(rho_air(300))
print(rho_O2(300))
print(D_O2(300))
print(lambda_O2(1000))