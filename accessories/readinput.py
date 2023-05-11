import os,sys

filename = 'input.txt'

with open(filename,'r') as file:
    data = []
    labels = []
    for lines in file:
        vals = lines.split()
        data.append(float(vals[2]))

[delta_0, dp_0, Tg_0, Tp_0, end, tstep, Re, pressure] = data

print(end)

