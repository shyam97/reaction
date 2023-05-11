import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

matlabfile = 'vals.txt'
matlabtime = 'mtime.txt'
pythonfile = 'debug.txt'

mdata = []
with open(matlabfile,'r') as mfile:
    for lines in mfile:
        if len(mdata) > 25410:
            break
        mdata.append([float(i) for i in lines.split()])

mdata = np.array(mdata)
print(mdata.shape)

mtime = []
with open(matlabtime,'r') as mfile:
    for lines in mfile:
        val = float(lines)
        mtime.append(val)

mtime = np.array(mtime)
print(mtime.shape)

pdata = []
with open(pythonfile,'r') as pfile:
    for lines in pfile:
        pdata.append([float(i) for i in lines.split()])
        if [float(i) for i in lines.split()][0] > 1700:
            break

pdata = np.array(pdata)

ptime = pdata[:,0]
pdata = np.delete(pdata,0,1)

names = ['Tp','Tg','mdotO_r','mdotO_dmax','mdotO','mdotFeO','hqFeO','hAdelT','hO2','hDot','mFe','mFeO','rp']

# for k in range(len(mdata[0])):

#     plt.figure(num=k,figsize=(3,3),dpi=300)
#     plt.plot(mtime,mdata[:,k],label='MATLAB')
#     plt.plot(ptime,pdata[:,k],label='Python')
#     plt.legend()
#     plt.title('%s' %names[k])
#     plt.tight_layout()
#     plt.savefig('images_new/%s.png' %names[k])

for k2 in range(len(mdata[0])-2):

    plt.figure(num=k2+1,figsize=(3,3),dpi=300)
    plt.plot(mdata[:,0],mdata[:,2+k2],label='MATLAB')
    plt.plot(pdata[:,0],pdata[:,2+k2],label='Python')
    plt.legend()
    plt.title('%s' %names[k2 + 2])
    plt.tight_layout()
    plt.savefig('images_new/%s.png' %names[k2+2])

# plt.figure(num=k+1,figsize=(3,3),dpi=300)
# plt.plot(mdata[:,0],mdata[:,3],label='MATLAB')
# plt.plot(pdata[:,0],pdata[:,3],label='Python')
# plt.legend()
# plt.title('mDO2max')
# plt.tight_layout()
# plt.savefig('mDO2max.png')

# plt.figure(num=k+2,figsize=(3,3),dpi=300)
# plt.plot(ptime,pdata[:,-4],label='B1')
# plt.plot(ptime,pdata[:,-3],label='B2')
# plt.tight_layout()
# plt.savefig('images/beta.png')

# plt.figure(num=k+3,figsize=(3,3),dpi=300)
# plt.plot(ptime,pdata[:,-2],label='R1')
# plt.plot(ptime,pdata[:,-1],label='R2')
# plt.tight_layout()
# plt.savefig('images/rho_o2.png')

# plt.figure(num=1, figsize=(10,2))
# fig,axs = plt.subplots(nrows=2, ncols=2, figsize=(10,2), dpi=300)
# axs.flatten()

# axs[0].plot()