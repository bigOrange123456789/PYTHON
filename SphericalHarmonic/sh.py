#1
from __future__ import print_function # only necessary if using Python 2.x

import matplotlib.pyplot as plt
import numpy as np

# input the global topo data
infile = 'etopo5.dat'
topo = np.loadtxt(infile)
# append the first column data to the end of the topo data 将第一列数据附加到拓扑数据的末尾
topo_global = np.concatenate((topo,np.vstack(topo[:,0])),axis=1)
#vstack按垂直方向（行顺序）堆叠数组构成一个新的数组，堆叠的数组需要具有相同的维度

# plot with matplotlib
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
ax.imshow(topo_global, origin='lower', extent=(0, 360, -90, 90), cmap='viridis')
ax.set(xlabel='longitude', ylabel='latitude')

#2
from pyshtools.shclasses import SHCoeffs, SHWindow, SHGrid

# reverse the data by rows
grid_topo = SHGrid.from_array(np.flip(topo,0)[:-1,:])

# print the information of the grid data
grid_topo.info()
print()

# print the topo data
print(grid_topo.data)
print()

# print the lats and lons 
print(grid_topo.lats())
print()
print(grid_topo.lons())
print()

# expand the grid into spherical harmonics.
coeffs_topo = grid_topo.expand()

# print the information of the spherical harmonic coefficients
coeffs_topo.info()
print()

# print the coefficients of spherical harmonics
print(coeffs_topo.coeffs)
