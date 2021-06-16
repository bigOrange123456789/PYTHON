#1
from __future__ import print_function # only necessary if using Python 2.x
import matplotlib.pyplot as plt
import numpy as np

# input the global topo data
topo = np.loadtxt('etopo5.dat')

#2
from pyshtools.shclasses import SHCoeffs, SHWindow, SHGrid

# reverse the data by rows
grid_topo = SHGrid.from_array(np.flip(topo,0)[:-1,:])

# print the information of the grid data
#grid_topo.info()

# expand the grid into spherical harmonics.将栅格展开为球谐函数
coeffs_topo = grid_topo.expand()

# print the information of the spherical harmonic coefficients
#coeffs_topo.info()

# print the coefficients of spherical harmonics打印球谐函数的系数
print("\ncoeffs_topo.coeffs",coeffs_topo.coeffs)
