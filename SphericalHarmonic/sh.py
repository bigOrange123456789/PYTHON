from __future__ import print_function # only necessary if using Python 2.x
import matplotlib.pyplot as plt
import numpy as np
from pyshtools.shclasses import SHCoeffs, SHWindow, SHGrid

# input the global topo data
topo = np.loadtxt('etopo5.dat')

# reverse the data by rows
grid_topo = SHGrid.from_array(np.flip(topo,0)[:-1,:])

# expand the grid into spherical harmonics.将栅格展开为球谐函数
coeffs_topo = grid_topo.expand()

# print the coefficients of spherical harmonics打印球谐函数的系数
print("\ncoeffs_topo.coeffs",coeffs_topo.coeffs)