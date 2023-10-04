import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

# [ p(j, k, n+1) - 2p(j, k, n) + p(j, k, n-1) ] / dt²
# [ p(j+1, k, n) - 2p(j, k, n) + p(j-1, k, n) ] / dx²
# [ p(j, k+1, n) - 2p(j, k, n) + p(j, k-1, n) ] / dz²
# c(j, k) ** 2
# s(j, k, n)

# CFL = c * (dt/dx)

# c < 0.5

"""

# Time extrapolation
for it in range(nt):

    # calculate partial derivatives (omit boundaries)
    for i in range(1, nx - 1):
        d2p[i] = (p[i + 1] - 2 * p[i]\
        + p[i - 1]) / dx ** 2

    # Time extrapolation
    pnew = 2 * p - pold + dt ** 2 * c ** 2 * d2p

    # Add source term at isrc
    pnew[isrc] = pnew[isrc] + dt ** 2 * src[it] / dx

    # Remap time levels
    pold, p = p, pnew

"""

# nt -> número máximo de time steps
# nx -> número de grid point para x
# isrc -> ponto da grid onde foi injetado o source
# src[it] -> função do source time scaled pelo incremento da grid
# nesse caso c é constante.

x_max = 10
nx = 100
dx = x_max / nx  # dx = x_max / (nx - 1)

z_max = 10
nz = 100
dz = z_max / nz  # dz = z_max / (nz - 1)

i_src = (int(nx/2), int(nz/2))

nt = 1000
dt = 0.0010

p = np.zeros((nx, nz))
p_old = np.zeros((nx, nz))
p_new = np.zeros((nx, nz))

c0 = 334.0

c = np.zeros((nx, nz))
c = c + c0

'''

# 1D Wave Propagation (Finite Difference Solution) 
# ------------------------------------------------

# Loop over time
for it in range(nt):

    # 2nd derivative in space
    for i in range(1, nx - 1):
        d2px[i] = (p[i + 1] - 2 * p[i] + p[i - 1]) / dx ** 2


    # Time Extrapolation
    # ------------------
    pnew = 2 * p - pold + c ** 2 * dt ** 2 * d2px

    # Add Source Term at isrc
    # -----------------------
    # Absolute pressure w.r.t analytical solution
    pnew[isrc] = pnew[isrc] + src[it] / (dx) * dt ** 2
    
            
    # Remap Time Levels
    # -----------------
    pold, p = p, pnew

'''