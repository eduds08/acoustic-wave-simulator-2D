import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

# dz = dx (change it later to work on a not regular grid!)


def laplacian(x, z, dx, P):
    return (P[x + 1, z] + P[x - 1, z] + P[z + 1, x] + P[z - 1, x] - 4 * P[x, z]) / (dx ** 2)


x_grid_size = 500
z_grid_size = x_grid_size
delta_x = 1
delta_z = delta_x

total_time_steps = 500
delta_time = 0.0010

p_present = np.zeros((x_grid_size, z_grid_size))
p_past = np.zeros((x_grid_size, z_grid_size))
p_future = np.zeros((x_grid_size, z_grid_size))

# source s = -8 f0 * (t - t0) * exp(...)

c = 1  # temporario

for t in range(total_time_steps):
    for x in range(1, x_grid_size - 1):  # 1 - 498
        for z in range(1, z_grid_size - 1):  # 1 - 498
            p_future[x, z] = (c ** 2) * laplacian(x, z, delta_x, p_present) * \
                (delta_time ** 2) + 2 * p_present[x, z] - p_past[x, z]
