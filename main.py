import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from matplotlib.animation import PillowWriter


# dz = dx (change it later to work on a not regular grid!)
def laplacian(x, z, dx, P):
    return (P[x + 1, z] + P[x - 1, z] + P[x, z + 1] + P[x, z - 1] - 4 * P[x, z]) / (dx ** 2)

writer = PillowWriter(fps=15)

x_grid_size = 200
z_grid_size = x_grid_size
delta_x = .5
delta_z = delta_x

total_time_steps = 1000
delta_time = 0.0010

source_x = int(x_grid_size / 2)
source_z = int(z_grid_size / 2)

c0 = 250.

epsilon = c0 * delta_time / delta_x

print(f'CFL Epsilon: {epsilon}\n')

f0 = 500
t0 = 1. / f0

source = np.zeros(total_time_steps + 1)
time = np.linspace(0, total_time_steps * delta_time, total_time_steps)

source = -8. * (time - t0) * f0 * \
    (np.exp(-1. * (time - t0) ** 2 * (f0 * 4) ** 2))

p_present = np.zeros((x_grid_size, z_grid_size))
p_past = np.zeros((x_grid_size, z_grid_size))
p_future = np.zeros((x_grid_size, z_grid_size))

c = np.zeros((x_grid_size, z_grid_size))
c += c0

fig = plt.figure()
writer.setup(fig, 'teste.gif', 100)

for t in range(total_time_steps):
    for x in range(1, x_grid_size - 1):  # 1 - 498
        for z in range(1, z_grid_size - 1):  # 1 - 498
            p_future[x, z] = (c[x, z] ** 2) * laplacian(x, z, delta_x, p_present) * \
                (delta_time ** 2) + 2 * p_present[x, z] - p_past[x, z]

    p_past = p_present
    p_present = p_future

    p_future[source_x, source_z] += source[t]
    
    plt.plot(p_future)

    writer.grab_frame()

writer.finish()