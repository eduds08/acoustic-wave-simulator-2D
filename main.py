import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import laplace
from PIL import Image

# def laplacian(x, z, dx, P):
#     return (P[x + 1, z] + P[x - 1, z] + P[x, z + 1] + P[x, z - 1] - 4 * P[x, z]) / (dx ** 2)

x_grid_size = 100
z_grid_size = x_grid_size
delta_x = 1
delta_z = delta_x

total_time_steps = 502
delta_time = 0.0010

source_x = int(x_grid_size / 2)
source_z = int(z_grid_size / 2)

c0 = 340

epsilon = c0 * delta_time / delta_x

print(f'CFL Epsilon: {epsilon}\n')

f0 = 25
t0 = 2 / f0

source = np.zeros(total_time_steps + 1)
time = np.linspace(0, total_time_steps * delta_time, total_time_steps)

source = -8. * (time - t0) * f0 * \
    (np.exp(-1. * (time - t0) ** 2 * (f0 * 4) ** 2))

p_present = np.zeros((x_grid_size, z_grid_size))
p_past = np.zeros((x_grid_size, z_grid_size))
p_future = np.zeros((x_grid_size, z_grid_size))

c = np.zeros((x_grid_size, z_grid_size))
c += c0

for t in range(total_time_steps):

    p_future = (c ** 2) * laplace(p_present) * (delta_time ** 2)

    p_future += 2 * p_present - p_past

    # for x in range(1, x_grid_size - 1):
    #     for z in range(1, z_grid_size - 1):

    #         p_future[x, z] = (c[x, z] ** 2) * laplacian(x, z, delta_x, p_present) * \
    #             (delta_time ** 2) + 2 * p_present[x, z] - p_past[x, z]

    p_past = p_present
    p_present = p_future

    p_future[source_x, source_z] += source[t]

    plt.imsave(f"imagens/image_filename{t}.png", p_future)

frames = []
imgs = []
for t in range(total_time_steps):
    imgs.append(f"imagens/image_filename{t}.png")
for i in imgs:
    new_frame = Image.open(i)
    frames.append(new_frame)

frames[0].save('simulation.gif', format='GIF',
               append_images=frames[1:],
               save_all=True,
               duration=50, loop=0)
