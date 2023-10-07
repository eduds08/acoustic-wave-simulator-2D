import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import laplace
from PIL import Image

# def laplacian(x, z, dx, P):
#     return (P[x + 1, z] + P[x - 1, z] + P[x, z + 1] + P[x, z - 1] - 4 * P[x, z]) / (dx ** 2)


def create_gif(total_time):
    imgs = []
    for t in range(total_time):
        imgs.append(Image.open(f"images/plot_{t}.png"))

    imgs[0].save('simulation.gif', format='GIF',
                 append_images=imgs[1:],
                 save_all=True,
                 duration=50, loop=0)


x_grid_size = 500
z_grid_size = x_grid_size
delta_x = 1
delta_z = delta_x

total_time_steps = 1000
delta_time = 0.001

source_x = int(x_grid_size / 2)
source_z = int(z_grid_size / 2)

c0 = 350

cfl = c0 * delta_time / delta_x
print(cfl)

f0 = 10
t0 = 2 / f0

source = np.zeros(total_time_steps + 1)
time = np.linspace(0, total_time_steps * delta_time, total_time_steps)

source = -8. * (time - t0) * f0 * \
    (np.exp(-1. * (time - t0) ** 2 * (f0 * 4) ** 2))

plt.figure()
plt.plot(source)
plt.show()

p_present = np.zeros((z_grid_size, x_grid_size))
p_past = np.zeros((z_grid_size, x_grid_size))
p_future = np.zeros((z_grid_size, x_grid_size))

c = np.zeros((z_grid_size, x_grid_size))
c += c0

for t in range(total_time_steps):

    p_future = (c ** 2) * laplace(p_present) * (delta_time ** 2)

    p_future += 2 * p_present - p_past

    # for x in range(1, x_grid_size - 1):
    #     for z in range(1, z_grid_size - 1):

    #         p_future[x, z] = (c[x, z] ** 2) * laplacian(x, z, delta_x, p_present) * \
    #             (delta_time ** 2) + 2 * p_present[x, z] - p_past[x, z]

    ### OPTIONAL: THREAD FOR PLOTTING FIGURE BY FIGURE WITH A PAUSE ###

    p_future[0, :] = p_present[1, :] + \
        ((cfl - 1) / (cfl + 1)) * (p_future[1, :] - p_present[0, :])

    p_future[z_grid_size - 1, :] = p_present[z_grid_size - 2, :] + \
        ((cfl - 1) / (cfl + 1)) * \
        (p_future[z_grid_size - 2, :] - p_present[z_grid_size - 1, :])

    p_future[:, 0] = p_present[:, 1] + \
        ((cfl - 1) / (cfl + 1)) * (p_future[:, 1] - p_present[:, 0])

    p_future[:, x_grid_size - 1] = p_present[:, x_grid_size - 2] + \
        ((cfl - 1) / (cfl + 1)) * \
        (p_future[:, x_grid_size - 2] - p_present[:, x_grid_size - 1])

    p_past = p_present
    p_present = p_future

    p_future[source_z, source_x] += source[t]

    plt.imsave(f"images/plot_{t}.png", p_future)

create_gif(total_time_steps)
