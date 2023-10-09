import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import laplace
from PIL import Image
import cv2


def laplacian_5_operator(size_z, size_x, dz, dx, p):
    pzz = np.zeros((size_z, size_x))
    pxx = np.zeros((size_z, size_x))

    for z in range(2, size_z - 2):
        pzz[z, :] = ((-1/12) * p[z + 2, :] + (4/3) * p[z + 1, :] - (5/2) *
                     p[z, :] + (4/3) * p[z - 1, :] - (1/12) * p[z - 2, :]) / (dz ** 2)
    for x in range(2, size_x - 2):
        pxx[:, x] = ((-1/12) * p[:, x + 2] + (4/3) * p[:, x + 1] - (5/2) *
                     p[:, x] + (4/3) * p[:, x - 1] - (1/12) * p[:, x - 2]) / (dx ** 2)

    return pzz + pxx


def create_gif_and_video(total_time):
    images_for_gif = []
    images_for_video = []
    for t in range(total_time):
        images_for_gif.append(Image.open(f"images/plot_{t}.png"))

        img = cv2.imread(f"images/plot_{t}.png")
        height, width, layers = img.shape
        size = (width, height)
        images_for_video.append(img)

    out = cv2.VideoWriter(
        'project.avi', cv2.VideoWriter_fourcc(*'DIVX'), 15, size)

    for i in range(len(images_for_video)):
        out.write(images_for_video[i])
    out.release()

    images_for_gif[0].save('simulation.gif', format='GIF',
                           append_images=images_for_gif[1:],
                           save_all=True,
                           duration=50, loop=0)


grid_size_x = 300
grid_size_z = grid_size_x
dx = 1
dz = dx

total_time = 1000
dt = 0.001

source_x = int(grid_size_x / 2)
source_z = int(grid_size_z / 2)

c0 = 450

cfl = c0 * dt / dx
print(cfl)

f0 = 10
t0 = 2 / f0

source = np.zeros(total_time + 1)
time = np.linspace(0, total_time * dt, total_time)

source = -8. * (time - t0) * f0 * \
    (np.exp(-1. * (time - t0) ** 2 * (f0 * 4) ** 2))

plt.figure()
plt.plot(source)
plt.show()

p_present = np.zeros((grid_size_z, grid_size_x))
p_past = np.zeros((grid_size_z, grid_size_x))
p_future = np.zeros((grid_size_z, grid_size_x))

c = np.zeros((grid_size_z, grid_size_x))
c += c0

for t in range(total_time):

    # p_future = (c ** 2) * laplace(p_present) * (dt ** 2)
    p_future = (c ** 2) * laplacian_5_operator(grid_size_z,
                                               grid_size_x, dz, dx, p_present) * (dt ** 2)
    p_future += 2 * p_present - p_past

    # ABORSÇÃO PARA 5 OPERADORES:
    p_future[1, :] = p_present[2, :] + \
        ((cfl - 1) / (cfl + 1)) * (p_future[2, :] - p_present[1, :])

    p_future[grid_size_z - 2, :] = p_present[grid_size_z - 3, :] + \
        ((cfl - 1) / (cfl + 1)) * \
        (p_future[grid_size_z - 3, :] - p_present[grid_size_z - 2, :])

    p_future[:, 1] = p_present[:, 2] + \
        ((cfl - 1) / (cfl + 1)) * (p_future[:, 2] - p_present[:, 1])

    p_future[:, grid_size_x - 2] = p_present[:, grid_size_x - 3] + \
        ((cfl - 1) / (cfl + 1)) * \
        (p_future[:, grid_size_x - 3] - p_present[:, grid_size_x - 2])

    p_future[0, :] = p_present[1, :] + \
        ((cfl - 1) / (cfl + 1)) * (p_future[1, :] - p_present[0, :])

    p_future[grid_size_z - 1, :] = p_present[grid_size_z - 2, :] + \
        ((cfl - 1) / (cfl + 1)) * \
        (p_future[grid_size_z - 2, :] - p_present[grid_size_z - 1, :])

    p_future[:, 0] = p_present[:, 1] + \
        ((cfl - 1) / (cfl + 1)) * (p_future[:, 1] - p_present[:, 0])

    p_future[:, grid_size_x - 1] = p_present[:, grid_size_x - 2] + \
        ((cfl - 1) / (cfl + 1)) * \
        (p_future[:, grid_size_x - 2] - p_present[:, grid_size_x - 1])

    p_past = p_present
    p_present = p_future

    p_future[source_z, source_x] += source[t]

    plt.imsave(f"images/plot_{t}.png", p_future)

create_gif_and_video(total_time)
