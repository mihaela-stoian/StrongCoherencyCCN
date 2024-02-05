import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch

def plotting_experiment_results(experiment, model1, rect0, rect1):
    x = np.linspace(0, 1, 1000)
    y = x
    z = torch.tensor([[[i, j] for i in x] for j in y]).reshape(-1, 2).float()
    out = model1(z).detach().reshape(len(x), len(y), 2)
    z0 = out[:, :, 0]
    z1 = out[:, :, 1]
    out = model1(z)
    fout = experiment.clayer(out)
    fout = fout.detach().reshape(len(x), len(y), 2)
    fz0 = fout[:, :, 0]
    fz1 = fout[:, :, 1]

    cmap = plt.get_cmap('seismic_r')
    # levels = np.linspace(0,10,1)
    # print(levels)

    fig, ax = plt.subplots(2, 2, figsize=(10, 8))
    cs0 = ax[0,0].contourf(x, y, z0, cmap=cmap, alpha=0.9, levels=10, vmin=0, vmax=1.0)
    ax[0,0].set_title("Class A - Final Output")
    cs3 = ax[0,1].contourf(x, y, fz1, cmap=cmap, alpha=0.9, levels=10, vmin=0, vmax=1.0)
    ax[0,1].set_title("Class B - Final Output")
    cs2 = ax[1,0].contourf(x, y, fz0, cmap=cmap, alpha=0.9, levels=10, vmin=0, vmax=1.0)
    ax[1,0].set_title("Class A - Output Before Layer")
    cs1 = ax[1,1].contourf(x, y, z1, cmap=cmap, alpha=0.9, levels=10, vmin=0, vmax=1.0)
    ax[1,1].set_title("Class B - Output Before Layer")

    cbar0 = fig.colorbar(cs0)
    cbar1 = fig.colorbar(cs1)
    cbar2 = fig.colorbar(cs2)
    cbar3 = fig.colorbar(cs3)

    # Add GT rectangles
    rect0_patch = patches.Rectangle(rect0.get_left_low_corner(),rect0.get_width(),rect0.get_height(),linewidth=3,edgecolor='green',facecolor='none')
    a = ax[0,0].add_patch(rect0_patch)
    rect0_patch = patches.Rectangle(rect0.get_left_low_corner(),rect0.get_width(),rect0.get_height(),linewidth=3,edgecolor='green',facecolor='none')
    a = ax[0,1].add_patch(rect0_patch)
    rect0_patch = patches.Rectangle(rect0.get_left_low_corner(),rect0.get_width(),rect0.get_height(),linewidth=3,edgecolor='green',facecolor='none')
    a = ax[1,0].add_patch(rect0_patch)
    rect0_patch = patches.Rectangle(rect0.get_left_low_corner(),rect0.get_width(),rect0.get_height(),linewidth=3,edgecolor='green',facecolor='none')
    a = ax[1,1].add_patch(rect0_patch)

    rect1_patch = patches.Rectangle(rect1.get_left_low_corner(),rect1.get_width(),rect1.get_height(),linewidth=3,edgecolor='yellow',facecolor='none')
    a = ax[0,1].add_patch(rect1_patch)
    rect1_patch = patches.Rectangle(rect1.get_left_low_corner(),rect1.get_width(),rect1.get_height(),linewidth=3,edgecolor='yellow',facecolor='none')
    a = ax[1,1].add_patch(rect1_patch)

