from typing import Dict, List
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
import matplotlib.pyplot as plt

WORKSPACE = np.array(((0.10, -0.05), # ((min_x, min_y)
                      (0.20, 0.05))) #  (max_x, max_y))
IMG_SIZE = 42


def plot_example_predictions(network: nn.Module,
                             samples: Dict[str, Tensor],
                             show: bool=True,
                            ) -> None:
    n_rows = 4
    n_cols = 5
    f, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols, n_rows))

    # swap from (B,C,H,W) to (B,H,W,C) for plotting
    imgs = torch.permute(samples['img'], (0, 2, 3, 1))
    gt_actions = samples['action'].cpu()

    with torch.no_grad():
        pred_actions = network( samples['img'] ).cpu()

    for r in range(n_rows):
        for c in range(n_cols):
            i = r * n_cols + c
            axs[r,c].imshow(imgs[i])
            add_action_to_plot(gt_actions[i], axs[r,c], color='k')
            add_action_to_plot(pred_actions[i], axs[r,c], color='b')
            if r==0 and c == n_cols-1:
                axs[r,c].legend(['ground truth', 'predicted'],
                                loc='center',
                                bbox_to_anchor=(0.5, 1.3),
                                prop=dict(size=8))

    [a.axis('off') for a in axs.flatten()]
    if show:
        plt.show()

def add_action_to_plot(action: Tensor, ax, color: str='k'):
    '''action should be (x,y,th)'''
    pxy = convert_xy_to_pxy(action[:2].numpy())

    ax.plot(pxy[1], pxy[0], color+'-', linewidth=1)
    c, s = np.cos(action[2]), np.sin(action[2])
    ax.arrow(pxy[1], pxy[0], 5*c, -5*s, width=0.5, head_width=3, color=color,
             head_length=0.2, length_includes_head=False)
    ax.arrow(pxy[1], pxy[0], -5*c, 5*s, width=0.5, head_width=3, color=color,
             head_length=0.2, length_includes_head=False)

def plot_loss_curve(log: List, show: bool=True) -> None:
    epochs = [d['epoch_id'] for d in log]
    train_loss = [d['train_loss'] for d in log]
    test_loss = [d['test_loss'] for d in log]

    plt.figure()
    plt.plot(epochs, train_loss, label='train')
    plt.plot(epochs, test_loss, label='test')
    plt.xlabel('epochs')
    plt.ylabel('mse loss')
    plt.ylim(0, 0.0008)
    plt.legend()
    if show:
        plt.show()

def convert_xy_to_pxy(xy: np.ndarray):
    # normalize coords from 0 to 1
    xy_norm = np.subtract(xy, WORKSPACE[0])/np.subtract(*WORKSPACE[::-1])
    xy_norm = np.clip(xy_norm, 0, 1)

    #xy axis are flipped from world to image space
    pxy = IMG_SIZE * xy_norm
    return pxy.astype(int)

def clamp_rotation(rot: float):
    '''Sets rotation angle to be between -PI/2 and PI/2
    '''
    rot = np.fmod(rot, 2*np.pi)
    if rot < -np.pi/2:
        rot += np.pi
    elif rot > np.pi/2:
        rot -= np.pi
    return rot

