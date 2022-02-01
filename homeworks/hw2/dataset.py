from typing import Optional, Dict, Callable, Tuple
import torch
from torch import Tensor
import numpy as np
import h5py
import matplotlib.pyplot as plt

from utils import add_action_to_plot

from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor


class RandomReflectionTransform:
    def __init__(self, p: float=0.5):
        self.p = p

    def __call__(self,
                 sample: Dict[str, Tensor],
                ) -> Dict[str, Tensor]:
        '''Applies transformation to sample by reflecting image about vertical
        axis (e.g. x axis of simulator) with probability of self.p (i.e. if
        self.p=0.5, then the transformation should occur 50% of the time)

        Parameters
        ----------
        sample
            'img' : float image tensor with shape (C, H, W),
            'action' : float tensor of action (x,y,theta) with shape (3)}

        Return
        ------
            'img' : float image tensor with shape (C, H, W), with the values
            flipped flipped about vertical axis
            'action' : float tensor of action (x,y,theta) with shape (3) with
            the position and rotation flipped around x-axis
        '''
        raise NotImplemented


class SuccessfulGraspDataset(Dataset):
    def __init__(self,
                 path_to_hdf5: str,
                 transform: Optional[Callable]=None,
                ):
        '''Dataset of successful grasps, stored as top down images and actions

        Attributes
        ----------
        transform : Optional[Callable]
            callable function that applies some data augmentation to a data
            sample, applied when indexing into this class. If None, data samples
            are not transformed
        imgs : np.ndarray
            array of rgb images with dtype uint8 and shape (L, H, W, C),
            where L is number of samples, H is image height, W is image width,
            and C is number of channels (3 for RGB)
        actions : np.ndarray
            top down grasp actions parametrized by (x,y,theta). theta is
            constrained from -PI/2 to PI/2. dtype = float, shape = (L, 3)
        '''

        self.transform = transform

        with h5py.File(path_to_hdf5, 'r') as hf:
            self.imgs = np.array(hf['imgs']) # shape (L, H, W, C)
            self.actions = np.array(hf['actions']) # shape (L, 3)

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        '''Indices into dataset and returns tensor of each component. If
        self.transform is not None, you should apply the transform to the sample.
        Do not send tensors to another device at this point

        Hint
        ----
        see ToTensor, for a quick way of reformating image as tensor

        Parameters
        ----------
        idx
            index of data sample

        Returns
        -------
        Dict[str, Tensor]
            keys are ('img', 'action').  value of 'img' is a float tensor with
            values ranging from 0 to 1 and shape (C,H,W). value of 'action' is
            float tensor of shape (3)
        '''
        raise NotImplemented

    def __len__(self):
        return len(self.imgs)


def get_data_loaders(train_dataset_path: str,
                     test_dataset_path: str,
                     batch_size: int,
                     use_augmentation: bool=False,
                    ) -> Tuple[DataLoader, DataLoader]:
    transform = RandomReflectionTransform() if use_augmentation else None
    train_dataset = SuccessfulGraspDataset(train_dataset_path, transform)

    test_dataset = SuccessfulGraspDataset(test_dataset_path, transform=None)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader

def visualize_dataset(dataset_path: str):
    dataset = SuccessfulGraspDataset(dataset_path)

    f, axs = plt.subplots(4, 7, figsize=(7,4))
    for i, ax in enumerate(axs.flatten()):
        sample = dataset[i]
        ax.imshow(torch.permute(sample['img'], (1, 2, 0)))
        add_action_to_plot(sample['action'], ax)
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    plt.suptitle(dataset_path)
    plt.tight_layout()
    plt.show()

def test_reflection_augmentation(dataset_path: str):
    dataset = SuccessfulGraspDataset(dataset_path)

    transform = RandomReflectionTransform(p=0.)

    f, axs = plt.subplots(4, 2, figsize=(4,8))
    for i, (ax0, ax1) in enumerate(axs):
        sample = dataset[i]
        ax0.imshow(torch.permute(sample['img'], (1, 2, 0)))
        add_action_to_plot(sample['action'], ax0)
        ax0.set_xticklabels([])
        ax0.set_yticklabels([])

        aug_sample = transform(sample)
        ax1.imshow(torch.permute(aug_sample['img'], (1, 2, 0)))
        add_action_to_plot(aug_sample['action'], ax1)
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])

        if i == 0:
            ax0.set_title('Original')
            ax1.set_title('Reflected')

    plt.suptitle(dataset_path)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    visualize_dataset('mini_dataset.hdf5')

    # test_reflection_augmentation('mini_dataset.hdf5')
