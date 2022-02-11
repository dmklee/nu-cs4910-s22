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
        if np.random.random() < self.p:
            new_img = torch.flip(sample['img'], [2]) # flip along W dimension
            new_action = sample['action']
            new_action[1] *= -1  # negate y-value
            new_action[2] *= -1  # flip theta around x-axis

            return dict(img=new_img, action=new_action)
        else:
            return sample


class SuccessfulGraspDataset(Dataset):
    def __init__(self,
                 path_to_hdf5: str,
                 transform: Optional[Callable]=None,
                ):
        self.transform = transform

        with h5py.File(path_to_hdf5, 'r') as hf:
            self.imgs = np.array(hf['imgs'])
            self.actions = np.array(hf['actions'])

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        img = ToTensor()(self.imgs[idx])

        action = torch.tensor(self.actions[idx], dtype=torch.float32)

        sample = {'img' : img, 'action' : action}

        if self.transform:
            sample = self.transform(sample)

        return sample

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

    transform = RandomReflectionTransform(p=1)

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

#ToDo: add some functionality to test the reflection transform
if __name__ == "__main__":
    visualize_dataset('test_dataset.hdf5')
    # test_reflection_augmentation('test_dataset.hdf5')
