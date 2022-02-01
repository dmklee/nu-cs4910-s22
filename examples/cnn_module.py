import torch
import torch.nn as nn
from torch import Tensor, Size


class CNN(nn.Module):
    def __init__(self, img_shape: Size):
        '''Convolulational neural network that outputs a vector to be processed
        by an MLP.  The network is composed of 4 conv layers with 32 channels;
        the first layer uses stride of 2, the rest use stride of 1; all layers
        have a kernel size of 3 with no padding.  After every layer, use a ReLU

        Parameters
        ----------
        img_shape
            shape of image input (C, H, W), where C is the number of channels,
            H is the height and W is the width

        Attributes
        ----------
        output_size : int
            length of vector output, this is needed to determine the number of
            input features for downstream MLP
        '''
        super().__init__()

        self.output_size = None
        pass

    def forward(self, x: Tensor) -> Tensor:
        '''Performs forward pass on batch of input images, returning vectors

        Parameters
        ----------
        x
            tensor of images, shape=(B,C,H,W), dtype=torch.float32

        Returns
        -------
        Tensor
            output vectors, shape=(B,self.output_size) of dtype=torch.float32
        '''
        pass


if __name__ == "__main__":
    pass
