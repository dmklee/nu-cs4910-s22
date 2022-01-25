import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

DATA = np.array([[0,0],[0,1],[1,0],[1,1]])
LABELS = np.array([0, 1, 1, 0])


class XORSolverNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        # add your layers here

        raise NotImplemented

    def forward(self, x: Tensor) -> Tensor:
        '''Perform forward pass using input x'''
        raise NotImplemented

    @torch.no_grad()
    def predict(self, x: Tensor) -> Tensor:
        '''predict labels of dtype int and shape=(x.size(0),)'''
        raise NotImplemented

    def compute_loss(self, x: Tensor, y: Tensor) -> Tensor:
        '''Compute loss on a given input tensor x and ground truth label y
        '''
        raise NotImplemented

def train(network: nn.Module,
          data: np.ndarray,
          labels: np.ndarray,
          n_epochs: int=2000,
          learning_rate: float=5e-1,
         ):
    # convert data, labels to tensors
    raise NotImplemented

    # create optimizer
    optimizer = torch.optim.SGD(network.parameters(), lr=learning_rate)

    for epoch_id in range(n_epochs):
        loss = network.compute_loss(data, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f'[Epoch {epoch_id}] Train loss: {loss.item():.3f}')

    return network

def visualize_predictions(network):
    extent = np.linspace(-0.2, 1.2, num=20, endpoint=True)
    x = np.dstack(np.meshgrid(extent, extent)).reshape(-1, 2)

    x_tensor = torch.tensor(x, dtype=torch.float32)
    pred = network.predict(x_tensor).detach().numpy()

    low_x = x[pred==0]
    high_x = x[pred==1]

    plt.figure()
    # plot predictions
    plt.plot(*low_x.T, 'rx')
    plt.plot(*high_x.T, 'bx')

    # plot dataset
    plt.plot([0,1],[0,1], 'ro', label='Output = 0')
    plt.plot([0,1],[1,0], 'bo', label='Output = 1')

    plt.legend()
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()

if __name__ == "__main__":
    network = XORSolverNetwork()

    network = train(network,
                    DATA,
                    LABELS)

    visualize_predictions(network)
