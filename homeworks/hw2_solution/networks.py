from typing import Tuple, List, Optional, Dict
import torch
import torch.nn as nn
from torch import Tensor, Size

class WeightedMSELoss(nn.Module):
    def __init__(self, weights: Tensor):
        super().__init__()
        self.weights = nn.parameter.Parameter(weights, requires_grad=False)

    def __call__(self, y_pred: Tensor, y_target: Tensor) -> Tensor:
        squared_error = torch.square(y_pred - y_target)

        loss = torch.mean(squared_error * self.weights)
        return loss


class GraspPredictorNetwork(nn.Module):
    def __init__(self, img_shape: Size, action_size: int):
        super().__init__()
        self.input_shape = img_shape
        self.output_size = action_size

        self.cnn = nn.Sequential(
            nn.Conv2d(self.input_shape[0], 16, 3, 2, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, 3, 2, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, 3, 2, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(64),
            nn.Flatten(),
        )

        #ToDo: determine the size of your latent vector
        self.latent_size = self.get_cnn_output_size()

        self.mlp = nn.Sequential(
            nn.Linear(self.latent_size, 256),
            nn.ReLU(True),
            nn.Linear(256, 256),
            nn.ReLU(True),
            nn.Linear(256, self.output_size),
        )

        # self.loss_fn = nn.MSELoss()
        loss_weights = torch.tensor([1,1, 0.001], dtype=torch.float32)
        self.loss_fn = WeightedMSELoss(loss_weights)

    def get_cnn_output_size(self):
        x = torch.zeros(self.input_shape).unsqueeze(0)
        return self.cnn(x).size(1)

    def forward(self,
                img: Tensor,
               ) -> Tensor:
        x = self.cnn(img)
        x = self.mlp(x)
        return x

    def compute_loss(self,
                     action_pred: Tensor,
                     action_target: Tensor,
                    ) -> Tensor:

        return self.loss_fn(action_pred, action_target)
