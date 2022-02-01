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
    def __init__(self, img_shape: Size, action_size: int=3):
        '''Network that predicts (x,y,theta) from images

        Parameters
        ----------
        img_shape
            shape of image tensor (C, H, W)
        action_size
            size of action vector
        '''
        super().__init__()

        raise NotImplemented

    def forward(self,
                img: Tensor,
               ) -> Tensor:
        raise NotImplemented

    def compute_loss(self,
                     action_pred: Tensor,
                     action_target: Tensor,
                    ) -> Tensor:
        raise NotImplemented
