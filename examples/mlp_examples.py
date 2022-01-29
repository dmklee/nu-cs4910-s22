import torch
import torch.nn as nn
from torch import Tensor
import numpy as np


class MLP1(nn.Module):
    '''
    Task: design nn.Module that predicts probability of top-down grasp success
    given an object's pose
    Input: object pose (6D) and grasp location (x,y,theta)
    Output: probability of success
    Guidance: use 3 linear layers with 128 hidden units
    '''
    def __init__(self):
        super().__init__()

        # 6 for obj pose, 3 for grasp location
        input_size = 6 + 3

        # only one value for probability of success
        output_size = 1

        # as noted in guidance, we use 3 linear layers here, with 128 units
        # we will apply the activation functions in forward pass
        self.layer1 = nn.Linear(input_size, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, output_size)

        # since we are predicting probability, it makes sense to use
        # a binary cross entropy loss
        self.loss_fn = nn.BCELoss()

    def forward(self, obj_pose: Tensor, grasp_loc: Tensor) -> Tensor:
        '''Performs forward pass on a batch of size B

        Parameters
        ----------
        obj_pose
            pose of the object; tensor of shape (B, 6), dtype=torch.float32
        grasp_loc
            location of top down grasp (x,y,th); tensor of shape (B, 3),
            dtype=torch.float32.  As we said in class, it might make sense to
            convert theta to <cos(theta), sin(theta)> here.

        Returns
        -------
        Tensor
            probability value of success, ranging from 0 to 1; tensor of shape
            (B,1) and dtype=torch.float32
        '''
        # first we need to concatenate the inputs into single vector
        x = torch.cat([obj_pose, grasp_loc], dim=1)

        # make sure to include relu between layers
        x = self.layer1(x)
        x = torch.relu(x)
        x = self.layer2(x)
        x = torch.relu(x)
        x = self.layer3(x)

        # sigmoid since we want output to be probability from 0 to 1
        x = torch.sigmoid(x)
        return x

    def compute_loss(self, y_pred: Tensor, y_target: Tensor) -> Tensor:
        '''Computes loss between predicted grasp success and ground truth grasp
        success using BCE loss

        Parameters
        ----------
        y_pred
            predicted grasp success; tensor of shape (B, 1); dtype=torch.float32
            this is the output of calling self.forward
        y_target
            ground truth grasp success from a dataset; tensor of shape (B,1);
            dtype=torch.float32.

        Returns
        -------
        Tensor
            scalar value representing loss over entire batch
        '''
        # the order here matters since we are using BCELoss !
        return self.loss_fn(y_pred, y_target)


class MLP2(nn.Module):
    '''
    Task: design nn.Module that predicts location of top-down grasp (x,y,th)
    given an object's pose
    Input: object pose (6D)
    Output: grasp location in SE(2)-space (x,y,th)
    Guidance: use multihead MLP. the backbone is 2 layer MLP with 64 units,
    the first head has 2 linear layers with 64 hidden units and predicts grasp
    position (x,y). the second head has 2 linear layers with 128 units and
    predicts grasp orientation (th)
    '''
    def __init__(self):
        super().__init__()

        # 6 for obj pose
        input_size = 6

        # I didnt specify in class, but lets use 64 units in between backbone
        # and the heads
        hidden_size = 64

        # I will show sequential here, since it makes the forward pass shorter
        self.backbone = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(True),   # make sure to add relu between all layers
            nn.Linear(64, hidden_size),
            nn.ReLU(True),
        )

        self.head1 = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(True),   # make sure to add relu between all layers
            nn.Linear(64, 64),
            nn.ReLU(True),
            nn.Linear(64, 2),
        )

        self.head2 = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(True),   # make sure to add relu between all layers
            nn.Linear(128, 1),
            nn.Tanh(),
        )

        # since we are performing regression, lets use MSE
        self.loss_fn = nn.MSELoss()

    def forward(self, obj_pose: Tensor) -> Tensor:
        '''Performs forward pass on a batch of size B

        Parameters
        ----------
        obj_pose
            pose of the object; tensor of shape (B, 6), dtype=torch.float32

        Returns
        -------
        Tensor
            location of top down grasp (x,y,th); tensor of shape (B, 3),
            dtype=torch.float32, here we limit theta to be between -PI/2 and PI/2
        '''
        x = self.backbone(obj_pose)

        # to do multi-head, just send output from backbone to multiple nn.Modules
        xy_pos = self.head1(x)
        th = np.pi/2 * self.head2(x)

        # now we concatenate outputs into single vector
        grasp_loc = torch.cat([xy_pos, th], dim=1)

        return grasp_loc

    def compute_loss(self, y_pred: Tensor, y_target: Tensor) -> Tensor:
        '''Computes loss between predicted grasp success and ground truth grasp
        success using BCE loss

        Parameters
        ----------
        y_pred
            predicted grasp location; tensor of shape (B,3); dtype=torch.float32
            this is the output of calling self.forward
        y_target
            ground truth grasp location; tensor of shape (B,3);
            dtype=torch.float32

        Returns
        -------
        Tensor
            scalar value representing loss over entire batch
        '''
        return self.loss_fn(y_pred, y_target)
