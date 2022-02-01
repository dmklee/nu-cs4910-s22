import datetime
import numpy as np
import torch
import torch.nn as nn

from networks import GraspPredictorNetwork
from dataset import get_data_loaders, RandomReflectionTransform
import utils

def train(model_name: str,
          train_data_path: str,
          test_data_path: str,
          n_epochs: int,
          learning_rate: float,
          batch_size: int,
          use_augmentation: bool=False,
          seed: int=0):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed(seed)

    # load datasets
    train_loader, test_loader = get_data_loaders(train_data_path,
                                                 test_data_path,
                                                 batch_size,
                                                 use_augmentation)

    # pop sample off dataloader
    sample = train_loader.dataset[0]
    img_shape = sample['img'].shape
    action_size = sample['action'].shape[0]

    # create network and send to device
    network = GraspPredictorNetwork(img_shape, action_size)
    network = network.to(device)
    network.train()

    # create optimizer
    optim = torch.optim.Adam(network.parameters(), lr=learning_rate)

    log = []
    last_epoch_timestamp = datetime.datetime.now()
    for epoch_id in range(1, n_epochs+1):

        train_loss = 0
        for batch_idx, batch_data in enumerate(train_loader):
            # send to device
            imgs = batch_data['img'].to(device)
            actions = batch_data['action'].to(device)

            pred_actions = network(imgs)

            loss = network.compute_loss(pred_actions, actions)
            optim.zero_grad()
            loss.backward()
            optim.step()

            train_loss += loss.item()

        train_loss /= batch_idx+1

        test_loss = 0
        network.eval()
        for batch_idx, batch_data in enumerate(test_loader):
            imgs = batch_data['img'].to(device)
            actions = batch_data['action'].to(device)

            with torch.no_grad():
                pred_actions = network(imgs)
                loss = network.compute_loss(pred_actions, actions)

            test_loss += loss.item()
        network.train()

        test_loss /= batch_idx+1

        elapsed_seconds = (datetime.datetime.now() - last_epoch_timestamp).total_seconds()
        log.append(dict(epoch_id=epoch_id,
                        train_loss=train_loss,
                        test_loss=test_loss,
                        duration=elapsed_seconds))
        print(f"Epoch {epoch_id}/{n_epochs} | Train: {train_loss:.6f} | " \
              f"Test: {test_loss:.6f} | Duration: {elapsed_seconds:.2f} s")

        last_epoch_timestamp = datetime.datetime.now()

    # save network for later
    torch.save(network.state_dict(), model_name + '.pt')

    utils.plot_loss_curve(log, show=False)
    network.eval()
    utils.plot_example_predictions(network.cpu(), batch_data)


if __name__ == "__main__":
    train(model_name='model',
          train_data_path='mini_dataset.hdf5',
          test_data_path='mini_dataset.hdf5',
          n_epochs=5,
          learning_rate=1e-3,
          batch_size=128,
          use_augmentation=False,
         )
