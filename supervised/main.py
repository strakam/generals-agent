import numpy as np
import os
import torch
from dataloader import ReplayDataset, per_worker_init_fn
from network import Network
from torchsummary import summary


def collate_fn(batch):
    observations = torch.from_numpy(np.array([b[0] for b in batch]))
    actions = torch.from_numpy(np.array([b[1] for b in batch]))
    return observations, actions


dataloader = torch.utils.data.DataLoader(
    ReplayDataset(
        [f"all_replays/new/{name}" for name in os.listdir("all_replays/new/")[:400]]
    ),
    batch_size=1,
    num_workers=12,
    worker_init_fn=per_worker_init_fn,
    collate_fn=collate_fn,
)

network = Network()
c = 10
for batch in dataloader:
    observations, actions = batch
    c = c - 1
    if c == 0:
        break
    print(observations.shape)
    print(actions.shape)


# dummy_in = (55, 24, 24)
# summary(network, dummy_in)
