import os
import torch
from dataloader import ReplayDataset, per_worker_init_fn
from network import Pyramid
from torchviz import make_dot

dataloader = torch.utils.data.DataLoader(
    ReplayDataset(
        [f"all_replays/new/{name}" for name in os.listdir("all_replays/new/")[:400]]
    ),
    batch_size=40,
    num_workers=12,
    worker_init_fn=per_worker_init_fn,
    collate_fn=lambda x:x
)


network = Pyramid()
# print number of network parameters
print(sum(p.numel() for p in network.parameters()))
# print network
print(network)

dummy_in = torch.randn(1, 55, 24, 24)
out = network(dummy_in)
print(out.shape)
