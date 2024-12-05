import os
import torch
from dataloader import ReplayDataset, per_worker_init_fn

dataloader = torch.utils.data.DataLoader(
    ReplayDataset(
        [f"all_replays/new/{name}" for name in os.listdir("all_replays/new/")[:400]]
    ),
    batch_size=40,
    num_workers=12,
    worker_init_fn=per_worker_init_fn,
    collate_fn=lambda x:x
)

c=0
for i in dataloader:
    c += 1
    print(c)
