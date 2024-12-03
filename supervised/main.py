import os
import torch
from dataloader import ReplayDataset, per_worker_init_fn

dataloader = torch.utils.data.DataLoader(
    ReplayDataset(
        [f"all_replays/new/{name}" for name in os.listdir("all_replays/new/")[:400]]
    ),
    batch_size=2,
    num_workers=1,
    worker_init_fn=per_worker_init_fn,
    collate_fn=lambda x:x
)

for i in dataloader:
    print(len(i))
    print((i[0]['A'].shape))
    break
