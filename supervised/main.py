import numpy as np
import os
import torch
import lightning as L
from dataloader import ReplayDataset, per_worker_init_fn
from network import Network
from torchsummary import summary
from generals.agents import RandomAgent
from replay_agent import ReplayAgent
from petting_replays import PettingZooGenerals


def collate_fn(batch):
    observations = torch.from_numpy(np.array([b[0] for b in batch])).float()
    actions = torch.from_numpy(np.array([b[1] for b in batch]))
    return observations, actions

replays = [f"all_replays/new/{name}" for name in os.listdir("all_replays/new/")[1:2]]

print(replays)
dataloader = torch.utils.data.DataLoader(
    ReplayDataset(
        [f"all_replays/new/{name}" for name in os.listdir("all_replays/new/")[1:2]]
    ),
    batch_size=64,
    num_workers=1,
    worker_init_fn=per_worker_init_fn,
    collate_fn=collate_fn,
)


network = Network(input_dims=(55, 24, 24), channel_sequence=[64, 32, 16])

trainer = L.Trainer(fast_dev_run = 800)
trainer.fit(network, train_dataloaders=dataloader)

random = RandomAgent(id="A")
replayer = ReplayAgent(id="B", color=(0, 0, 238))
replayer2 = ReplayAgent(id="C", color=(0, 0, 238))
env = PettingZooGenerals(
    agents={
        "A": random,
        "B": replayer
    },
    replay_files=[f"all_replays/new/{name}" for name in os.listdir("all_replays/new/")[1:2]],
    render_mode="human"
)

obs, info = env.reset()
replayer.give_actions(info[1])
replayer2.give_actions(info[0])
print(replayer2.replay_moves)
while True:
    network_obs = torch.from_numpy(obs['A']).unsqueeze(0).float()
    network_action = network(network_obs)
    v, s, d = network_action
    s = s.argmax(1) # convert index from one-hot vector of size 24*24 to two numbers
    i,j = s // 24, s % 24
    d = d.argmax(1)
    action = [0, i[0], j[0], d[0], 0]
    timestep = obs['A'][13][0][0]
    if timestep in replayer2.replay_moves:
        print(action)
        print(f'replayer {replayer2.replay_moves[timestep]}')
    actions = {
        "A": action,
        "B": replayer.act(obs['B'])
    }

    # yield (obs['B'], actions[b])
    obs, _, terminated, truncated, _ = env.step(actions)

    env.render()
