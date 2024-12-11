import numpy as np
import os
import torch
import lightning as L
from dataloader import ReplayDataset, per_worker_init_fn
from network import Network
from generals.agents import RandomAgent
from replay_agent import ReplayAgent
from petting_replays import PettingZooGenerals

torch.manual_seed(0)
a, b = 1, 2


def collate_fn(batch):
    observations = torch.from_numpy(np.array([b[0][0] for b in batch])).float()
    masks = torch.from_numpy(np.array([b[0][1] for b in batch])).float()
    actions = torch.from_numpy(np.array([b[1] for b in batch]))
    return observations, masks, actions


replays = [f"all_replays/new/{name}" for name in os.listdir("all_replays/new/")[a:b]]

print(replays)
dataloader = torch.utils.data.DataLoader(
    ReplayDataset(
        [f"all_replays/new/{name}" for name in os.listdir("all_replays/new/")[a:b]]
    ),
    batch_size=128,
    num_workers=1,
    worker_init_fn=per_worker_init_fn,
    collate_fn=collate_fn,
)

# t=0
# for b in dataloader:
#     obs, masks, actions = b
#     i, j, d = actions[0][1], actions[0][2], actions[0][3]
#     print(masks[0, i, j], t, actions[0])
#     t+=1
#     if t == 50:
#         break

network = Network(input_dims=(55, 24, 24))

trainer = L.Trainer(fast_dev_run=500)
trainer.fit(network, train_dataloaders=dataloader)

# save model
torch.save(network.state_dict(), "network.pt")
# #
# #
dataloader = torch.utils.data.DataLoader(
    ReplayDataset(
        [f"all_replays/new/{name}" for name in os.listdir("all_replays/new/")[a:b]]
    ),
    batch_size=1,
    num_workers=1,
    worker_init_fn=per_worker_init_fn,
    collate_fn=collate_fn,
)
#
#
# load model
network = Network(input_dims=(55, 24, 24))
network.load_state_dict(torch.load("network.pt"))

trues = []
for i, b in enumerate(dataloader):
    obs, masks, actions = b
    if i == 100:
        break
    _, s, d = network(obs, masks)
    s = s.argmax(1)
    i, j = s // 24, s % 24
    d = d.argmax(1)
    print(f"pred {i, j, d} --- real {actions[0]}", end="")
    if actions[0][1] == i and actions[0][2] == j and d == actions[0][3]:
        trues.append(1)
        print()
    else:
        trues.append(0)
        print(" oops")

print(f"Accuracy: {sum(trues)/len(trues)}")

random = RandomAgent(id="A")
replayer = ReplayAgent(id="B", color=(0, 0, 238))
replayer2 = ReplayAgent(id="C", color=(0, 0, 238))
env = PettingZooGenerals(
    agents={"A": random, "B": replayer},
    replay_files=[
        f"all_replays/new/{name}" for name in os.listdir("all_replays/new/")[a:b]
    ],
    render_mode="human",
)

obs, info, starts = env.reset()
replayer.replay_moves = info[1]
replayer.general_position = starts["B"]
replayer2.replay_moves = info[0]
print(replayer2.replay_moves)
t = 0
while True:
    network_obs = torch.from_numpy(obs["A"][0]).unsqueeze(0).float()
    network_mask = torch.from_numpy(obs["A"][1]).unsqueeze(0).float()
    network_action = network(network_obs, network_mask)
    v, s, d = network_action
    s = s.argmax(1)  # convert index from one-hot vector of size 24*24 to two numbers
    i, j = s // 24, s % 24
    # d = d.argmax(1)
    d = d.argmax(1)
    timestep = t
    real_action = (
        replayer2.replay_moves[timestep]
        if timestep in replayer2.replay_moves
        else [1, 0, 0, 0, 0]
    )
    dir = real_action[3]
    action = [0, int(i[0]), int(j[0]), d, 0]
    if d == 4:
        action = [1, 0, 0, 0, 0]
    if timestep in replayer2.replay_moves:
        print(f"{action} ------ {real_action} ---- {env.time}")
    actions = {"A": action, "B": replayer.act(obs["B"])}
    t += 1

    # yield (obs['B'], actions[b])
    obs, _, terminated, truncated, _ = env.step(actions)

    env.render()
