import os
import torch
import lightning as L
from dataloader import ReplayDataset, per_worker_init_fn, collate_fn
from network import Network
from pytorch_lightning.loggers.neptune import NeptuneLogger

torch.manual_seed(0)

key_file = open("neptune_token.txt", "r")
key = key_file.read()
neptune_logger = NeptuneLogger(
    api_key=key,
    project="strakam/supervised-agent",
)


replays = [
    f"all_replays/new_value/{name}" for name in os.listdir("all_replays/new_value/")
]
replays += [
    f"all_replays/old_value/{name}" for name in os.listdir("all_replays/old_value/")
]

dataloader = torch.utils.data.DataLoader(
    ReplayDataset(replays),
    batch_size=768,
    num_workers=64,
    worker_init_fn=per_worker_init_fn,
    collate_fn=collate_fn,
)

model = Network(input_dims=(55, 24, 24))
# also try torch.compile(model, mode="reduce-overhead")
# also try compiled_model = torch.compile(model, options={"shape_padding": True})
# model = torch.compile(model)


trainer = L.Trainer(
    logger=neptune_logger,
    log_every_n_steps=20,
    gradient_clip_val=5.0,
    gradient_clip_algorithm="norm",
)
# trainer = L.Trainer()
trainer.fit(model, train_dataloaders=dataloader)

# save model
torch.save(model.state_dict(), "network.pt")

# random = RandomAgent(id="A")
# replayer = ReplayAgent(id="B", color=(0, 0, 238))
# replayer2 = ReplayAgent(id="C", color=(0, 0, 238))
# env = PettingZooGenerals(
#     agents={"A": random, "B": replayer},
#     replay_files=replays,
#     render_mode=None,
# )
#
# options = {"replay_file": "debug"}
# obs, info, starts = env.reset(options=options)
# replayer.replay_moves = info[1]
# replayer.general_position = starts["B"]
# replayer2.replay_moves = info[0]
# t = 0
# while True:
#     # network_obs = torch.from_numpy(obs["A"][0]).unsqueeze(0).float()
#     # network_mask = torch.from_numpy(obs["A"][1]).unsqueeze(0).float()
#     # network_action = network(network_obs, network_mask)
#     # v, s, d = network_action
#     # s = s.argmax(1)  # convert index from one-hot vector of size 24*24 to two numbers
#     # i, j = s // 24, s % 24
#     # # d = d.argmax(1)
#     # d = d.argmax(1)
#     timestep = t
#     real_action = (
#         replayer2.replay_moves[timestep]
#         if timestep in replayer2.replay_moves
#         else [1, 0, 0, 0, 0]
#     )
#     dir = real_action[3]
#     # action = [0, int(i[0]), int(j[0]), d, 0]
#     # if d == 4:
#     #     action = [1, 0, 0, 0, 0]
#     # if timestep in replayer2.replay_moves:
#     #     print(f"{action} ------ {real_action} ---- {env.time}")
#     # else:
#     #
#     actions = {"A": real_action, "B": replayer.act(obs["B"])}
#     t += 1
#
#     # yield (obs['B'], actions[b])
#     obs, _, terminated, truncated, _ = env.step(actions)
#
#     # env.render()
