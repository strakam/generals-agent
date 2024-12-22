import os
import torch
import lightning as L
from dataloader import ReplayDataset, per_worker_init_fn, collate_fn
from generals.agents import RandomAgent
from generals import PettingZooGenerals
from network import Network

# from pytorch_lightning.loggers.neptune import NeptuneLogger
from lightning.pytorch.callbacks import ModelCheckpoint

torch.manual_seed(0)
torch.set_float32_matmul_precision("high")

# key_file = open("neptune_token.txt", "r")
# key = key_file.read()
# neptune_logger = NeptuneLogger(
#     api_key=key,
#     project="strakam/supervised-agent",
# )


replays = [f"all_replays/new/{name}" for name in os.listdir("all_replays/new/")]

dataloader = torch.utils.data.DataLoader(
    ReplayDataset(replays),
    batch_size=256,
    num_workers=12,
    worker_init_fn=per_worker_init_fn,
    collate_fn=collate_fn,
)

model = Network(input_dims=(55, 24, 24), compile=False)

checkpoint_callback = ModelCheckpoint(
    dirpath="checkpoints",
    save_top_k=-1,
    every_n_train_steps=2000,
)

trainer = L.Trainer(
    # logger=neptune_logger,
    log_every_n_steps=10,
    max_epochs=-1,
    gradient_clip_val=5.0,
    gradient_clip_algorithm="norm",
    callbacks=[checkpoint_callback],
)
# trainer = L.Trainer()
trainer.fit(model, train_dataloaders=dataloader)

# save model
torch.save(model.state_dict(), "network.pt")
print("yay")
exit()

network = RandomAgent(id="A")
random = RandomAgent(id="B")
env = PettingZooGenerals(
    agents={"A": network, "B": random},
    render_mode="human",
)


obs, info = env.reset()
while True:
    print(obs["A"])
    network_obs = torch.from_numpy(obs["A"][0]).unsqueeze(0).float()
    network_mask = torch.from_numpy(obs["A"][1]).unsqueeze(0).float()
    network_action = network(network_obs, network_mask)
    v, s, d = network_action
    s = s.argmax(1)  # convert index from one-hot vector of size 24*24 to two numbers
    i, j = s // 24, s % 24
    # d = d.argmax(1)
    d = d.argmax(1)
    actions = {"A": [1, 0, 0, 0, 0], "B": random.act(obs["B"])}
    # yield (obs['B'], actions[b])
    obs, _, terminated, truncated, _ = env.step(actions)

    env.render()
