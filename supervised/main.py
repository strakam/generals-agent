import os
import torch
import lightning as L
from dataloader import ReplayDataset, per_worker_init_fn, collate_fn
from network import Network

from pytorch_lightning.loggers.neptune import NeptuneLogger
from lightning.pytorch.callbacks import ModelCheckpoint

# N_SAMPLES = 2 * 106752611
SEED = 7
N_SAMPLES = 2 * 60995855
BUFFER_SIZE = 8000
LEARNING_RATE = 3e-4
BATCH_SIZE = 128#1792
N_WORKERS = 4
LOG_EVERY_N_STEPS = 10
# EVAL_INTERVAL = 5000
EVAL_N_GAMES = 5
N_EPOCHS = 3
CLIP_VAL = 1.5
MAX_STEPS = N_SAMPLES // BATCH_SIZE * N_EPOCHS

torch.manual_seed(SEED)
torch.set_float32_matmul_precision("high")

key_file = open("neptune_token.txt", "r")
key = key_file.read()
neptune_logger = NeptuneLogger(
    api_key=key,
    project="strakam/supervised-agent",
)


# replays = [f"all_replays/old/{name}" for name in os.listdir("all_replays/old/")]
replays = [f"all_replays/new/{name}" for name in os.listdir("all_replays/new/")]
# replays = ["all_replays/new/gHtmCKd1F"]

torch.randperm(len(replays))


dataset = ReplayDataset(replays, buffer_size=BUFFER_SIZE)
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    num_workers=N_WORKERS,
    worker_init_fn=per_worker_init_fn,
    collate_fn=collate_fn,
)

one, two = 0, 0
weight_one, weight_two = 0, 0
c = 0
for b in dataloader:
    obs, mask, values, actions, weights, replay_ids = b
    # count number of samples with weight == 1.0 and with weight 0.02
    one += torch.sum(weights == 1.0)
    two += torch.sum(weights == 0.02)
    weight_one += torch.sum(weights[weights == 1.0])
    weight_two += torch.sum(weights[weights == 0.02])
    c+=1
    if c > 2000:
        break
print(one, two)
print(weight_one, weight_two)
print(one/(one+two))
print(weight_one/(weight_one+weight_two))
exit()

    


model = Network(
    lr=LEARNING_RATE, n_steps=MAX_STEPS, input_dims=(29, 24, 24), compile=True
)

checkpoint_callback = ModelCheckpoint(
    dirpath="/storage/praha1/home/strakam3/checkpoints",
    save_top_k=-1,
    every_n_train_steps=1000,
)

# eval_callback = EvalCallback(
#     network=model,
#     eval_interval=EVAL_INTERVAL,
#     n_eval_games=EVAL_N_GAMES,
# )

trainer = L.Trainer(
    # logger=neptune_logger,
    log_every_n_steps=LOG_EVERY_N_STEPS,
    max_steps=MAX_STEPS,
    max_epochs=-1,
    gradient_clip_val=CLIP_VAL,
    gradient_clip_algorithm="norm",
    callbacks=[checkpoint_callback],
)
trainer.fit(model, train_dataloaders=dataloader)

# save model
torch.save(model.state_dict(), "final_network.pt")
