import os
import torch
import lightning as L
from dataloader import ReplayDataset, per_worker_init_fn, collate_fn
from callbacks import EvalCallback
from network import Network

from pytorch_lightning.loggers.neptune import NeptuneLogger
from lightning.pytorch.callbacks import ModelCheckpoint

N_SAMPLES = 2*106752611
BUFFER_SIZE = 8000
LEARNING_RATE = 7e-4
N_CHANNELS = 32
BATCH_SIZE = 2048
N_WORKERS = 4
LOG_EVERY_N_STEPS = 10
EVAL_INTERVAL = 5000
EVAL_N_GAMES = 5
MAX_STEPS = N_SAMPLES // BATCH_SIZE

torch.manual_seed(0)
torch.set_float32_matmul_precision("high")

key_file = open("neptune_token.txt", "r")
key = key_file.read()
neptune_logger = NeptuneLogger(
    api_key=key,
    project="strakam/supervised-agent",
)


replays = [f"all_replays/old/{name}" for name in os.listdir("all_replays/old/")]
replays += [f"all_replays/new/{name}" for name in os.listdir("all_replays/new/")]

torch.randperm(len(replays))


dataset = ReplayDataset(replays, buffer_size=BUFFER_SIZE)
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    num_workers=N_WORKERS,
    worker_init_fn=per_worker_init_fn,
    collate_fn=collate_fn,
)

model = Network(lr=LEARNING_RATE, input_dims=(N_CHANNELS, 24, 24), compile=True)

checkpoint_callback = ModelCheckpoint(
    dirpath="checkpoints",
    save_top_k=-1,
    every_n_train_steps=2000,
)

eval_callback = EvalCallback(
    network=model,
    eval_interval=EVAL_INTERVAL,
    n_eval_games=EVAL_N_GAMES,
)

trainer = L.Trainer(
    # logger=neptune_logger,
    log_every_n_steps=LOG_EVERY_N_STEPS,
    max_steps=MAX_STEPS,
    max_epochs=-1,
    gradient_clip_val=5.0,
    gradient_clip_algorithm="norm",
    callbacks=[checkpoint_callback, eval_callback],
)
# trainer = L.Trainer()
trainer.fit(model, train_dataloaders=dataloader)

# save model
torch.save(model.state_dict(), "final_network.pt")
