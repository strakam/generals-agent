import os
import torch
import lightning as L
from dataloader import ReplayDataset, per_worker_init_fn, collate_fn
from network import Network

from pytorch_lightning.loggers.neptune import NeptuneLogger
from lightning.pytorch.callbacks import ModelCheckpoint

SEED = 43
DATASET = "good_pov"
# N_SAMPLES = 2 * 60995855 # For all new replays
# N_SAMPLES = 53_000_000  # Above 50
N_SAMPLE = 9_000_000  # Above 70
N_SAMPLES = 16_000_000  # For high elo povs

BUFFER_SIZE = 18000
LEARNING_RATE = 2e-4
BATCH_SIZE = 1792
N_WORKERS =  42
LOG_EVERY_N_STEPS = 20
EVAL_N_GAMES = 5
N_EPOCHS = 4
CLIP_VAL = 2.0
MAX_STEPS = N_SAMPLES // BATCH_SIZE * N_EPOCHS
STORAGE = "/storage/praha1/home/strakam3/checkpoints"

torch.manual_seed(SEED)
torch.set_float32_matmul_precision("high")

key_file = open("neptune_token.txt", "r")
key = key_file.read()
neptune_logger = NeptuneLogger(
    api_key=key,
    project="strakam/supervised-agent",
)


path = f"datasets/{DATASET}"
replays = [f"{path}/{name}" for name in os.listdir(path)]
torch.randperm(len(replays))


dataset = ReplayDataset(replays, buffer_size=BUFFER_SIZE)
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    num_workers=N_WORKERS,
    worker_init_fn=per_worker_init_fn,
    collate_fn=collate_fn,
)

# channel_sequence = [320, 384, 448, 448]
model = Network(lr=LEARNING_RATE, n_steps=MAX_STEPS, compile=True)

checkpoint_callback = ModelCheckpoint(
    dirpath=STORAGE,
    save_top_k=-1,
    every_n_train_steps=4000,
)

trainer = L.Trainer(
    logger=neptune_logger,
    log_every_n_steps=LOG_EVERY_N_STEPS,
    max_steps=MAX_STEPS,
    max_epochs=-1,
    gradient_clip_val=CLIP_VAL,
    gradient_clip_algorithm="norm",
    callbacks=[checkpoint_callback],
)

trainer.fit(model, train_dataloaders=dataloader)
