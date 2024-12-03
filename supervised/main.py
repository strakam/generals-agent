from petting_replays import PettingZooGenerals
from replay_agent import ReplayAgent
import torch
import math
import os


class ReplayDataset(torch.utils.data.IterableDataset):
    def __init__(self, replay_files):
        self.replay_files = replay_files

        self.agent_A = ReplayAgent(id="A", color="red")
        self.agent_B = ReplayAgent(id="B", color="blue")

        self.env = None

    def __iter__(self):
        obs, info = self.env.reset()
        self.agent_A.give_actions(info[0])
        self.agent_B.give_actions(info[1])
        while True:
            a, b = self.agent_A.id, self.agent_B.id
            actions = {a: self.agent_A.act(obs[a]), b: self.agent_B.act(obs[b])}

            yield (1,1)
            obs, _, terminated, truncated, _ = self.env.step(actions)

            self.env.render()
            if all(terminated.values()) or all(truncated.values()):
                obs, info = self.env.reset()
                self.agent_A.give_actions(info[0])
                self.agent_B.give_actions(info[1])


def per_worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset  # the dataset copy in this worker process

    length = len(dataset.replay_files)
    per_worker = int(math.ceil((length / float(worker_info.num_workers))))

    worker_id = worker_info.id
    start = worker_id * per_worker
    end = min(start + per_worker, length)

    dataset.env = PettingZooGenerals(
        agents={
            dataset.agent_A.id: dataset.agent_A,
            dataset.agent_B.id: dataset.agent_B,
        },
        replay_files=[
            name for name in dataset.replay_files[start : end]
        ],
        render_mode=None
    )
#
dataloader = torch.utils.data.DataLoader( ReplayDataset(
        [f"all_replays/new/{name}" for name in os.listdir("all_replays/new/")[:400]]
    ),
    batch_size=20,
    num_workers=1,
    worker_init_fn=per_worker_init_fn,
    collate_fn=lambda x: x,
)
c = 0
for o in dataloader:
    c += 1

# dataloader = torch.utils.data.DataLoader(
#     ReplayDataset(
#         [f"all_replays/new/{name}" for name in os.listdir("all_replays/new/")[:4]]
#     ),
#     batch_size=2,
#     num_workers=2,
#     worker_init_fn=per_worker_init_fn,
# )
# k = list(dataloader)
# print(k)
#
# a1 = ReplayAgent(id="A", color="red")
# a2 = ReplayAgent(id="B", color="blue")
# # Load datasets
# # "all_replays/new/" and "all_replays/old/" are the directories containing the replays
# samples = 20
# print(os.listdir("all_replays/new/")[:samples])
# dataloader = torch.utils.data.DataLoader(
#     ReplayDataset(
#         [f"all_replays/new/{name}" for name in os.listdir("all_replays/new/")[:samples]]
#     ),
#     batch_size=5,
#     num_workers=2,
#     worker_init_fn=per_worker_init_fn,
# )
# pprint.pprint(list(dataloader))
# exit()
#
#
# def memory_usage():
#     process = psutil.Process(os.getpid())
#     return process.memory_info().rss / (1024**2)  # Convert bytes to MB
#
#
# sample = 20
# new_names = os.listdir("all_replays/new/")[:sample]
# old_names = os.listdir("all_replays/old/")[:sample]
#
# data_loader = torch.utils.data.DataLoader(
#     ReplayDataset([f"all_replays/new/{name}" for name in new_names]),
#     batch_size=5,
#     num_workers=2,
#     worker_init_fn=per_worker_init_fn,
# )
