from petting_replays import PettingZooGenerals
from replay_agent import ReplayAgent
import torch
import math


class ReplayDataset(torch.utils.data.IterableDataset):
    def __init__(self, replay_files):
        self.replay_files = replay_files

        self.A = ReplayAgent(id="A", color="red")
        self.B = ReplayAgent(id="B", color="blue")

        self.env = None

    def __iter__(self):
        obs, info = self.env.reset()
        self.A.give_actions(info[0])
        self.B.give_actions(info[1])
        yield obs['A']
        yield obs['B']
        while True:
            a, b = self.A.id, self.B.id
            actions = {a: self.A.act(obs[a]), b: self.B.act(obs[b])}

            yield obs['A']
            yield obs['B']
            obs, _, terminated, truncated, _ = self.env.step(actions)

            self.env.render()
            if all(terminated.values()) or all(truncated.values()):
                obs, info = self.env.reset()
                self.A.give_actions(info[0])
                self.B.give_actions(info[1])


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
            dataset.A.id: dataset.A,
            dataset.B.id: dataset.B,
        },
        replay_files=[name for name in dataset.replay_files[start:end]],
        render_mode=None,
    )
