from petting_replays import PettingZooGenerals
from replay_agent import ReplayAgent
import torch
import math
import numpy as np


class ReplayDataset(torch.utils.data.IterableDataset):
    def __init__(self, replay_files):
        self.replay_files = replay_files

        self.A = ReplayAgent(id="A", color="red")
        self.B = ReplayAgent(id="B", color="blue")
        self.env = None

    def reset_players(self):
        obs, moves, bases, values, replay = self.env.reset()
        self.A.replay_moves = moves[0]
        self.B.replay_moves = moves[1]
        self.A.general_position = bases[0]
        self.B.general_position = bases[1]
        self.A.value = values[0]
        self.B.value = values[1]
        self.A.replay = replay
        self.B.replay = replay

        return obs

    def check_validity(self, obs, action):
        _, i, j, d, _ = action
        mask = obs[1]
        army = obs[0][i][j]
        own = obs[5][i][j]
        if army > 1 and own == 1 and mask[i][j][d] == 1:
            return True
        return False


    def __iter__(self):
        obs = self.reset_players()
        while True:
            a, b = self.A.id, self.B.id
            actions = {a: self.A.act(obs[a]), b: self.B.act(obs[b])}
            if self.check_validity(obs[a], actions[a]):
                yield (obs[a], self.A.value, actions[a], self.A.replay)
            if self.check_validity(obs[b], actions[b]):
                yield (obs[b], self.B.value, actions[b], self.B.replay)
            obs, _, terminated, truncated, _ = self.env.step(actions)

            self.env.render()
            if all(terminated.values()) or all(truncated.values()):
                obs = self.reset_players()


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


def collate_fn(batch):
    observations = torch.from_numpy(np.array([b[0][0] for b in batch])).float()
    masks = torch.from_numpy(np.array([b[0][1] for b in batch])).float()
    values = torch.from_numpy(np.array([b[1] for b in batch])).float()
    actions = torch.from_numpy(np.array([b[2] for b in batch]))
    replay_ids = [b[3] for b in batch]
    return observations, masks, values, actions, replay_ids
