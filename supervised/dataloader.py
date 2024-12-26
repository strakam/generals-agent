from petting_replays import PettingZooGenerals
from generals.core.action import compute_valid_action_mask
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

        self.buffer = [
            [
                np.empty((29, 24, 24), dtype=np.float32),
                np.empty((24, 24, 4), dtype=np.float32),
                0.0,
                np.empty(5, dtype=np.int32),
            ]
            for _ in range(50)
        ]
        self.buffer_idx = 0
        self.filled = False

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

    def check_validity(self, mask, own, action):
        _, i, j, d, _ = action
        if d == 4 or mask[i][j][d] == 1:
            assert own[i][j] == 1, "Invalid action"
            return True
        return False

    def save_sample(self, obs, mask, value, action):
        self.buffer[self.buffer_idx][0][:] = obs
        self.buffer[self.buffer_idx][1][:] = mask
        self.buffer[self.buffer_idx][2] = value
        self.buffer[self.buffer_idx][3][:] = action
        self.buffer_idx += 1
        if self.buffer_idx == len(self.buffer):
            self.filled = True
            self.buffer_idx = 0
            # np.random.shuffle(self.buffer)

    def __iter__(self):
        obs = self.reset_players()
        while True:
            a, b = self.A.id, self.B.id
            actions = {a: self.A.act(obs[a]), b: self.B.act(obs[b])}
            obs_a = self.A.last_observation
            obs_b = self.B.last_observation
            mask_a = compute_valid_action_mask(obs[a])
            mask_b = compute_valid_action_mask(obs[b])
            if self.check_validity(mask_a, self.A.last_observation[9], actions[a]):
                yield obs_a, mask_a, self.A.value, actions[a]
                if self.filled:
                    yield self.buffer[self.buffer_idx]
                self.save_sample(obs_a, mask_a, self.A.value, actions[a])

            if self.check_validity(mask_b, self.B.last_observation[9], actions[b]):
                if self.filled:
                    yield self.buffer[self.buffer_idx]
                self.save_sample(obs_b, mask_b, self.B.value, actions[b])
            obs, _, terminated, truncated, _ = self.env.step(actions)
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
    observations = torch.tensor(np.array([b[0] for b in batch]), dtype=torch.float32)
    masks = torch.tensor(np.array([b[1] for b in batch]), dtype=torch.float32)
    values = torch.tensor(np.array([b[2] for b in batch]), dtype=torch.float32)
    actions = torch.tensor(np.array([b[3] for b in batch]), dtype=torch.int64)
    return observations, masks, values, actions
