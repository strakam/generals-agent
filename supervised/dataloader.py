from generals.envs import PettingZooGenerals
from generals.core.action import compute_valid_move_mask
from agent import SupervisedAgent
import json
import torch
import math
import numpy as np


class ReplayDataset(torch.utils.data.IterableDataset):
    def __init__(self, replay_files: list[str], buffer_size: int = 1000):
        self.replay_files = replay_files
        self.replays = None
        self.env = None
        self.A, self.B = SupervisedAgent(id="red"), SupervisedAgent(id="blue")

        self.buffer = [
            [
                np.empty((self.A.n_channels, 24, 24), dtype=np.float32),  # Obs
                np.empty((24, 24, 4), dtype=np.float32),  # Mask
                0.0,  # Value
                np.empty(5, dtype=np.int32),  # Action
            ]
            for _ in range(buffer_size)
        ]
        self.current_replay = None
        self.buffer_idx, self.replay_idx = 0, 0
        self.filled = False

    def check_validity(self, obs: np.ndarray, mask: np.ndarray, action: np.ndarray, stars: int) -> bool:
        """
        Check if an action is valid and from a high-rated player.
        """
        _, i, j, direction, _ = action
        army_size = obs[0][i][j]

        # Check if passing or making valid move with sufficient army
        is_passing = direction == 4
        is_valid_move = direction < 4 and mask[i][j][direction] == 1 and army_size > 1

        # Only use moves from high-rated players
        is_high_rated = stars >= 70

        return is_high_rated and (is_passing or is_valid_move)

    def save_sample(self, obs, mask, value, action, timestep, game_length):
        # Calculate discounted value based on game progress
        progress = timestep / game_length
        discounted_value = value * progress  # Linear discount from 0 to final value

        self.buffer[self.buffer_idx][0][:] = obs
        self.buffer[self.buffer_idx][1][:] = mask
        self.buffer[self.buffer_idx][2] = discounted_value  # Use discounted value
        self.buffer[self.buffer_idx][3][:] = action
        self.buffer_idx += 1
        if self.buffer_idx == len(self.buffer):
            self.filled = True
            self.buffer_idx = 0
            np.random.shuffle(self.buffer)

    def teacher_action(self, moves, base, timestep):
        # throw indicates whether action should be used in training
        if timestep in moves:
            return moves[timestep], False
        return [1, base[0], base[1], 4, 0], True

    def __iter__(self):
        a, b = self.A.id, self.B.id
        map, moves, values, bases, length, stars = self.get_new_replay()

        obs, _ = self.env.reset(options={"grid": map})
        while True:
            if self.filled:
                while self.buffer_idx < len(self.buffer):
                    yield self.buffer[self.buffer_idx]
                    self.buffer_idx += 1
                self.buffer_idx = 0
                self.filled = False

            timestep = obs[a]["timestep"]

            # Process both agents
            agents = [(self.A, a, 0), (self.B, b, 1)]
            actions = {}
            for agent, agent_id, idx in agents:
                # Get teacher action
                action, throw = self.teacher_action(moves[idx], bases[idx], timestep)
                actions[agent_id] = action

                # Process observation
                _ = agent.act(obs[agent_id])
                agent_obs = agent.last_observation[0]
                agent_mask = compute_valid_move_mask(obs[agent_id])

                # Save valid samples
                if self.check_validity(agent_obs, agent_mask, action, stars[idx]) and timestep > 21 and not throw:
                    self.save_sample(agent_obs, agent_mask, values[idx], action, timestep, length)

            obs, _, terminated, _, _ = self.env.step(actions)

            if terminated or timestep >= length:
                map, moves, values, bases, length, stars = self.get_new_replay()
                obs, _ = self.env.reset(options={"grid": map})
                self.A.reset()
                self.B.reset()

    def get_new_replay(self):
        game = json.load(open(self.replays[self.replay_idx], "r"))
        self.current_replay = self.replays[self.replay_idx]
        width = game["mapWidth"]
        height = game["mapHeight"]

        player_moves = [{}, {}]
        player_values = [-1.0, -1.0]
        player_stars = game["stars"]

        # Determine winner
        if len(game["afks"]) == 0:
            winner = game["moves"][-1][0]
        else:
            winner = 1 - game["afks"][0][0]
        player_values[winner] = 1.0

        for move in game["moves"]:
            index, i, j, is50, turn = move[0], move[1], move[2], move[3], move[4]
            i, j = divmod(move[1], width)
            if move[2] == move[1] + 1:
                direction = 3
            elif move[2] == move[1] - 1:
                direction = 2
            elif move[2] == move[1] + game["mapWidth"]:
                direction = 1
            elif move[2] == move[1] - game["mapWidth"]:
                direction = 0
            else:
                continue  # very few replays have moves that dont do anything
            player_moves[index][turn] = [0, i, j, direction, is50]

        # calculate game length as time of the last move
        game_length = game["moves"][-1][4]
        map = ["." for _ in range(width * height)]

        # place cities
        for pos, value in zip(game["cities"], game["cityArmies"]):
            map[pos] = str(value - 40) if value != 50 else "x"

        # place mountains
        for pos in game["mountains"]:
            map[pos] = "#"

        # place generals
        generals = game["generals"]
        map[generals[0]] = "A"
        map[generals[1]] = "B"
        player_bases = [
            divmod(generals[0], game["mapWidth"]),
            divmod(generals[1], game["mapWidth"]),
        ]

        # convert to 2D array
        map = [map[i : i + game["mapWidth"]] for i in range(0, len(map), game["mapWidth"])]

        # Pad the game with '#' to make it 24x24
        pad_width = 24 - width
        pad_height = 24 - height
        map = [[*row, *["#" for _ in range(pad_width)]] for row in map]
        map.extend([["#" for _ in range(24)] for _ in range(pad_height)])

        map_str = "\n".join(["".join(row) for row in map])
        self.replay_idx += 1
        self.replay_idx %= len(self.replays)
        return (
            map_str,
            player_moves,
            player_values,
            player_bases,
            game_length,
            player_stars,
        )


def per_worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset  # the dataset copy in this worker process

    length = len(dataset.replay_files)
    per_worker = int(math.ceil((length / float(worker_info.num_workers))))

    worker_id = worker_info.id
    start = worker_id * per_worker
    end = min(start + per_worker, length)

    dataset.replays = dataset.replay_files[start:end]
    print(f"Worker {worker_id} handling replays {start} to {end}")
    print(f"Worker {worker_id} {dataset.replays[:3]}")

    dataset.env = PettingZooGenerals(agents=["red", "blue"])


def collate_fn(batch):
    observations = torch.tensor(np.array([b[0] for b in batch]), dtype=torch.float32)
    masks = torch.tensor(np.array([b[1] for b in batch]), dtype=torch.float32)
    values = torch.tensor(np.array([b[2] for b in batch]), dtype=torch.float32)
    actions = torch.tensor(np.array([b[3] for b in batch]), dtype=torch.int64)
    return observations, masks, values, actions
