import torch
import numpy as np
from generals.core.game import Action
import lightning as L
from generals.core.action import compute_valid_move_mask
from scipy.ndimage import maximum_filter
from generals.core.observation import Observation
from generals.agents import Agent


class NeuroAgent(Agent):
    def __init__(
        self,
        network: L.LightningModule | None = None,
        id: str = "Neuro",
        color: tuple[int, int, int] = (242, 61, 106),
        replay_moves: dict[int, list[int]] | None = None,
        general_position: tuple[int, int] | None = None,
        history_size: int | None = 5,
    ):
        super().__init__(id, color)
        self.network = network

        self.replay_moves = replay_moves
        self.general_position = general_position
        self.history_size = history_size

        self.reset()

    def reset(self):
        self.army_stack = np.zeros((self.history_size, 24, 24))
        self.enemy_stack = np.zeros((self.history_size, 24, 24))
        self.last_army = np.zeros((24, 24))
        self.last_enemy_army = np.zeros((24, 24))
        self.cities = np.zeros((24, 24)).astype(bool)
        self.generals = np.zeros((24, 24)).astype(bool)
        self.mountains = np.zeros((24, 24)).astype(bool)
        self.seen = np.zeros((24, 24)).astype(bool)
        self.enemy_seen = np.zeros((24, 24)).astype(bool)

        self.last_observation = np.zeros((31, 24, 24))

    def augment_observation(self, obs: Observation) -> np.ndarray:
        _obs = {}
        for k, channel in obs.items():
            if type(channel) is np.ndarray:
                pad_value = int(k == "mountains")
                pad_h = (0, 24 - channel.shape[0])
                pad_w = (0, 24 - channel.shape[1])
                _obs[k] = np.pad(channel, (pad_h, pad_w), constant_values=pad_value)
            else:
                _obs[k] = np.full((24, 24), channel)
        obs = _obs

        self.army_stack[1:, :, :] = self.army_stack[:-1, :, :]
        self.army_stack[0, :, :] = (obs["armies"] * obs["owned_cells"]) - self.last_army

        self.enemy_stack[1:, :, :] = self.enemy_stack[:-1, :, :]
        self.enemy_stack[0, :, :] = (
            obs["armies"] * obs["opponent_cells"]
        ) - self.last_enemy_army

        self.last_army = obs["armies"] * obs["owned_cells"]
        self.last_enemy_army = obs["armies"] * obs["opponent_cells"]

        self.seen = np.logical_or(
            self.seen,
            maximum_filter(obs["owned_cells"], size=3).astype(bool),
        )

        self.enemy_seen = np.logical_or(
            self.enemy_seen,
            maximum_filter(obs["opponent_cells"], size=3).astype(bool),
        )

        self.cities |= obs["cities"]
        self.generals |= obs["generals"]
        self.mountains |= obs["mountains"]

        return np.stack(
            [
                obs["armies"],
                obs["armies"] * obs["owned_cells"],
                obs["armies"] * obs["opponent_cells"],
                obs["armies"] * obs["neutral_cells"],
                self.seen,
                self.enemy_seen,  # enemy sight
                self.generals,
                self.cities,
                self.mountains,
                obs["neutral_cells"],
                obs["owned_cells"],
                obs["opponent_cells"],
                obs["fog_cells"],
                obs["structures_in_fog"],
                obs["timestep"] * np.ones((24, 24)),
                (obs["timestep"] % 50) * np.ones((24, 24)) / 50,
                obs["priority"] * np.ones((24, 24)),
                obs["owned_land_count"] * np.ones((24, 24)),
                obs["owned_army_count"] * np.ones((24, 24)),
                obs["opponent_land_count"] * np.ones((24, 24)),
                obs["opponent_army_count"] * np.ones((24, 24)),
                *self.army_stack,
                *self.enemy_stack,
            ]
        )

    def act(self, observation: Observation) -> Action:
        """
        Randomly selects a valid action.
        """
        self.last_observation = self.augment_observation(observation)
        mask = compute_valid_move_mask(observation)
        mask = np.expand_dims(mask, axis=0)
        obs = np.expand_dims(self.last_observation, axis=0)

        mask = torch.from_numpy(mask).float()
        obs = torch.from_numpy(obs).float()

        with torch.no_grad():
            s, d = self.network(obs, mask)
        s = s[0].detach().numpy()
        d = d[0].detach().numpy()

        s = np.argmax(s)
        i, j = divmod(s, 24)
        d = np.argmax(d)
        if d == 4:
            return [1, 0, 0, 0, 0]
        return [0, i, j, d, 0]
