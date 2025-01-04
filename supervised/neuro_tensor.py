import torch
import numpy as np
from generals.core.game import Action
import lightning as L
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
        batch_size: int | None = 1,
    ):
        super().__init__(id, color)
        self.network = network

        self.replay_moves = replay_moves
        self.general_position = general_position
        self.history_size = history_size
        self.batch_size = batch_size

        self.reset()

    def reset(self):
        self.army_stack = torch.zeros((self.batch_size, self.history_size, 24, 24))
        self.enemy_stack = torch.zeros((self.batch_size, self.history_size, 24, 24))
        self.last_army = torch.zeros((self.batch_size, 24, 24))
        self.last_enemy_army = torch.zeros((self.batch_size, 24, 24))
        self.cities = torch.zeros((self.batch_size, 24, 24)).bool()
        self.generals = torch.zeros((self.batch_size, 24, 24)).bool()
        self.mountains = torch.zeros((self.batch_size, 24, 24)).bool()
        self.seen = torch.zeros((self.batch_size, 24, 24)).bool()
        self.enemy_seen = torch.zeros((self.batch_size, 24, 24)).bool()

        self.last_observation = torch.zeros((self.batch_size, 31, 24, 24))

    def augment_observation(self, obs: np.ndarray) -> torch.Tensor:
        obs = torch.from_numpy(obs)
        armies = 0
        generals = 1
        cities = 2
        mountains = 3
        neutral_cells = 4
        owned_cells = 5
        opponent_cells = 6
        fog_cells = 7
        structures_in_fog = 8
        owned_land_count = 9
        owned_army_count = 10
        opponent_land_count = 11
        opponent_army_count = 12
        timestep = 13
        priority = 14

        self.army_stack[:, 1:, :, :] = self.army_stack[:, :-1, :, :]
        self.army_stack[:, 0, :, :] = (
            obs[:, armies, :, :] * obs[:, owned_cells, :, :] - self.last_army
        )

        self.enemy_stack[:, 1:, :, :] = self.enemy_stack[:, :-1, :, :]
        self.enemy_stack[:, 0, :, :] = (
            obs[:, armies, :, :] * obs[:, opponent_cells, :, :] - self.last_enemy_army
        )

        self.last_army = obs[:, armies, :, :] * obs[:, owned_cells, :, :]
        self.last_enemy_army = obs[:, armies, :, :] * obs[:, opponent_cells, :, :]

        # maximum filter as F.max_pool2d with kernel_size=3 and stride=1, padding=1
        self.seen = torch.logical_or(
            self.seen,
            torch.nn.functional.max_pool2d(obs[:, owned_cells, :, :], 3, 1, 1).bool(),
        )

        self.enemy_seen = torch.logical_or(
            self.enemy_seen,
            torch.nn.functional.max_pool2d(
                obs[:, opponent_cells, :, :], 3, 1, 1
            ).bool(),
        )

        self.cities |= obs[:, cities, :, :].bool()
        self.generals |= obs[:, generals, :, :].bool()
        self.mountains |= obs[:, mountains, :, :].bool()

        ones = torch.ones((self.batch_size, 24, 24))
        channels = torch.stack(
            [
                obs[:, armies, :, :],
                obs[:, armies, :, :] * obs[:, owned_cells, :, :],
                obs[:, armies, :, :] * obs[:, opponent_cells, :, :],
                obs[:, armies, :, :] * obs[:, neutral_cells, :, :],
                self.seen,
                self.enemy_seen,  # enemy sight
                self.generals,
                self.cities,
                self.mountains,
                obs[:, neutral_cells, :, :],
                obs[:, owned_cells, :, :],
                obs[:, opponent_cells, :, :],
                obs[:, fog_cells, :, :],
                obs[:, structures_in_fog, :, :],
                obs[:, timestep, :, :] * ones,
                (obs[:, timestep, :, :] % 50) * ones / 50,
                obs[:, priority, :, :] * ones,
                obs[:, owned_land_count, :, :] * ones,
                obs[:, owned_army_count, :, :] * ones,
                obs[:, opponent_land_count, :, :] * ones,
                obs[:, opponent_army_count, :, :] * ones,
            ],
            dim=1,
        )
        army_stacks = torch.cat([self.army_stack, self.enemy_stack], dim=1)
        augmented_obs = torch.cat([channels, army_stacks], dim=1)
        self.last_observation = augmented_obs
        return augmented_obs

    def act(self, obs: np.ndarray, mask: np.ndarray | None = None) -> Action:
        """
        Randomly selects a valid action.
        """
        self.augment_observation(obs)  # obs will be converted to tensor here
        mask = torch.from_numpy(mask).float()

        assert len(obs.shape) == len(mask.shape)
        if len(obs.shape) == 3:
            obs = obs.unsqueeze(0)
            mask = mask.unsqueeze(0)

        with torch.no_grad():
            square, direction = self.network(self.last_observation, mask)

        # use termperature to control randomness
        square = torch.argmax(square, dim=1)
        direction = torch.argmax(direction, dim=1)
        row = square // 24
        col = square % 24
        zeros = torch.zeros(self.batch_size)
        return torch.stack([zeros, row, col, direction, zeros], dim=1).numpy()
