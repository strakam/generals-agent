import torch
import numpy as np
from generals.core.action import Action, compute_valid_move_mask
import lightning as L
from generals.agents import Agent
from generals.core.observation import Observation
from torch.nn.functional import max_pool2d as max_pool2d


class NeuroAgent(Agent):
    def __init__(
        self,
        network: L.LightningModule | None = None,
        id: str = "Neuro",
        history_size: int | None = 5,
        batch_size: int | None = 1,
    ):
        super().__init__(id)
        self.network = network
        self.history_size = history_size
        self.batch_size = batch_size

        self.reset()

    @torch.compile
    def reset(self):
        """
        Reset the agent's internal state.
        The state contains things that the agent remembers over time (positions of generals, etc.).
        """
        shape = (self.batch_size, 24, 24)
        n_channels = 21 + 2 * self.history_size
        self.army_stack = torch.zeros((self.batch_size, self.history_size, 24, 24))
        self.enemy_stack = torch.zeros((self.batch_size, self.history_size, 24, 24))
        self.last_army = torch.zeros(shape)
        self.last_enemy_army = torch.zeros(shape)
        self.cities = torch.zeros(shape).bool()
        self.generals = torch.zeros(shape).bool()
        self.mountains = torch.zeros(shape).bool()
        self.seen = torch.zeros(shape).bool()
        self.enemy_seen = torch.zeros(shape).bool()
        self.last_observation = torch.zeros((self.batch_size, n_channels, 24, 24))

    @torch.compile
    def augment_observation(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Here the agent augments what it knows about the game with the new observation.
        This is then further used to make a decision.
        """
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

        army_stack_clone = self.army_stack.clone()  # can't do inplace otherwise
        enemy_stack_clone = self.enemy_stack.clone()
        self.army_stack[:, 1:, :, :] = army_stack_clone[:, :-1, :, :]
        self.army_stack[:, 0, :, :] = (
            obs[:, armies, :, :] * obs[:, owned_cells, :, :] - self.last_army
        )

        self.enemy_stack[:, 1:, :, :] = enemy_stack_clone[:, :-1, :, :]
        self.enemy_stack[:, 0, :, :] = (
            obs[:, armies, :, :] * obs[:, opponent_cells, :, :] - self.last_enemy_army
        )

        self.last_army = obs[:, armies, :, :] * obs[:, owned_cells, :, :]
        self.last_enemy_army = obs[:, armies, :, :] * obs[:, opponent_cells, :, :]

        self.seen = torch.logical_or(
            self.seen,
            max_pool2d(obs[:, owned_cells, :, :], 3, 1, 1).bool(),
        )

        self.enemy_seen = torch.logical_or(
            self.enemy_seen,
            max_pool2d(obs[:, opponent_cells, :, :], 3, 1, 1).bool(),
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
        augmented_obs = torch.cat([channels, army_stacks], dim=1).float()
        self.last_observation = augmented_obs
        return augmented_obs

    @torch.compile
    def act(self, obs: np.ndarray, mask: np.ndarray) -> Action:
        """
        Based on a new observation, augment the internal state and return an action.
        """
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).float()
        self.augment_observation(obs)
        mask = torch.from_numpy(mask).float()

        with torch.no_grad():
            square, direction = self.network(self.last_observation, mask)

        square = torch.argmax(square, dim=1)
        direction = torch.argmax(direction, dim=1)
        row = square // 24
        col = square % 24
        zeros = torch.zeros(self.batch_size)
        actions = torch.stack([zeros, row, col, direction, zeros], dim=1)
        return actions.numpy().astype(int)


class OnlineAgent(NeuroAgent):
    def __init__(
        self,
        network: L.LightningModule | None = None,
        id: str = "Neuro",
        history_size: int | None = 5,
        batch_size: int | None = 1,
    ):
        super().__init__(network, id, history_size, batch_size)
        self.batch_size = 1  # Online agent only supports batch size of 1

    def act(self, obs: Observation) -> Action:
        """
        Based on a new observation, augment the internal state and return an action.
        """
        mask = torch.from_numpy(compute_valid_move_mask(obs)).unsqueeze(0)
        obs = torch.tensor(obs.as_tensor(pad_to=24)).unsqueeze(0)
        action = super().act(obs, mask)[0]  # Take the only action
        if action[3] == 4:
            return [1, 0, 0, 0, 0]
        return action
