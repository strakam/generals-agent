import torch
import numpy as np
from generals.core.action import Action, compute_valid_move_mask
import lightning as L
from generals.agents import Agent
from generals.core.observation import Observation
from torch.nn.functional import max_pool2d as max_pool2d
from functools import wraps


def conditional_compile(func):
    """
    Decorator that applies torch.compile only for NeuroAgent and OnlineAgent,
    but not for SupervisedAgent
    """

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if isinstance(self, (NeuroAgent, OnlineAgent)) and not isinstance(
            self, SupervisedAgent
        ):
            return torch.compile(func)(self, *args, **kwargs)
        return func(self, *args, **kwargs)

    return wrapper


class NeuroAgent(Agent):
    def __init__(
        self,
        network: L.LightningModule | None = None,
        id: str = "Neuro",
        history_size: int | None = 5,
        batch_size: int | None = 1,
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
    ):
        super().__init__(id)
        self.network = network
        self.history_size = history_size
        self.batch_size = batch_size
        self.n_channels = 22 + 2 * history_size
        self.device = device

        if self.network is not None:
            self.network.to(device)

        self.reset()

    @conditional_compile
    def reset(self):
        """
        Reset the agent's internal state.
        The state contains things that the agent remembers over time (positions of generals, etc.).
        """
        shape = (self.batch_size, 24, 24)
        history_shape = (self.batch_size, self.history_size, 24, 24)

        device = self.device
        self.army_stack = torch.zeros(history_shape, device=device)
        self.enemy_stack = torch.zeros(history_shape, device=device)
        self.last_army = torch.zeros(shape, device=device)
        self.last_enemy_army = torch.zeros(shape, device=device)
        self.cities = torch.zeros(shape, device=device).bool()
        self.generals = torch.zeros(shape, device=device).bool()
        self.mountains = torch.zeros(shape, device=device).bool()
        self.seen = torch.zeros(shape, device=device).bool()
        self.i_know_enemy_seen = torch.zeros(shape, device=device).bool()
        self.i_know_enemy_owns = torch.zeros(shape, device=device).bool()
        self.last_observation = torch.zeros(
            (self.batch_size, self.n_channels, 24, 24), device=device
        )

    @conditional_compile
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

        # note moves that enemy did based on changes in his army counts, same for us
        army_stack_clone = self.army_stack.clone().to(self.device)
        enemy_stack_clone = self.enemy_stack.clone().to(self.device)
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

        # what i've seen
        self.seen = torch.logical_or(
            self.seen,
            max_pool2d(obs[:, owned_cells, :, :], 3, 1, 1).bool(),
        )

        # what i know enemy has seen
        self.i_know_enemy_seen = torch.logical_or(
            self.i_know_enemy_seen,
            max_pool2d(obs[:, opponent_cells, :, :], 3, 1, 1).bool(),
        )

        # figure out what enemy owns
        opponent_owned_in_the_past = self.i_know_enemy_owns.clone()
        what_i_see_he_owns = obs[:, opponent_cells, :, :].bool()
        i_owned_before = self.last_observation[:, 1, :, :].bool()
        i_own_now = obs[:, owned_cells, :, :].bool()

        # opponent stole from me
        stolen_cells = torch.logical_and(i_owned_before, ~i_own_now)

        # what he owned + what i see he owns now
        enemy_owns = torch.logical_or(opponent_owned_in_the_past, what_i_see_he_owns)

        # plus what he stole from me (i dont see it anymore)
        enemy_owns = torch.logical_or(enemy_owns, stolen_cells)

        # minus what i took from him
        enemy_owns = torch.logical_and(enemy_owns, ~i_own_now)

        self.i_know_enemy_owns = enemy_owns

        # remember structures that we saw forever
        self.cities |= obs[:, cities, :, :].bool()
        self.generals |= obs[:, generals, :, :].bool()
        self.mountains |= obs[:, mountains, :, :].bool()

        ones = torch.ones((self.batch_size, 24, 24)).to(self.device)
        channels = torch.stack(
            [
                obs[:, armies, :, :],
                obs[:, armies, :, :] * obs[:, owned_cells, :, :],
                obs[:, armies, :, :] * obs[:, opponent_cells, :, :],
                obs[:, armies, :, :] * obs[:, neutral_cells, :, :],
                self.seen,
                self.i_know_enemy_seen,
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
                self.i_know_enemy_owns,
            ],
            dim=1,
        )
        army_stacks = torch.cat([self.army_stack, self.enemy_stack], dim=1)
        augmented_obs = torch.cat([channels, army_stacks], dim=1).float()
        self.last_observation = augmented_obs
        return augmented_obs

    @conditional_compile
    def act(self, obs: np.ndarray, mask: np.ndarray) -> Action:
        """
        Based on a new observation, augment the internal state and return an action.
        """
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).float().to(self.device)
        self.augment_observation(obs)
        mask = torch.from_numpy(mask).float().to(self.device)

        with torch.no_grad():
            square, direction = self.network(self.last_observation, mask)

        square = torch.argmax(square, dim=1)
        direction = torch.argmax(direction, dim=1)
        row = square // 24
        col = square % 24
        zeros = torch.zeros(self.batch_size).to(self.device)
        actions = torch.stack([zeros, row, col, direction, zeros], dim=1)
        # actions, where direction is 4, set the first value to 1
        actions[actions[:, 3] == 4, 0] = 1
        return actions.cpu().numpy().astype(int)


class SupervisedAgent(NeuroAgent):
    def __init__(
        self,
        network: L.LightningModule | None = None,
        id: str = "Neuro",
        history_size: int | None = 5,
        device: torch.device = "cpu",
    ):
        super().__init__(network, id, history_size, 1, device)

    def act(self, obs: Observation) -> Action:
        obs = torch.tensor(obs.as_tensor()).unsqueeze(0)
        self.augment_observation(obs)
        return [1, 0, 0, 0, 0]  # pass


class OnlineAgent(NeuroAgent):
    def __init__(
        self,
        network: L.LightningModule | None = None,
        id: str = "Neuro",
        history_size: int | None = 5,
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
    ):
        super().__init__(network, id, history_size, 1, device)
        self.network.eval()

    def act(self, obs: Observation) -> Action:
        """
        Based on a new observation, augment the internal state and return an action.
        """
        obs.pad_observation(24)
        mask = torch.from_numpy(compute_valid_move_mask(obs)).unsqueeze(0)
        obs = torch.tensor(obs.as_tensor()).unsqueeze(0)
        action = super().act(obs, mask)[0]  # Take the only action
        return action

    def precompile(self):
        # Run the agent once to precompile the code
        self.reset()
        self.augment_observation(torch.zeros((1, 15, 24, 24)))
        obs = Observation(
            armies=np.zeros((24, 24)),
            generals=np.zeros((24, 24)),
            cities=np.zeros((24, 24)),
            mountains=np.zeros((24, 24)),
            neutral_cells=np.zeros((24, 24)),
            owned_cells=np.zeros((24, 24)),
            opponent_cells=np.zeros((24, 24)),
            fog_cells=np.zeros((24, 24)),
            structures_in_fog=np.zeros((24, 24)),
            owned_land_count=0,
            owned_army_count=0,
            opponent_land_count=0,
            opponent_army_count=0,
            timestep=0,
            priority=0,
        )
        self.act(obs)
        print("Precompiled the agent")
