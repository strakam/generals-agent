from typing import Optional, Tuple
import torch
import numpy as np
from generals.core.action import Action, compute_valid_move_mask
import lightning as L
from generals.agents import Agent
from generals.core.observation import Observation
from torch.nn.functional import max_pool2d
from functools import wraps
from supervised.network import Network

GRID_SIZE = 24
DEFAULT_HISTORY_SIZE = 5
DEFAULT_BATCH_SIZE = 1


def conditional_compile(func):
    """
    Decorator that applies torch.compile only for NeuroAgent and OnlineAgent,
    but not for SupervisedAgent
    """

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if isinstance(self, (NeuroAgent, OnlineAgent, SelfPlayAgent)) and not isinstance(self, SupervisedAgent):
            return torch.compile(func)(self, *args, **kwargs)
        return func(self, *args, **kwargs)

    return wrapper


class NeuroAgent(Agent):
    """
    Represents a neural network-based agent in the game.

    Attributes:
        network (Optional[L.LightningModule]): The neural network model.
        id (str): Identifier for the agent.
        history_size (Optional[int]): Number of past states to consider.
        batch_size (Optional[int]): Batch size for processing.
        device (torch.device): Device to run the computations on.
    """

    def __init__(
        self,
        network: Optional[L.LightningModule] = None,
        id: str = "Neuro",
        history_size: Optional[int] = DEFAULT_HISTORY_SIZE,
        batch_size: Optional[int] = DEFAULT_BATCH_SIZE,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        """
        Initializes the NeuroAgent with the specified parameters.

        Args:
            network (Optional[L.LightningModule]): The neural network model.
            id (str): Identifier for the agent.
            history_size (Optional[int]): Number of past states to consider.
            batch_size (Optional[int]): Batch size for processing.
            device (torch.device): Device to run the computations on.
        """
        super().__init__(id)
        self.network = network
        self.history_size = history_size
        self.batch_size = batch_size
        self.n_channels = 21 + 2 * self.history_size
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
        self.enemy_seen = torch.zeros(shape, device=device).bool()
        self.last_observation = torch.zeros((self.batch_size, self.n_channels, 24, 24), device=device)

    def reset_histories(self, obs: torch.Tensor):
        # When timestep of the observation is 0, we want to reset all data corresponding to given batch sample
        timestep_mask = obs[:, 13, 0, 0] == 0.0

        attributes_to_reset = [
            "army_stack",
            "enemy_stack",
            "last_army",
            "last_enemy_army",
            "seen",
            "enemy_seen",
            "cities",
            "generals",
            "mountains",
        ]

        for attr in attributes_to_reset:
            getattr(self, attr)[timestep_mask] = 0

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

        self.reset_histories(obs)

        # Calculate current army states
        current_army = obs[:, armies, :, :] * obs[:, owned_cells, :, :]
        current_enemy_army = obs[:, armies, :, :] * obs[:, opponent_cells, :, :]

        # Update history stacks by shifting and adding new differences
        self.army_stack = torch.roll(self.army_stack, shifts=1, dims=1)
        self.enemy_stack = torch.roll(self.enemy_stack, shifts=1, dims=1)

        self.army_stack[:, 0, :, :] = current_army - self.last_army
        self.enemy_stack[:, 0, :, :] = current_enemy_army - self.last_enemy_army

        # Store current states for next iteration
        self.last_army = current_army
        self.last_enemy_army = current_enemy_army

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

        ones = torch.ones((self.batch_size, 24, 24)).to(self.device)
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

    @conditional_compile
    def act(self, obs: np.ndarray, mask: np.ndarray) -> Tuple[Action, float]:
        """
        Based on a new observation, augment the internal state and return an action.
        """
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).float().to(self.device)

        obs = obs.float().to(self.device)
        self.augment_observation(obs)
        mask = torch.from_numpy(mask).float().to(self.device)

        with torch.no_grad():
            square, direction = self.network(self.last_observation, mask)

        square = torch.argmax(square, dim=1)
        direction = torch.argmax(direction, dim=1)
        row = square // GRID_SIZE
        col = square % GRID_SIZE
        zeros = torch.zeros(self.batch_size).to(self.device)
        actions = torch.stack([zeros, row, col, direction, zeros], dim=1)
        # actions, where direction is 4, set the first value to 1
        actions[actions[:, 3] == 4, 0] = 1
        return actions.cpu().numpy().astype(int)


class SelfPlayAgent(NeuroAgent):
    def __init__(
        self,
        network: L.LightningModule | None = None,
        id: str = "Neuro",
        history_size: int | None = DEFAULT_HISTORY_SIZE,
        batch_size: int = DEFAULT_BATCH_SIZE,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        super().__init__(network, id, history_size, batch_size, device)

    @conditional_compile
    def act(self, obs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            square_logits, direction_logits = self.network(obs, mask)

        square = torch.argmax(square_logits, dim=1)
        direction = torch.argmax(direction_logits, dim=1)
        row = square // GRID_SIZE
        col = square % GRID_SIZE
        zeros = torch.zeros(self.batch_size).to(self.device)
        actions = torch.stack([zeros, row, col, direction, zeros], dim=1)
        actions[actions[:, 3] == 4, 0] = 1

        # Get log probabilities of selected actions
        square_logprob = torch.log_softmax(square_logits, dim=1)[torch.arange(square.shape[0]), square]
        direction_logprob = torch.log_softmax(direction_logits, dim=1)[torch.arange(direction.shape[0]), direction]
        logprob = square_logprob + direction_logprob

        return actions, logprob


class SupervisedAgent(NeuroAgent):
    def __init__(
        self,
        network: L.LightningModule | None = None,
        id: str = "Neuro",
        history_size: int | None = DEFAULT_HISTORY_SIZE,
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
        history_size: int | None = DEFAULT_HISTORY_SIZE,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        super().__init__(network, id, history_size, 1, device)

    def act(self, obs: Observation) -> Action:
        """
        Based on a new observation, augment the internal state and return an action.
        """
        obs.pad_observation(24)
        mask = torch.from_numpy(compute_valid_move_mask(obs)).unsqueeze(0)
        obs = torch.tensor(obs.as_tensor()).unsqueeze(0)
        action = super().act(obs, mask)[0]
        return action

    def precompile(self):
        """
        Precompile the agent to speed up inference.
        This is because when an agent plays first game, it would be idle for a while
        because it needs to compile the code.
        """
        self.reset()
        self.augment_observation(torch.zeros((1, 15, GRID_SIZE, GRID_SIZE)))
        shape = (GRID_SIZE, GRID_SIZE)
        obs = Observation(
            armies=np.zeros(shape),
            generals=np.zeros(shape),
            cities=np.zeros(shape),
            mountains=np.zeros(shape),
            neutral_cells=np.zeros(shape),
            owned_cells=np.zeros(shape),
            opponent_cells=np.zeros(shape),
            fog_cells=np.zeros(shape),
            structures_in_fog=np.zeros(shape),
            owned_land_count=0,
            owned_army_count=0,
            opponent_land_count=0,
            opponent_army_count=0,
            timestep=0,
            priority=0,
        )
        self.act(obs)
        print("Precompiled the agent")


def load_agent(path, batch_size=1, mode="base", eval_mode=True) -> NeuroAgent:
    """Load a trained agent from a checkpoint file.

    Args:
        path: Path to the checkpoint file
        batch_size: Batch size for the agent
        mode: Type of agent to create ("online", "supervised", or "base")
        eval_mode: Whether to put the model in evaluation mode

    Returns:
        NeuroAgent: Loaded agent ready for inference
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network = torch.load(path, map_location=device)
    state_dict = network["state_dict"]

    model = Network(channel_sequence=[256, 320, 384, 384], compile=True)
    model_keys = model.state_dict().keys()
    filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_keys}
    model.load_state_dict(filtered_state_dict)

    if eval_mode:
        model.eval()
    agent_id = path.split("/")[-1].split(".")[0]

    if mode == "online":
        agent = OnlineAgent(model, id=agent_id, device=device)
    elif mode == "supervised":
        agent = SupervisedAgent(model, id=agent_id, batch_size=batch_size, device=device)
    else:  # "base" or default case
        agent = NeuroAgent(model, id=agent_id, batch_size=batch_size, device=device)

    return agent
