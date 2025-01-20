from typing import Optional
import torch
import numpy as np
from generals.core.action import Action, compute_valid_move_mask
import lightning as L
from generals.agents import Agent
from generals.core.observation import Observation
from torch.nn.functional import max_pool2d
from functools import wraps
from modules.network import Network

GRID_SIZE = 24
DEFAULT_HISTORY_SIZE = 8
DEFAULT_BATCH_SIZE = 1


def conditional_compile(func):
    """
    Decorator that applies torch.compile only for NeuroAgent and OnlineAgent,
    but not for SupervisedAgent
    """

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if isinstance(
            self, (NeuroAgent, OnlineAgent, SelfPlayAgent)
        ) and not isinstance(self, SupervisedAgent):
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
        history_size: Optional[int] = 8,
        batch_size: Optional[int] = 1,
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
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
        self.history_size = history_size or DEFAULT_HISTORY_SIZE
        self.batch_size = batch_size or DEFAULT_BATCH_SIZE
        self.n_channels = 18 + 4 * self.history_size
        self.device = device

        if self.network is not None:
            self.network.to(device)

        self.reset()

    @conditional_compile
    def reset(self):
        """
        Reset the agent's internal state.
        """
        shape = (self.batch_size, GRID_SIZE, GRID_SIZE)
        history_shape = (self.batch_size, self.history_size, GRID_SIZE, GRID_SIZE)

        device = self.device
        tensor_specs = {
            "army_stack": history_shape,
            "enemy_stack": history_shape,
            "last_army": shape,
            "last_enemy_army": shape,
            "army_diff_stack": history_shape,
            "land_diff_stack": history_shape,
            "army_diff_buffer": (self.batch_size, 10 * self.history_size),
            "land_diff_buffer": (self.batch_size, 10 * self.history_size),
            "last_observation": (
                self.batch_size,
                self.n_channels,
                GRID_SIZE,
                GRID_SIZE,
            ),
        }

        bool_tensor_specs = {
            "cities": shape,
            "generals": shape,
            "mountains": shape,
            "seen": shape,
            "i_know_enemy_seen": shape,
            "i_know_enemy_owns": shape,
        }

        for name, spec in tensor_specs.items():
            setattr(self, name, torch.zeros(spec, device=device, dtype=torch.float))

        for name, spec in bool_tensor_specs.items():
            setattr(self, name, torch.zeros(spec, device=device, dtype=torch.bool))

    def reset_histories(self, obs: torch.Tensor):
        # When timestep of the observation is 0, we want to reset all data corresponding to given batch sample
        timestep_mask = obs[:, 13, 0, 0] == 0.0

        attributes_to_reset = [
            "army_stack",
            "enemy_stack",
            "army_diff_buffer",
            "land_diff_buffer",
            "last_army",
            "last_enemy_army",
            "seen",
            "i_know_enemy_seen",
            "i_know_enemy_owns",
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

        if obs.device != self.device:
            obs = obs.to(self.device)

        # Reset histories if needed
        self.reset_histories(obs)

        # Roll army stacks in-place
        self.army_stack.copy_(torch.roll(self.army_stack, shifts=1, dims=1))
        self.enemy_stack.copy_(torch.roll(self.enemy_stack, shifts=1, dims=1))

        # Update newest entries in-place
        self.army_stack[:, 0].copy_(
            obs[:, armies] * obs[:, owned_cells] - self.last_army
        )
        self.enemy_stack[:, 0].copy_(
            obs[:, armies] * obs[:, opponent_cells] - self.last_enemy_army
        )

        self.last_army = obs[:, armies, :, :] * obs[:, owned_cells, :, :]
        self.last_enemy_army = obs[:, armies, :, :] * obs[:, opponent_cells, :, :]

        # Update sliding window of army and land differences
        _army_diff = obs[:, owned_army_count, 0, 0] - obs[:, opponent_army_count, 0, 0]
        _land_diff = obs[:, owned_land_count, 0, 0] - obs[:, opponent_land_count, 0, 0]

        self.army_diff_buffer = torch.roll(self.army_diff_buffer, shifts=-1, dims=1)
        self.army_diff_buffer[:, -1] = _army_diff

        self.land_diff_buffer = torch.roll(self.land_diff_buffer, shifts=-1, dims=1)
        self.land_diff_buffer[:, -1] = _land_diff

        army_diff_channels = (
            self.army_diff_buffer[:, ::10]
            .unsqueeze(-1)
            .unsqueeze(-1)
            .expand(-1, -1, 24, 24)
        )

        land_diff_channels = (
            self.land_diff_buffer[:, ::10]
            .unsqueeze(-1)
            .unsqueeze(-1)
            .expand(-1, -1, 24, 24)
        )

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
                obs[:, armies, :, :] * obs[:, owned_cells, :, :],  # 0
                obs[:, armies, :, :] * obs[:, opponent_cells, :, :],  # 1
                obs[:, armies, :, :] * obs[:, neutral_cells, :, :],  # 2
                self.seen,  # 3
                self.i_know_enemy_seen,  # 4
                self.i_know_enemy_owns,  # 5
                self.generals,  # 6
                self.cities,  # 7
                self.mountains,  # 8
                obs[:, owned_cells, :, :],  # 9
                obs[:, structures_in_fog, :, :],  # 10
                obs[:, timestep, :, :] * ones,  # 11
                (obs[:, timestep, :, :] % 50) * ones / 50,  # 12
                obs[:, priority, :, :] * ones,  # 13
                obs[:, owned_land_count, :, :] * ones,  # 14
                obs[:, opponent_land_count, :, :] * ones,  # 15
                obs[:, owned_army_count, :, :] * ones,  # 16
                obs[:, opponent_army_count, :, :] * ones,  # 17
            ],
            dim=1,
        )
        augmented_obs = torch.cat(
            [
                channels,
                army_diff_channels,  # 18:18+history_size
                land_diff_channels,  # 18+history_size:18+2*history_size
                self.army_stack,  # 18+2*history_size:18+3*history_size
                self.enemy_stack,  # 18+3*history_size:18+4*history_size
            ],
            dim=1,
        )
        self.last_observation = augmented_obs
        return augmented_obs

    @conditional_compile
    def act(self, obs: np.ndarray, mask: np.ndarray) -> Action:
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
        history_size: int | None = 8,
        batch_size: int = 1,
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
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
        square_logprob = torch.log_softmax(square_logits, dim=1)[
            torch.arange(square.shape[0]), square
        ]
        direction_logprob = torch.log_softmax(direction_logits, dim=1)[
            torch.arange(direction.shape[0]), direction
        ]
        logprob = square_logprob + direction_logprob

        return actions, logprob


class SupervisedAgent(NeuroAgent):
    def __init__(
        self,
        network: L.LightningModule | None = None,
        id: str = "Neuro",
        history_size: int | None = 8,
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
        history_size: int | None = 8,
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
    ):
        super().__init__(network, id, history_size, 1, device)

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
    # Map location based on availability
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
        agent = SupervisedAgent(
            model, id=agent_id, batch_size=batch_size, device=device
        )
    else:  # "base" or default case
        agent = NeuroAgent(model, id=agent_id, batch_size=batch_size, device=device)

    return agent
