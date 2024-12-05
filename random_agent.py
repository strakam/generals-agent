import numpy as np
import torch
import wrappers

from generals.core.game import Action
from generals.core.observation import Observation
from network import Network

from generals.agents import Agent


class RandomAgent(Agent):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(
        self,
        id: str = "Noob",
        color: tuple[int, int, int] = (242, 61, 106),
        split_prob: float = 0.25,
        idle_prob: float = 0.05,
    ):
        super().__init__(id, color)
        self.network = Network(9)
        # get network params
        print(self.network)

        self.split_probability = split_prob
        self.idle_probability = idle_prob

    # @wrappers.typed_torch_function(device, torch.float32)
    def predict(self, obs: Observation) -> Action:
        image = np.array([
            obs["armies"],
            obs["cities"],
            obs["generals"],
            obs["mountains"],
            obs["neutral_cells"],
            obs["owned_cells"],
            obs["opponent_cells"],
            obs["fog_cells"],
            obs["structures_in_fog"]
        ])
        image = torch.tensor(image, dtype=torch.float32).to(self.device)
        info_vector = torch.tensor([
            obs["owned_land_count"],
            obs["opponent_land_count"],
            obs["owned_army_count"],
            obs["opponent_army_count"],
            obs["timestep"],
            obs["priority"]
        ])
        # create batch dimension
        image = image.unsqueeze(0)
        info_vector = info_vector.unsqueeze(0)
        preds = self.network(image, info_vector)
        # get index of the argmax value for each batch action, index should be 3 numbers since we have 4D tensor
        flat_index = torch.argmax(preds.view(1, -1), dim=1)
        # convert flat index to 3D index
        index = np.unravel_index(flat_index, (4, 4, 4))
        i, j, direction = index
        return {
            "pass": 0,
            "cell": np.array([i, j]),
            "direction": direction,
            "split": 0,
        }

    def act(self, observation: Observation) -> Action:
        """
        Randomly selects a valid action.
        """
        mask = observation["action_mask"]
        observation = observation["observation"]
        self.predict(observation)

        valid_actions = np.argwhere(mask == 1)
        if len(valid_actions) == 0:  # No valid actions
            return {
                "pass": 1,
                "cell": np.array([0, 0]),
                "direction": 0,
                "split": 0,
            }
        pass_turn = 0 if np.random.rand() > self.idle_probability else 1
        split_army = 0 if np.random.rand() > self.split_probability else 1

        action_index = np.random.choice(len(valid_actions))
        cell = valid_actions[action_index][:2]
        direction = valid_actions[action_index][2]

        action = {
            "pass": pass_turn,
            "cell": cell,
            "direction": direction,
            "split": split_army,
        }
        return action

    def reset(self):
        pass
