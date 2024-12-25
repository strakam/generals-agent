import numpy as np
from generals.core.game import Action
from generals.core.action import compute_valid_action_mask
from scipy.ndimage import maximum_filter
from generals.core.observation import Observation

from generals.agents import Agent


class NeuroAgent(Agent):
    def __init__(
        self,
        id: str = "19108",
        color: tuple[int, int, int] = (242, 61, 106),
        replay_moves: dict[int, list[int]] | None = None,
        general_position: tuple[int, int] | None = None,
        history_size: int | None = 10,
    ):
        super().__init__(id, color)
        self.replay_moves = replay_moves
        self.general_position = general_position
        self.history_size = history_size

        self.army_stack = np.zeros((self.history_size, 24, 24))
        self.army_stack[0, :, :] = np.zeros((24, 24)).astype(bool)
        self.last_army = np.zeros((24, 24)).astype(bool)
        self.cities = np.zeros((24, 24)).astype(bool)
        self.generals = np.zeros((24, 24)).astype(bool)
        self.mountains = np.zeros((24, 24)).astype(bool)
        self.enemy_saw = np.zeros((24, 24)).astype(bool)

        self.last_observation = np.zeros((29, 24, 24))

    def augment_observation(self, obs: Observation) -> np.ndarray:
        self.army_stack[1:, :, :] = self.army_stack[:-1, :, :]
        self.army_stack[0, :, :] = (obs["armies"] * obs["owned_cells"]) - self.last_army

        self.last_army = obs["armies"] * obs["owned_cells"]

        self.enemy_saw = np.logical_or(
            self.enemy_saw,
            maximum_filter(obs["opponent_cells"], size=3).astype(bool),
        )

        self.cities |= obs["cities"]
        self.generals |= obs["generals"]
        self.mountains |= obs["mountains"]
        return (
            np.stack(
                [
                    obs["armies"],
                    obs["armies"] * obs["owned_cells"],  # my armies
                    obs["armies"] * obs["opponent_cells"],  # opponent armies
                    obs["armies"] * obs["neutral_cells"],  # neutral armies
                    self.enemy_saw,  # enemy sight
                    self.generals,
                    self.cities,
                    self.mountains,
                    obs["neutral_cells"],
                    obs["owned_cells"],
                    obs["opponent_cells"],
                    obs["fog_cells"],
                    obs["structures_in_fog"],
                    obs["timestep"] * np.ones((24, 24)),
                    obs["priority"] * np.ones((24, 24)),
                    obs["owned_land_count"] * np.ones((24, 24)),
                    obs["owned_army_count"] * np.ones((24, 24)),
                    obs["opponent_land_count"] * np.ones((24, 24)),
                    obs["opponent_army_count"] * np.ones((24, 24)),
                    *self.army_stack,
                ]
            ),
            compute_valid_action_mask(obs),
        )

    def act(self, observation: Observation) -> Action:
        """
        Randomly selects a valid action.
        """
        time = int(observation[0][13][0][0])
        self.last_observation = self.augment_observation(observation)

        if time not in self.replay_moves:
            return [1, self.general_position[0], self.general_position[1], 4, 0]

        return self.replay_moves[time]

    def reset(self):
        pass
