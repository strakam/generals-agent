from generals.core.game import Action
from generals.core.observation import Observation

from generals.agents import Agent


class ReplayAgent(Agent):
    def __init__(
        self,
        id: str = "19108",
        color: tuple[int, int, int] = (242, 61, 106),
        replay_moves: dict[int, list[int]] | None = None,
        general_position: tuple[int, int] | None = None,
    ):
        super().__init__(id, color)
        self.replay_moves = replay_moves
        self.general_position = general_position

    def act(self, observation: Observation) -> Action:
        """
        Randomly selects a valid action.
        """
        time = int(observation[0][13][0][0])

        if time not in self.replay_moves:
            return [1, self.general_position[0], self.general_position[1], 4, 0]

        return self.replay_moves[time]

    def reset(self):
        pass
