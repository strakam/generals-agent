from generals.core.game import Action
from generals.core.observation import Observation

from generals.agents import Agent


class ReplayAgent(Agent):
    def __init__(
        self,
        id: str = "19108",
        color: tuple[int, int, int] = (242, 61, 106),
        replay_moves: dict[int, list[int]] | None = None,
    ):
        super().__init__(id, color)
        self.replay_moves = replay_moves

    def give_actions(self, replay_moves: dict[int, list[int]]):
        self.replay_moves = replay_moves

    def act(self, observation: Observation) -> Action:
        """
        Randomly selects a valid action.
        """
        time = observation[13][0][0] # timestep is 13th channel

        if time not in self.replay_moves:
            return [1, 0, 0, 0, 0]

        return self.replay_moves[time]

    def reset(self):
        pass
