import functools
from collections.abc import Callable
from copy import deepcopy
from typing import Any, TypeAlias
import json

import pettingzoo  # type: ignore
from gymnasium import spaces

from generals.agents.agent import Agent
from generals.core.game import Action, Game, Info, Observation
from generals.core.grid import GridFactory
from generals.gui import GUI
from generals.gui.properties import GuiMode

AgentID: TypeAlias = str
Reward: TypeAlias = float
RewardFn: TypeAlias = Callable[[Observation, Action, bool, Info], Reward]


class PettingZooGenerals(pettingzoo.ParallelEnv):
    metadata: dict[str, Any] = {
        "render_modes": ["human"],
        "render_fps": 6,
    }

    def __init__(
        self,
        agents: dict[AgentID, Agent],
        replay_files: list[str],
        grid_factory: GridFactory | None = None,
    ):
        self.render_mode = None
        self.grid_factory = grid_factory if grid_factory is not None else GridFactory()

        # Agents
        self.agent_data = {agents[id].id: {"color": agents[id].color} for id in agents}
        self.agents = [agents[id].id for id in agents]
        self.possible_agents = self.agents

        # Replay stuff
        self.replay_files = replay_files
        self.replay_idx = 0

        assert len(self.possible_agents) == len(
            set(self.possible_agents)
        ), "Agent ids must be unique - you can pass custom ids to agent constructors."

    @functools.cache
    def observation_space(self, agent: AgentID) -> spaces.Space:
        assert agent in self.possible_agents, f"Agent {agent} not in possible agents"
        return self.game.observation_space

    @functools.cache
    def action_space(self, agent: AgentID) -> spaces.Space:
        assert agent in self.possible_agents, f"Agent {agent} not in possible agents"
        return self.game.action_space

    def render(self):
        if self.render_mode == "human":
            _ = self.gui.tick(fps=self.metadata["render_fps"])

    def next_replay(self):
        game = json.load(open(self.replay_files[self.replay_idx]))
        width = game["mapWidth"]
        height = game["mapHeight"]

        player_moves = [{}, {}]
        for move in game["moves"]:
            index, i, j, is50, turn = move[0], move[1], move[2], move[3], move[4]
            i, j = divmod(move[1], width)
            if move[2] == move[1] + 1:
                direction = 3
            elif move[2] == move[1] - 1:
                direction = 2
            elif move[2] == move[1] + game["mapWidth"]:
                direction = 1
            elif move[2] == move[1] - game["mapWidth"]:
                direction = 0
            player_moves[index][turn] = [0, i, j, direction, is50]

        map = ["." for _ in range(width * height)]

        # place cities
        for pos, value in zip(game["cities"], game["cityArmies"]):
            map[pos] = str(value - 40) if value != 50 else "x"

        # place mountains
        for pos in game["mountains"]:
            map[pos] = "#"

        # place generals
        generals = game["generals"]
        map[generals[0]] = "A"
        map[generals[1]] = "B"

        # convert to 2D array
        map = [
            map[i : i + game["mapWidth"]] for i in range(0, len(map), game["mapWidth"])
        ]

        # Pad the game with '#' to make it 24x24
        pad_width = 24 - width
        pad_height = 24 - height
        map = [[*row, *["#" for _ in range(pad_width)]] for row in map]
        map.extend([["#" for _ in range(24)] for _ in range(pad_height)])

        map_str = "\n".join(["".join(row) for row in map])
        self.replay_idx += 1
        self.replay_idx %= len(self.replay_files)
        return map_str, player_moves

    def reset(
        self, seed: int | None = None, options: dict | None = None
    ) -> tuple[dict[AgentID, Observation], dict[AgentID, dict]]:
        if options is None:
            options = {}
        self.agents = deepcopy(self.possible_agents)

        grid_string, player_moves = self.next_replay()

        self.game_length = max(len(player_moves[0].keys()), len(player_moves[1].keys()))
        grid = self.grid_factory.grid_from_string(grid_string)

        self.game = Game(grid, self.agents)
        self.time = 0

        if self.render_mode == "human":
            self.gui = GUI(self.game, self.agent_data, GuiMode.TRAIN)

        observations = {
            agent: self.game.agent_observation(agent).as_dict() for agent in self.agents
        }
        return observations, player_moves

    def step(
        self, actions: dict[AgentID, Action]
    ) -> tuple[
        dict[AgentID, Observation],
        dict[AgentID, float],
        dict[AgentID, bool],
        dict[AgentID, bool],
        dict[AgentID, Info],
    ]:
        observations, infos = self.game.step(actions)
        observations = {
            agent: observation.as_dict() for agent, observation in observations.items()
        }
        # You probably want to set your truncation based on self.game.time
        truncated = {
            agent: True if self.time > self.game_length else False
            for agent in self.agents
        }
        terminated = {
            agent: True if self.game.is_done() else False for agent in self.agents
        }
        rewards = {agent: 0 for agent in self.agents}

        # if any agent dies, all agents are terminated
        terminate = any(terminated.values())
        if terminate:
            self.agents = []
        self.time += 1
        return observations, rewards, terminated, truncated, infos

    def close(self) -> None:
        if self.render_mode == "human":
            self.gui.close()
