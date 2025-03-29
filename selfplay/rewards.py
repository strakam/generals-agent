import numba as nb
import numpy as np
from generals.core.rewards import RewardFn
from generals.core.observation import Observation
from generals.core.action import Action


@nb.njit
def calculate_army_size(castles, ownership):
    return np.sum(castles * ownership)


def compute_num_generals_owned(obs: Observation) -> float:
    """Count the number of generals owned by the player."""
    return float(np.sum(obs.generals * obs.owned_cells))


def compute_num_cities_owned(obs: Observation) -> float:
    """Count the number of cities owned by the player."""
    return float(np.sum(obs.cities * obs.owned_cells))


class CityRewardFn(RewardFn):
    """A reward function that shapes the reward based on the number of cities owned."""

    def __init__(self, shaping_weight: float = 0.3):
        self.shaping_weight = shaping_weight

    def __call__(self, prior_obs: Observation, prior_action: Action, obs: Observation) -> float:
        original_reward = compute_num_generals_owned(obs) - compute_num_generals_owned(prior_obs)

        if obs.owned_army_count == 0 or obs.opponent_army_count == 0:
            return original_reward

        city_now = calculate_army_size(obs.cities, obs.owned_cells)
        city_prev = calculate_army_size(prior_obs.cities, prior_obs.owned_cells)
        city_change = city_now - city_prev

        return float(original_reward + self.shaping_weight * city_change)


class RatioRewardFn(RewardFn):
    """A reward function that shapes the reward based on the number of generals owned."""

    def __init__(self, clip_value: float = 1.5, shaping_weight: float = 0.5):
        self.maximum_ratio = clip_value
        self.shaping_weight = shaping_weight

    def calculate_ratio_reward(self, my_army: int, opponent_army: int) -> float:
        ratio = my_army / opponent_army
        ratio = np.log(ratio) / np.log(self.maximum_ratio)
        return np.minimum(np.maximum(ratio, -1.0), 1.0)

    def __call__(self, prior_obs: Observation, prior_action: Action, obs: Observation) -> float:
        original_reward = compute_num_generals_owned(obs) - compute_num_generals_owned(prior_obs)
        # If the game is done, we dont want to shape the reward
        if obs.owned_army_count == 0 or obs.opponent_army_count == 0:
            return original_reward

        prev_ratio_reward = self.calculate_ratio_reward(prior_obs.owned_army_count, prior_obs.opponent_army_count)
        current_ratio_reward = self.calculate_ratio_reward(obs.owned_army_count, obs.opponent_army_count)
        ratio_reward = current_ratio_reward - prev_ratio_reward

        return float(original_reward + self.shaping_weight * ratio_reward)


class WinLoseRewardFn(RewardFn):
    """A reward function that shapes the reward based on the number of cities owned."""

    def __init__(self):
        pass

    def __call__(self, prior_obs: Observation, prior_action: Action, obs: Observation) -> float:
        original_reward = compute_num_generals_owned(obs) - compute_num_generals_owned(prior_obs)

        if prior_action[4] == 1:
            original_reward += 0.0015  # Encourage splitting a bit

        return float(original_reward)


class CompositeRewardFn(RewardFn):
    """A reward function that shapes the reward based on the number of cities owned."""

    def __init__(self):
        self.city_weight = 0.4
        self.ratio_weight = 0.3
        self.maximum_army_ratio = 1.5
        self.maximum_land_ratio = 1.4

    def calculate_ratio_reward(self, mine: int, opponents: int, max_ratio: float) -> float:
        ratio = mine / opponents
        ratio = np.log(ratio) / np.log(max_ratio)
        return np.minimum(np.maximum(ratio, -1.0), 1.0)

    def __call__(self, prior_obs: Observation, prior_action: Action, obs: Observation) -> float:
        original_reward = compute_num_generals_owned(obs) - compute_num_generals_owned(prior_obs)

        if obs.owned_army_count == 0 or obs.opponent_army_count == 0:
            return original_reward

        previous_army_ratio = self.calculate_ratio_reward(
            prior_obs.owned_army_count, prior_obs.opponent_army_count, self.maximum_army_ratio
        )
        current_army_ratio = self.calculate_ratio_reward(
            obs.owned_army_count, obs.opponent_army_count, self.maximum_army_ratio
        )
        army_reward = current_army_ratio - previous_army_ratio

        previous_land_ratio = self.calculate_ratio_reward(
            prior_obs.owned_land_count, prior_obs.opponent_land_count, self.maximum_land_ratio
        )
        current_land_ratio = self.calculate_ratio_reward(
            obs.owned_land_count, obs.opponent_land_count, self.maximum_land_ratio
        )
        land_reward = current_land_ratio - previous_land_ratio

        city_reward = compute_num_cities_owned(obs) - compute_num_cities_owned(prior_obs)

        if prior_action[4] == 1:
            original_reward += 0.0015  # Encourage splitting a bit

        return float(
            original_reward
            + self.ratio_weight * army_reward
            + self.city_weight * city_reward
            + self.ratio_weight * land_reward
        )
