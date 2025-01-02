from generals.envs import PettingZooGenerals
import tqdm
import time


def matchup(agent1, agent2, gf, num_games=10, render_mode=None):
    agents = {
        agent1.id: agent1,
        agent2.id: agent2,
    }
    env = PettingZooGenerals(agents=agents, grid_factory=gf, render_mode=render_mode)
    win_count = {agent1.id: 0, agent2.id: 0}
    print(f"{agent1.id} vs {agent2.id}")
    for _ in tqdm.tqdm(range(num_games)):
        observations, _ = env.reset()
        done = False
        t = 0
        time_spend_thinking = 0
        game_start = time.time()
        while not done:
            actions = {}
            start = time.time()
            for agent in env.agents:
                actions[agent] = agents[agent].act(observations[agent])
            think = time.time() - start
            time_spend_thinking += think
            observations, _, terminated, truncated, infos = env.step(actions)
            done = any(terminated.values()) or any(truncated.values())
            t += 1
        for agent in agents:
            if infos[agent]["is_winner"]:
                win_count[agent] += 1
        print(f"Game length {t}.")
        print(f"Time spend thinking: {time_spend_thinking}")
        print(f"Game time: {time.time() - game_start}")
    return win_count / num_games
