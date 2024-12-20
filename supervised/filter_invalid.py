import os
from petting_replays import PettingZooGenerals
from replay_agent import ReplayAgent
import tqdm
import json
from joblib import Parallel, delayed

replays = [f"all_replays/old/{name}" for name in os.listdir("all_replays/old/")][19000:]


def simulate_replay(replay_name):
    replay = json.load(open(replay_name, "r"))

    A = ReplayAgent(id="A", color="red")
    B = ReplayAgent(id="B", color="blue")
    env = PettingZooGenerals(
        agents={"A": A, "B": B},
        replay_files=[replay_name],
        render_mode=None,
    )

    try:
        obs, moves, bases, values, replay = env.reset()
    except Exception as e:
        return replay_name
    A.replay_moves = moves[0]
    B.replay_moves = moves[1]
    A.general_position = bases[0]
    B.general_position = bases[1]
    A.value = values[0]
    B.value = values[1]
    A.replay = replay
    B.replay = replay
    done = False
    t = 0
    while not done:
        a, b = A.id, B.id
        p, i, j, d, s = A.act(obs[a])
        army = obs['A'][0][0][i, j]
        if army == 0: # wants to move a cell with no troops
            return replay
        p, i, j, d, s = B.act(obs[b])
        army = obs['B'][0][0][i, j]
        if army == 0: # wants to move a cell with no troops
            return replay
        actions = {a: A.act(obs[a]), b: B.act(obs[b])}
        obs, _, terminated, truncated, _ = env.step(actions)
        t+=1

        if all(terminated.values()) or all(truncated.values()):
            return None


n = 0
parallel = Parallel(n_jobs=12, return_as="generator_unordered")
output_generator = parallel(delayed(simulate_replay)(replay) for replay in replays)
print("Starting simulation")
with open("to_kill.txt", "w") as f:
    for replay in tqdm.tqdm(output_generator):
        if replay is not None:
            print(replay)
            n+=1
            print(n)
            f.write(replay + "\n")
