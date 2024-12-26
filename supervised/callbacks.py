import lightning as L
from generals import GridFactory
from generals.envs import PettingZooGenerals
from generals.agents import ExpanderAgent
from neuro_agent import NeuroAgent


class EvalCallback(L.Callback):
    def __init__(self, network, opponent, eval_interval, n_eval_games):
        self.network = network
        self.opponent = opponent
        self.eval_interval = eval_interval
        self.n_eval_games = n_eval_games

    def evaluate_agent(self, trainer):
        agents = {
            "expander": ExpanderAgent(),
            "neuro": NeuroAgent(self.network),
        }
        gf = GridFactory(min_grid_dims=(8, 8), max_grid_dims=(12, 12))
        env = PettingZooGenerals(agents=agents, grid_factory=gf, render_mode=None)
        wins = 0
        for _ in range(self.n_eval_games):
            obs, info = env.reset()
            done = False
            while not done:
                actions = {
                    "expander": agents["expander"].act(obs["expander"]),
                    "neuro": agents["neuro"].act(obs["neuro"]),
                }
                obs, _, _, _, info = env.step(actions)
                if info["neuro"]["is_done"]:
                    done = True
                    if info["neuro"]["is_winner"]:
                        wins += 1
        winrate = wins / self.n_eval_games
        trainer.logger.log_metrics({"winrate": winrate}, step=trainer.global_step)

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        if trainer.global_step % self.eval_interval == 0:
            self.evaluate_agent()
