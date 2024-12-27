import lightning as L
from generals import GridFactory
from generals.envs import PettingZooGenerals
from generals.agents import ExpanderAgent
from neuro_agent import NeuroAgent


class EvalCallback(L.Callback):
    def __init__(self, network, eval_interval, n_eval_games):
        self.network = network
        self.eval_interval = eval_interval
        self.n_eval_games = n_eval_games

    def evaluate_agent(self, trainer):
        agents = {
            "Neuro": NeuroAgent(self.network),
            "Expander": ExpanderAgent(),
        }
        gf = GridFactory(min_grid_dims=(5, 5), max_grid_dims=(9, 9))
        env = PettingZooGenerals(
            agents=agents, grid_factory=gf, truncation=1000, render_mode=None
        )
        wins = 0
        for _ in range(self.n_eval_games):
            obs, info = env.reset()
            done = False
            while not done:
                actions = {
                    "Expander": agents["Expander"].act(obs["Expander"]),
                    "Neuro": agents["Neuro"].act(obs["Neuro"]),
                }
                obs, _, _, _, info = env.step(actions)
                if info["Neuro"]["is_done"]:
                    done = True
                    if info["Neuro"]["is_winner"]:
                        wins += 1
        env.close()
        winrate = wins / self.n_eval_games
        trainer.logger.log_metrics({"winrate": winrate}, step=trainer.global_step)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_step % self.eval_interval == 0:
            self.evaluate_agent(trainer)
