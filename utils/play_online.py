import time
import numpy as np
import argparse
import torch
from generals.remote import GeneralsIOClient

# from checkpoints.sup112.neuro_agent import NeuroAgent
from supervised.neuro_tensor import OnlineAgent
from checkpoints.sup112.network import Network


parser = argparse.ArgumentParser()
parser.add_argument("--lobby", default=None, type=str, help="Replay ID")
parser.add_argument("--public", default=False, action="store_true", help="Public lobby")


def main(args):
    network = torch.load(
        "checkpoints/sup119/epoch=0-step=40000.ckpt", map_location="cpu"
    )
    state_dict = network["state_dict"]
    channel_sequence = [320, 384, 448, 448]
    model = Network(lr=1e-4, channel_sequence=channel_sequence, n_steps=9, compile=True)
    model_keys = model.state_dict().keys()
    filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_keys}
    model.load_state_dict(filtered_state_dict)

    agent = OnlineAgent(model)
    user_id = "trLflJK8s45a"
    with GeneralsIOClient(agent, user_id, args.public) as client:
        while True:
            if client.status == "off":
                agent.reset()
                if args.lobby is not None:
                    client.join_private_lobby(args.lobby)
                else:
                    # Sleep only in 1v1 queue
                    client.join_1v1_queue()
                    timeout = np.random.randint(10, 20)
                    print("Sleeping for", timeout, "seconds")
                    time.sleep(timeout)
            if client.status == "lobby":
                client.join_game()


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
