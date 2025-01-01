import argparse
import torch
from generals.remote import GeneralsIOClient
from neuro2 import Neuro2Agent
from network2 import Network


parser = argparse.ArgumentParser()
parser.add_argument("--lobby", default=None, type=str, help="Replay ID")


def main(args):
    network = torch.load("checkpoints2/epoch=0-step=78000.ckpt", map_location="cpu")
    state_dict = network["state_dict"]
    model = Network(lr=1e-4, n_steps=9, input_dims=(29, 24, 24), compile=True)
    model_keys = model.state_dict().keys()
    filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_keys}
    model.load_state_dict(filtered_state_dict)

    agent = Neuro2Agent(model)
    user_id = "[Bot]testerboi"
    lobby_id = args.lobby
    with GeneralsIOClient(agent, user_id) as client:
        while True:
            print(client.status)
            if client.status == "off":
                client.join_private_lobby(lobby_id)
            if client.status == "lobby":
                agent.reset()
                print("RESET!")
                client.join_game()


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
