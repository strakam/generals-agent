import time
import numpy as np
import argparse
import torch
from generals.remote import GeneralsIOClient
from supervised.agent import OnlineAgent
from supervised.network import Network
# from checkpoints.network import Network
# from checkpoints.neuro_tensor import OnlineAgent

def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Generals.io Agent Configuration")
    parser.add_argument("--lobby", default=None, type=str, help="Replay ID for private lobby")
    parser.add_argument("--public", default=False, action="store_true", help="Flag to join a public lobby")
    parser.add_argument("--user_id", default="trLflJK8s45a", type=str, help="User ID for the agent")
    return parser.parse_args()

def load_model_checkpoint(checkpoint_path):
    """Loads the model checkpoint and initializes the model."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["state_dict"]

    model = Network(compile=True)
    model_keys = model.state_dict().keys()
    filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_keys}
    model.load_state_dict(filtered_state_dict)

    return model

def initialize_agent(checkpoint_path):
    """Initializes the agent with the loaded model."""
    model = load_model_checkpoint(checkpoint_path)

    agent = OnlineAgent(model)
    agent.precompile()
    return agent

def join_game_loop(client, agent, lobby_id):
    """Main loop for joining and playing games."""
    while True:
        if client.status == "off":
            agent.reset()

            if lobby_id:
                client.join_private_lobby(lobby_id)
            else:
                client.join_1v1_queue()
                timeout = np.random.randint(10, 20)
                print(f"Sleeping for {timeout} seconds...")
                time.sleep(timeout)

        elif client.status == "lobby":
            client.join_game()

def main():
    args = parse_arguments()

    checkpoint_path = "checkpoints/sup212/step=64000.ckpt"
    agent = initialize_agent(checkpoint_path)

    with GeneralsIOClient(agent, args.user_id, args.public) as client:
        join_game_loop(client, agent, args.lobby)

if __name__ == "__main__":
    main()

