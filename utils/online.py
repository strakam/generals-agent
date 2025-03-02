import time
import numpy as np
import argparse
from generals.remote import GeneralsIOClient
from supervised.agent import OnlineAgent, load_fabric_checkpoint

def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Generals.io Agent Configuration")
    parser.add_argument("--lobby", default=None, type=str, help="Replay ID for private lobby")
    parser.add_argument("--public", default=False, action="store_true", help="Flag to join a public lobby")
    parser.add_argument("--user_id", default="trLflJK8s45a", type=str, help="User ID for the agent")
    return parser.parse_args()

def join_game_loop(client, agent, lobby_id):
    """Main loop for joining and playing games."""
    while True:
        if client.status == "off":
            agent.reset()

            if lobby_id:
                client.join_private_lobby(lobby_id)
            else:
                client.join_1v1_queue()
                timeout = np.random.randint(120, 150)
                print(f" ...Sleeping for {timeout} seconds...", end=" ", flush=True)
                time.sleep(timeout)

        elif client.status == "lobby":
            client.join_game()

def load_online_agent(checkpoint_path: str) -> OnlineAgent:
    """Load an agent for online play"""
    agent = load_fabric_checkpoint(
        checkpoint_path,
        batch_size=1,  # Online play is always single instance
        eval_mode=True,
        mode="online"
    )
    agent.precompile()  # Ensure the agent is ready for real-time play
    return agent

def main():
    args = parse_arguments()

    # checkpoint_path = "checkpoints/sup114/step=52000.ckpt"
    checkpoint_path = "checkpoints/selfplay/snowballer.ckpt"
    # checkpoint_path = "checkpoints/sup335/step=50000.ckpt"
    agent = load_online_agent(checkpoint_path)

    with GeneralsIOClient(agent, args.user_id, args.public) as client:
        join_game_loop(client, agent, args.lobby)

if __name__ == "__main__":
    main()

