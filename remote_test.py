import torch
from generals.remote import GeneralsIOClient
from neuro_agent import NeuroAgent
from network import Network


network = torch.load("checkpoints2/epoch=0-step=54000.ckpt", map_location="cpu")
state_dict = network["state_dict"]
model = Network(
    lr=1e-4, n_steps=9, input_dims=(29, 24, 24), compile=True
)
model_keys = model.state_dict().keys()
filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_keys}
model.load_state_dict(filtered_state_dict)

agent = NeuroAgent(model)
user_id = "[Bot]testerboi"
lobby_id = "ki8l"
with GeneralsIOClient(agent, user_id) as client:
    while True:
        print(client.status)
        if client.status == "off":
            client.join_private_lobby(lobby_id)
        if client.status == "lobby":
            client.join_game()
