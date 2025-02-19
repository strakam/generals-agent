import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.functional import max_pool2d
import lightning as L

torch._dynamo.config.capture_scalar_outputs = True


class Network(L.LightningModule):
    GRID_SIZE = 24
    DEFAULT_HISTORY_SIZE = 5
    DEFAULT_BATCH_SIZE = 1

    def __init__(
        self,
        lr: float = 1e-4,
        n_steps: int = 100000,
        repeats: list[int] = [2, 2, 1, 1],
        channel_sequence: list[int] = [192, 224, 256, 256],
        compile: bool = False,
        history_size: int = DEFAULT_HISTORY_SIZE,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ):
        super().__init__()
        self.lr = lr
        self.n_steps = n_steps
        self.history_size = history_size
        self.batch_size = batch_size
        self.n_channels = 21 + 2 * self.history_size

        self.backbone = Pyramid(self.n_channels, repeats, channel_sequence)
        final_channels = channel_sequence[0]

        self.square_head = nn.Sequential(
            Pyramid(final_channels, [1], [final_channels]),
            nn.Conv2d(final_channels, 1, kernel_size=3, padding=1),
        )

        self.direction_head = nn.Sequential(
            Pyramid(final_channels + 1, [1], [final_channels]),
            nn.Conv2d(final_channels, 5, kernel_size=3, padding=1),
        )

        # self.value_head = nn.Sequential(
        #     Pyramid(final_channels, [], []),
        #     nn.Conv2d(final_channels, 1, kernel_size=3, padding=1),
        #     nn.ReLU(),
        #     nn.Flatten(),
        #     nn.Linear(24 * 24, 1),
        #     Lambda(lambda x: torch.tanh(x)),  # Scale up tanh
        # )

        self.square_loss = nn.CrossEntropyLoss()
        self.direction_loss = nn.CrossEntropyLoss()

        if compile:
            self.backbone = torch.compile(self.backbone, fullgraph=True, dynamic=False)
            self.square_head = torch.compile(self.square_head, fullgraph=True, dynamic=False)
            self.direction_head = torch.compile(self.direction_head, fullgraph=True, dynamic=False)
            # self.value_head = torch.compile(self.value_head, fullgraph=True, dynamic=False)

    @torch.compile(dynamic=False, fullgraph=True)
    def reset(self):
        """
        Reset the network's internal state.
        The state contains things that the network remembers over time (positions of generals, etc.).
        """
        shape = (self.batch_size, 24, 24)
        history_shape = (self.batch_size, self.history_size, 24, 24)
        device = self.device

        self.register_buffer("army_stack", torch.zeros(history_shape, device=device))
        self.register_buffer("enemy_stack", torch.zeros(history_shape, device=device))
        self.register_buffer("last_army", torch.zeros(shape, device=device))
        self.register_buffer("last_enemy_army", torch.zeros(shape, device=device))
        self.register_buffer("cities", torch.zeros(shape, dtype=torch.bool, device=device))
        self.register_buffer("generals", torch.zeros(shape, dtype=torch.bool, device=device))
        self.register_buffer("mountains", torch.zeros(shape, dtype=torch.bool, device=device))
        self.register_buffer("seen", torch.zeros(shape, dtype=torch.bool, device=device))
        self.register_buffer("enemy_seen", torch.zeros(shape, dtype=torch.bool, device=device))

        if device.type == "cuda":
            torch.cuda.synchronize()  # Ensure buffers are fully initialized on GPU

    def reset_histories(self, obs: torch.Tensor):
        # When timestep of the observation is 0, we want to reset all data corresponding to given batch sample
        timestep_mask = obs[:, 13, 0, 0] == 0.0

        attributes_to_reset = [
            "army_stack",
            "enemy_stack",
            "last_army",
            "last_enemy_army",
            "seen",
            "enemy_seen",
            "cities",
            "generals",
            "mountains",
        ]

        for attr in attributes_to_reset:
            getattr(self, attr)[timestep_mask] = 0

    @torch.compile(dynamic=False, fullgraph=True)
    def augment_observation(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Here the network augments what it knows about the game with the new observation.
        This is then further used to make a decision.
        """
        armies = 0
        generals = 1
        cities = 2
        mountains = 3
        neutral_cells = 4
        owned_cells = 5
        opponent_cells = 6
        fog_cells = 7
        structures_in_fog = 8
        owned_land_count = 9
        owned_army_count = 10
        opponent_land_count = 11
        opponent_army_count = 12
        timestep = 13
        priority = 14

        self.reset_histories(obs)

        # Calculate current army states
        current_army = obs[:, armies, :, :] * obs[:, owned_cells, :, :]
        current_enemy_army = obs[:, armies, :, :] * obs[:, opponent_cells, :, :]

        # Update history stacks by shifting and adding new differences
        self.army_stack[:, 1:, :, :] = self.army_stack[:, :-1, :, :].clone()
        self.enemy_stack[:, 1:, :, :] = self.enemy_stack[:, :-1, :, :].clone()

        self.army_stack[:, 0, :, :] = current_army - self.last_army
        self.enemy_stack[:, 0, :, :] = current_enemy_army - self.last_enemy_army

        # Store current states for next iteration
        self.last_army = current_army
        self.last_enemy_army = current_enemy_army

        self.seen |= max_pool2d(obs[:, owned_cells, :, :], 3, 1, 1).bool()
        self.enemy_seen |= max_pool2d(obs[:, opponent_cells, :, :], 3, 1, 1).bool()

        self.cities |= obs[:, cities, :, :].bool()
        self.generals |= obs[:, generals, :, :].bool()
        self.mountains |= obs[:, mountains, :, :].bool()

        ones = torch.ones((self.batch_size, 24, 24), device=self.device)
        channels = torch.stack(
            [
                obs[:, armies, :, :],
                obs[:, armies, :, :] * obs[:, owned_cells, :, :],
                obs[:, armies, :, :] * obs[:, opponent_cells, :, :],
                obs[:, armies, :, :] * obs[:, neutral_cells, :, :],
                self.seen,
                self.enemy_seen,  # enemy sight
                self.generals,
                self.cities,
                self.mountains,
                obs[:, neutral_cells, :, :],
                obs[:, owned_cells, :, :],
                obs[:, opponent_cells, :, :],
                obs[:, fog_cells, :, :],
                obs[:, structures_in_fog, :, :],
                obs[:, timestep, :, :] * ones,
                (obs[:, timestep, :, :] % 50) * ones / 50,
                obs[:, priority, :, :] * ones,
                obs[:, owned_land_count, :, :] * ones,
                obs[:, owned_army_count, :, :] * ones,
                obs[:, opponent_land_count, :, :] * ones,
                obs[:, opponent_army_count, :, :] * ones,
            ],
            dim=1,
        )
        army_stacks = torch.cat([self.army_stack, self.enemy_stack], dim=1)
        augmented_obs = torch.cat([channels, army_stacks], dim=1).float()
        return augmented_obs

    @torch.compile(dynamic=False, fullgraph=True)
    def normalize_observations(self, obs):
        timestep_normalize = 500
        army_normalize = 500
        land_normalize = 200

        # Combine all army-related normalizations into one operation
        # This includes: first 4 channels, army counts (18, 20), and history stacks (21+)
        obs[:, [0, 1, 2, 3, 18, 20] + list(range(21, obs.shape[1])), :, :] /= army_normalize

        # Timestep normalization
        obs[:, 14, :, :] /= timestep_normalize

        # Land count normalization
        obs[:, [17, 19], :, :] /= land_normalize

        return obs

    @torch.compile(dynamic=False, fullgraph=True)
    def prepare_masks(self, obs, direction_mask):
        square_mask = (1 - obs[:, 10, :, :].unsqueeze(1)) * -1e9
        direction_mask = 1 - direction_mask.permute(0, 3, 1, 2)
        B, C, h, w = direction_mask.shape
        mask = torch.full((B, C, 24, 24), 1, device=self.device, dtype=direction_mask.dtype)
        mask[:, :, :h, :w] = direction_mask
        zero_layer = torch.zeros(B, 1, 24, 24, device=self.device)
        direction_mask = torch.cat((mask, zero_layer), dim=1) * -1e9

        return square_mask, direction_mask

    def forward(self, obs, mask, action=None):
        obs = self.normalize_observations(obs.float())
        square_mask, direction_mask = self.prepare_masks(obs, mask.float())

        representation = self.backbone(obs)

        square_logits_unmasked = self.square_head(representation)
        square_logits = (square_logits_unmasked + square_mask).flatten(1)

        # Sample square from categorical distribution
        square_dist = torch.distributions.Categorical(logits=square_logits)
        if action is None:
            square = square_dist.sample()
        else:
            square = action[:, 1] * 24 + action[:, 2]
        i, j = square // 24, square % 24

        # Get direction logits based on sampled square
        square_reshaped = F.one_hot(square.long(), num_classes=24 * 24).float().reshape(-1, 1, 24, 24)
        representation_with_square = torch.cat((representation, square_reshaped), dim=1)
        direction = self.direction_head(representation_with_square)
        direction += direction_mask
        direction = direction[torch.arange(direction.shape[0]), :, i.long(), j.long()]

        # Sample direction
        direction_dist = torch.distributions.Categorical(logits=direction)
        if action is None:
            direction = direction_dist.sample()
        else:
            direction = action[:, 3]

        # Calculate log probabilities
        square_logprob = square_dist.log_prob(square)
        direction_logprob = direction_dist.log_prob(direction)
        logprob = square_logprob + direction_logprob

        entropy = square_dist.entropy() + direction_dist.entropy()

        # Create action tensor with shape [batch_size, 5]
        zeros = torch.zeros_like(square, dtype=torch.float)
        action = torch.stack([zeros, i, j, direction, zeros], dim=1)
        action[action[:, 3] == 4, 0] = 1  # pass action

        return action, logprob, entropy

    @torch.compile(dynamic=False, fullgraph=True)
    def calculate_loss(self, newlogprobs, oldlogprobs, entropy, returns, args):
        ratio = torch.exp(newlogprobs - oldlogprobs)
        pg_loss1 = -returns * ratio
        pg_loss2 = -returns * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()
        entropy_loss = entropy.mean()
        loss = pg_loss - args.ent_coef * entropy_loss
        return loss, pg_loss, entropy_loss, ratio

    def training_step(self, batch, args):
        obs = batch["observations"]
        masks = batch["masks"]
        actions = batch["actions"]
        returns = batch["returns"]
        oldlogprobs = batch["logprobs"]

        # Flag batch samples where the raw owned cells channel (index 10) sums to zero.
        # If a sample has no owned cells then its loss contributions will be zero.
        valid_mask = (obs[:, 10, :, :].sum(dim=(1, 2)) != 0).float()
        # Flag samples where the player owns multiple generals
        # Get masks for owned cells and generals
        owned_cells = obs[:, 10, :, :] > 0
        generals = obs[:, 6, :, :] > 0
        # Count locations where both owned cells and generals overlap
        num_owned_generals = ((owned_cells & generals).float().sum(dim=(1, 2)) > 1).float()
        # Zero out samples where player owns multiple generals
        valid_mask = valid_mask * (1 - num_owned_generals)

        # Compute network outputs
        _, newlogprobs, entropy = self(obs, masks, actions)

        # Zero out the loss components of invalid samples so they have no effect.
        # Since pg_loss involves returns and entropy_loss is an average,
        # setting returns and entropy to zero for these samples effectively cancels their loss.
        returns = returns * valid_mask
        entropy = entropy * valid_mask

        loss, pg_loss, entropy_loss, ratio = self.calculate_loss(newlogprobs, oldlogprobs, entropy, returns, args)

        return loss, pg_loss, entropy_loss, ratio, newlogprobs

    @torch.compile(dynamic=False, fullgraph=True)
    def predict(self, obs, mask):
        obs = self.normalize_observations(obs.float())
        square_mask, direction_mask = self.prepare_masks(obs, mask.float())

        representation = self.backbone(obs)

        # Get square logits and apply mask
        square_logits_unmasked = self.square_head(representation)
        square_logits = (square_logits_unmasked + square_mask).flatten(1)

        # Use argmax instead of sampling
        square = square_logits.argmax(dim=1).long()
        i, j = square // 24, square % 24

        # Get direction logits based on selected square
        square_reshaped = F.one_hot(square, num_classes=24 * 24).float().reshape(-1, 1, 24, 24)
        representation_with_square = torch.cat((representation, square_reshaped), dim=1)
        direction = self.direction_head(representation_with_square)
        direction += direction_mask
        direction = direction[torch.arange(direction.shape[0]), :, i, j]

        # Use argmax for direction
        direction = direction.argmax(dim=1)

        # Create action tensor with shape [batch_size, 5]
        zeros = torch.zeros_like(square, dtype=torch.float)
        action = torch.stack([zeros, i, j, direction, zeros], dim=1)
        action[action[:, 3] == 4, 0] = 1  # pass action

        return action

    def configure_optimizers(self, lr: float = None, n_steps: int = None):
        lr = lr or self.lr
        n_steps = n_steps or self.n_steps

        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, amsgrad=True, eps=1e-07)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_steps, eta_min=1e-5)
        return optimizer, scheduler

    def on_after_backward(self):
        # Track gradients for high-level modules: backbone, value_head, square_head, direction_head
        high_level_modules = {
            "backbone": self.backbone,
            "square_head": self.square_head,
            "direction_head": self.direction_head,
        }

        grad_norms = {}
        for name, module in high_level_modules.items():
            # Calculate the norm of the gradients for each module
            grad_norm = 0
            for param in module.parameters():
                if param.grad is not None:
                    grad_norm += param.grad.norm().item() ** 2
            grad_norm = grad_norm**0.5  # Take the square root to get the total norm
            grad_norms[name] = grad_norm

        return grad_norms


class Pyramid(nn.Module):
    def __init__(
        self,
        input_channels: int = 55,
        stage_blocks: list[int] = [2, 2, 2],
        stage_channels: list[int] = [256, 320, 384],
    ):
        super().__init__()
        # First convolution to adjust input channels
        first_channels = 192 if stage_channels == [] else stage_channels[0]
        self.first_conv = nn.Sequential(
            nn.Conv2d(input_channels, first_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        # Encoder stages
        self.encoder_stages = nn.ModuleList()
        self.skip_connections = []

        skip_channels = []

        for i in range(len(stage_blocks)):
            stage = nn.ModuleList()
            for j in range(stage_blocks[i]):
                in_channels = stage_channels[i - 1] if j == 0 and i > 0 else stage_channels[i]
                out_channels = stage_channels[i]
                stage.append(ConvResBlock(in_channels, out_channels))
                skip_channels.append(in_channels)
            if i < len(stage_blocks) - 1:  # Downsample
                channels = stage_channels[i]
                stage.append(ConvResBlock(channels, channels, stride=2))
                skip_channels.append(channels)
            self.encoder_stages.append(stage)

        # Decoder stages
        self.decoder_stages = nn.ModuleList()

        for i in range(len(stage_blocks) - 1, -1, -1):
            stage = nn.ModuleList()
            for j in range(stage_blocks[i]):
                in_channels = stage_channels[i + 1] if j == 0 and i < len(stage_blocks) - 1 else stage_channels[i]
                out_channels = stage_channels[i]
                stage.append(DeconvResBlock(in_channels, out_channels, skip_channels.pop()))
            if i > 0:  # Upsample
                channels = stage_channels[i]
                stage.append(DeconvResBlock(channels, channels, skip_channels.pop(), stride=2))
            self.decoder_stages.append(stage)

    def forward(self, x):
        skip_connections = []
        x = self.first_conv(x)
        for stage in self.encoder_stages:
            for block in stage:
                skip_out, x = block(x)
                skip_connections.append(skip_out)
        for stage in self.decoder_stages:
            for block in stage:
                skip_in = skip_connections.pop()
                x = block(x, skip_in)
        return x


class ConvResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, 3, stride=stride, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels // 2, out_channels, 3, padding=1),
            nn.ReLU(),
        )
        self.residual_conv = nn.Conv2d(in_channels, out_channels, 1, stride=stride)

    def forward(self, x):
        skip_out = x
        x = self.conv(x) + self.residual_conv(x)
        return skip_out, x


class DeconvResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, skip_channels: int, stride: int = 1):
        super().__init__()
        self.stride = stride

        if stride == 2:
            self.conv1 = nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels,
                    out_channels // 2,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                    output_padding=1,
                ),
                nn.ReLU(),
            )
            self.residual_conv = nn.ConvTranspose2d(in_channels, out_channels, 1, stride=stride, output_padding=1)
        else:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels // 2, 3, stride=stride, padding=1),
                nn.ReLU(),
            )
            self.residual_conv = nn.Conv2d(in_channels, out_channels, 1, stride=stride)

        self.skip_conv = nn.Conv2d(
            skip_channels,
            out_channels // 2,
            kernel_size=1,
            stride=1,
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels // 2, out_channels, 3, padding=1),
            nn.ReLU(),
        )

    def forward(self, x, skip_in):
        skip = self.residual_conv(x)
        skip_in = self.skip_conv(skip_in)

        x = self.conv1(x)
        x = x + skip_in
        x = self.conv2(x)
        x = x + skip

        return x


class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    @torch.compile
    def forward(self, x):
        return self.func(x)


def load_network(path: str, batch_size: int, eval_mode: bool = True) -> Network:
    """Load a network from a checkpoint file.

    Args:
        path: Path to the checkpoint file
        eval_mode: Whether to put the model in evaluation mode

    Returns:
        Network: Loaded network
    """
    # model = torch.compile(model, fullgraph=True, dynamic=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network = torch.load(path, map_location=device)
    state_dict = network["state_dict"]

    model = Network(channel_sequence=[192, 224, 256, 256], repeats=[2, 2, 1, 1], compile=True, batch_size=batch_size)
    model_keys = model.state_dict().keys()
    filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_keys}
    model.load_state_dict(filtered_state_dict)
    return model


def load_fabric_checkpoint(path: str, batch_size: int, eval_mode: bool = True) -> Network:
    """Load a network from a Fabric-style checkpoint file.

    Args:
        path: Path to the Fabric checkpoint file
        batch_size: Batch size for the network
        eval_mode: Whether to put the model in evaluation mode

    Returns:
        Network: Loaded network
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(path, map_location=device)

    # Handle both direct state dict and Fabric-style nested state dict
    if "model" in checkpoint and isinstance(checkpoint["model"], dict):
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    model = Network(batch_size=batch_size, compile=True)
    model_keys = model.state_dict().keys()
    filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_keys}
    model.load_state_dict(filtered_state_dict)

    model = model.to(device)
    if eval_mode:
        model.eval()

    return model
