import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.functional import max_pool2d
import lightning as L

torch._dynamo.config.capture_scalar_outputs = True


class Network(L.LightningModule):
    GRID_SIZE = 24
    DEFAULT_HISTORY_SIZE = 7
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
        temperature: float = 1.0,
    ):
        super().__init__()
        self.lr = lr
        self.n_steps = n_steps
        self.history_size = history_size
        self.batch_size = batch_size
        self.temperature = temperature
        self.n_channels = 23 + 2 * self.history_size

        self.backbone = Pyramid(self.n_channels, repeats, channel_sequence)
        final_channels = channel_sequence[0]

        self.value_head = nn.Sequential(
            Pyramid(final_channels, [], []),
            nn.Conv2d(final_channels, 1, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(24 * 24, 1),
        )

        self.policy_head = nn.Sequential(
            Pyramid(final_channels, [2], [final_channels]),
            nn.Conv2d(final_channels, 9, kernel_size=3, padding=1),
        )

        self.action_loss = nn.CrossEntropyLoss()

        if compile:
            self.backbone = torch.compile(self.backbone, fullgraph=True, dynamic=False)
            self.policy_head = torch.compile(self.policy_head, fullgraph=True, dynamic=False)
            self.value_head = torch.compile(self.value_head, fullgraph=True, dynamic=False)

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
        self.register_buffer("last_enemy_army_seen_value", torch.zeros(shape, device=device))
        self.register_buffer("last_enemy_army_seen_timestep", torch.zeros(shape, device=device))

    def reset_histories(self, obs: torch.Tensor):
        # When timestep of the observation is 0, we want to reset all data corresponding to given batch sample
        timestep_mask = obs[:, 13, 0, 0] == 0.0

        attributes_to_reset = [
            "army_stack",
            "enemy_stack",
            "last_army",
            "last_enemy_army",
            "last_enemy_army_seen_value",
            "last_enemy_army_seen_timestep",
            "seen",
            "enemy_seen",
            "cities",
            "generals",
            "mountains",
            "last_enemy_army_seen_value",
            "last_enemy_army_seen_timestep",
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

        self.last_enemy_army_seen_value = torch.where(
            current_enemy_army > 0, current_enemy_army, self.last_enemy_army_seen_value
        )
        self.last_enemy_army_seen_value += 0.01
        self.last_enemy_army_seen_timestep = torch.where(current_enemy_army > 0, 0, self.last_enemy_army_seen_timestep)

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
                self.last_enemy_army_seen_timestep,
                self.last_enemy_army_seen_value,
            ],
            dim=1,
        )
        army_stacks = torch.cat([self.army_stack, self.enemy_stack], dim=1)
        augmented_obs = torch.cat([channels, army_stacks], dim=1).float()
        return augmented_obs

    @torch.compile(dynamic=False, fullgraph=True)
    def normalize_observations(self, obs):
        timestep_normalize = 300
        army_normalize = 250
        land_normalize = 100

        # Combine all army-related normalizations into one operation
        # This includes: first 4 channels, army counts (18, 20), and history stacks (21+)
        obs[:, [0, 1, 2, 3, 18, 20] + list(range(22, obs.shape[1])), :, :] /= army_normalize

        # Timestep normalization
        obs[:, 14, :, :] /= timestep_normalize

        # Land count normalization
        obs[:, [17, 19], :, :] /= land_normalize

        return obs

    @torch.compile(dynamic=False, fullgraph=True)
    def prepare_masks(self, direction_mask):
        direction_mask = 1 - direction_mask.permute(0, 3, 1, 2)
        pad_h = 24 - direction_mask.shape[2]
        pad_w = 24 - direction_mask.shape[3]
        mask = F.pad(direction_mask, (0, pad_w, 0, pad_h), mode="constant", value=1)

        # We now need to extend the direction mask for 9 directions (4 full army, 4 half army, 1 pass)
        # Duplicate the first 4 channels (UP, DOWN, LEFT, RIGHT) for half-army moves
        # The last channel is for PASS
        full_mask = mask[:, :4, :, :]  # First 4 channels (UP, DOWN, LEFT, RIGHT)
        half_mask = mask[:, :4, :, :]  # Duplicate for half-army moves
        zero_layer = torch.zeros(mask.shape[0], 1, 24, 24).to(self.device)  # PASS layer

        direction_mask = torch.cat((full_mask, half_mask, zero_layer), dim=1)
        direction_mask = direction_mask * -1e9

        return direction_mask

    def augment_representation(self, obs):
        # Apply random rotation (0, 90, 180, or 270 degrees)
        if torch.rand(1).item() > 0.5:
            k = torch.randint(0, 4, (1,)).item()  # Random rotation 0-3 times (90 degrees each)
            obs = torch.rot90(obs, k, dims=[2, 3])
            
        # Apply random horizontal flip
        if torch.rand(1).item() > 0.5:
            obs = torch.flip(obs, dims=[3])
            
        # Apply random vertical flip
        if torch.rand(1).item() > 0.5:
            obs = torch.flip(obs, dims=[2])
            
        return obs


    def forward(self, obs, mask, action=None):
        obs = self.normalize_observations(obs.float())
        mask = self.prepare_masks(mask.float())

        # Use no_grad for backbone computation since it's frozen
        with torch.no_grad():
            representation = self.backbone(obs)

        representation_for_value = self.augment_representation(representation)

        value = self.value_head(representation_for_value).flatten()
        action_logits = self.policy_head(representation) + mask

        # Prepare flattened logits for categorical distribution
        action_logits_flat = action_logits.view(action_logits.shape[0], 9, -1)
        combined_logits = action_logits_flat.reshape(action_logits.shape[0], -1)
        # Apply temperature scaling to logits before creating distribution
        # Lower temperature makes distribution more peaked, higher makes it more uniform

        # Scale logits by inverse temperature
        combined_logits = combined_logits / self.temperature
        action_dist = torch.distributions.Categorical(logits=combined_logits)
        
        if action is None:
            # Sample action
            combined_idx = action_dist.sample()
            
            # Convert combined index to action components
            direction = combined_idx // (24 * 24)
            position = combined_idx % (24 * 24)
            i, j = position // 24, position % 24
            
            # Determine action type
            is_half_army = (direction >= 4) & (direction < 8)
            is_pass = direction == 8
            
            # Create standardized action format
            final_direction = torch.where(is_pass, 
                                         torch.full_like(direction, 8), 
                                         torch.where(is_half_army, direction - 4, direction))
            
            # Create action tensor: [is_pass, i, j, direction, is_half_army]
            action = torch.stack([
                is_pass.float(),  # Directly use is_pass instead of zeros + update
                i.float(), 
                j.float(), 
                final_direction.float(), 
                is_half_army.float()
            ], dim=1)
        else:
            # Calculate combined index from existing action
            target_i, target_j = action[:, 1], action[:, 2]
            target_direction, is_half_army = action[:, 3], action[:, 4]
            
            adjusted_direction = torch.where(target_direction < 4,
                                            target_direction + 4 * is_half_army,
                                            torch.full_like(target_direction, 8))
            
            combined_idx = adjusted_direction * 24 * 24 + target_i * 24 + target_j
        
        # Calculate log probability and entropy
        logprob = action_dist.log_prob(combined_idx)
        entropy = action_dist.entropy()

        return action, value, logprob, entropy

    @torch.compile(dynamic=False, fullgraph=True)
    def calculate_loss(self, newlogprobs, oldlogprobs, entropy, advantages, returns, values, newvalues, args):
        # Policy loss calculation using advantages
        ratio = torch.exp(newlogprobs - oldlogprobs)
        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

        # Value loss calculation
        value_loss = 0.5 * ((newvalues - returns) ** 2).mean()

        # Entropy loss
        entropy_loss = entropy.mean()

        # Total loss
        loss = pg_loss + args.vf_coef * value_loss - args.ent_coef * entropy_loss
        return loss, pg_loss, value_loss, entropy_loss, ratio

    def training_step(self, batch, args):
        obs = batch["observations"]
        masks = batch["masks"]
        actions = batch["actions"]
        advantages = batch["advantages"]
        returns = batch["returns"]
        values = batch.get("values", None)  # May not exist in older code
        oldlogprobs = batch["logprobs"]

        # Normalize advantages (optional but common in PPO)
        if hasattr(args, "norm_adv") and args.norm_adv:
            advantages_mean, advantages_std = advantages.mean(), advantages.std()
            advantages = (advantages - advantages_mean) / (advantages_std + 1e-8)

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
        _, newvalues, newlogprobs, entropy = self(obs, masks, actions)

        # Zero out the loss components of invalid samples so they have no effect.
        advantages = advantages * valid_mask
        returns = returns * valid_mask
        entropy = entropy * valid_mask

        loss, pg_loss, value_loss, entropy_loss, ratio = self.calculate_loss(
            newlogprobs, oldlogprobs, entropy, advantages, returns, values, newvalues, args
        )

        return loss, pg_loss, value_loss, entropy_loss, ratio, newlogprobs

    def predict(self, obs, mask):
        obs = self.normalize_observations(obs)
        direction_mask = self.prepare_masks(mask)

        x = self.backbone(obs)
        value = self.value_head(x)
        direction = self.policy_head(x) + direction_mask

        # Reshape direction to [batch, 9, 24*24]
        direction_flat = direction.view(direction.shape[0], 9, -1)

        # Create a tensor of shape [batch, 9*24*24] containing all possible square+direction combinations
        combined_logits = direction_flat.reshape(direction.shape[0], -1)

        # Get the argmax index from combined logits
        combined_idx = torch.argmax(combined_logits, dim=1)

        # Extract direction and position from combined index
        # combined_idx = (direction * 24 * 24) + (i * 24) + j
        adjusted_direction = combined_idx // (24 * 24)  # Integer division to get direction
        position = combined_idx % (24 * 24)  # Remainder gives position
        i, j = position // 24, position % 24

        # Determine if it's a half-army move and the direction
        is_half_army = (adjusted_direction >= 4) & (adjusted_direction < 8)
        is_pass = adjusted_direction == 8

        # Convert back to the original direction format (0-3 for directions, 4 for pass)
        final_direction = torch.where(
            is_pass,
            torch.full_like(adjusted_direction, 8),  # Pass is direction 4
            torch.where(
                is_half_army,
                adjusted_direction - 4,  # Half army directions (4-7) -> (0-3)
                adjusted_direction,  # Full army directions (0-3) stay the same
            ),
        )

        # Create action tensor with shape [batch_size, 5]
        zeros = torch.zeros_like(i, dtype=torch.float)
        action = torch.stack([zeros, i, j, final_direction, is_half_army.long()], dim=1)
        action[action[:, 3] == 8, 0] = 1  # pass action

        return action, value


    def configure_optimizers(self, lr: float = None, n_steps: int = None):
        lr = lr or self.lr
        n_steps = n_steps or self.n_steps

        # # Freeze the backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Only optimize the heads
        trainable_params = []
        trainable_params.extend(self.policy_head.parameters())
        trainable_params.extend(self.value_head.parameters())

        optimizer = torch.optim.AdamW(trainable_params, lr=lr, amsgrad=True, eps=1e-08, weight_decay=0.1, betas=(0.9, 0.999))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_steps, eta_min=1e-5)
        return optimizer, scheduler

    def on_after_backward(self):
        # Track gradients for high-level modules: backbone, value_head, policy_head
        high_level_modules = {
            "backbone": self.backbone,
            "policy_head": self.policy_head,
            "value_head": self.value_head,
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
        first_channels = 256 if stage_channels == [] else stage_channels[0]
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

    model = Network(channel_sequence=[256, 256, 288, 288], repeats=[2, 2, 2, 1], compile=True, batch_size=batch_size)

    # The checkpoint["model"] is already a state dict, no need to call .state_dict()
    state_dict = checkpoint["model"]

    # Filter and load state dict
    model_keys = model.state_dict().keys()
    filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_keys}
    model.load_state_dict(filtered_state_dict)

    if eval_mode:
        model.eval()

    return model
