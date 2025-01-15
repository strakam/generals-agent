import torch
import torch.nn as nn
from torch.nn import functional as F
import lightning as L


class Network(L.LightningModule):
    def __init__(
        self,
        lr: float = 1e-4,
        n_steps: int = 100000,
        repeats: list[int] = [2, 2, 2, 1],
        channel_sequence: list[int] = [256, 320, 384, 384],
        compile: bool = False,
    ):
        super().__init__()
        c, h, w = 34, 24, 24
        self.lr = lr
        self.n_steps = n_steps

        self.backbone = Pyramid(c, repeats, channel_sequence)
        final_channels = channel_sequence[0]

        self.value_head = nn.Sequential(
            Pyramid(final_channels, [], []),
            nn.Conv2d(final_channels, 1, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(h * w, 1),
            Lambda(lambda x: 2.0 * torch.tanh(x)),  # Scale up tanh
        )

        self.square_head = nn.Sequential(
            Pyramid(final_channels, [1], [final_channels]),
            nn.Conv2d(final_channels, 1, kernel_size=3, padding=1),
        )

        self.direction_head = nn.Sequential(
            Pyramid(final_channels + 1, [1], [final_channels]),
            nn.Conv2d(final_channels, 5, kernel_size=3, padding=1),
        )

        self.square_loss = nn.CrossEntropyLoss()
        weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 1 / 20])
        self.direction_loss = nn.CrossEntropyLoss(weight=weights)
        self.value_loss = nn.MSELoss()

        if compile:
            self.backbone = torch.compile(self.backbone)
            self.value_head = torch.compile(self.value_head)
            self.square_head = torch.compile(self.square_head)
            self.direction_head = torch.compile(self.direction_head)

    @torch.compile
    def normalize_observations(self, obs):
        single_tile_army_normalize = 200
        timestep_normalize = 200
        army_normalize = 400
        land_normalize = 150

        obs[:, :4, :, :] = obs[:, :4, :, :] / single_tile_army_normalize
        obs[:, 14, :, :] = obs[:, 14, :, :] / timestep_normalize

        obs[:, 18, :, :] = obs[:, 18, :, :] / army_normalize
        obs[:, 20, :, :] = obs[:, 20, :, :] / army_normalize
        obs[:, 17, :, :] = obs[:, 17, :, :] / land_normalize
        obs[:, 19, :, :] = obs[:, 19, :, :] / land_normalize
        obs[:, 24:, :, :] = obs[:, 24:, :, :] / single_tile_army_normalize
        return obs

    # @torch.compile
    # def normalize_observations(self, obs):
    #     timestep_normalize = 500
    #     army_normalize = 500
    #     land_normalize = 200
    #
    #     obs[:, :4, :, :] = obs[:, :4, :, :] / army_normalize
    #     obs[:, 14, :, :] = obs[:, 14, :, :] / timestep_normalize
    #
    #     obs[:, 18, :, :] = obs[:, 18, :, :] / army_normalize
    #     obs[:, 20, :, :] = obs[:, 20, :, :] / army_normalize
    #     obs[:, 17, :, :] = obs[:, 17, :, :] / land_normalize
    #     obs[:, 19, :, :] = obs[:, 19, :, :] / land_normalize
    #     obs[:, 24:, :, :] = obs[:, 24:, :, :] / army_normalize
    #     return obs

    @torch.compile
    def prepare_masks(self, obs, direction_mask):
        square_mask = (1 - obs[:, 10, :, :].unsqueeze(1)) * -1e9
        direction_mask = 1 - direction_mask.permute(0, 3, 1, 2)
        pad_h = 24 - direction_mask.shape[2]
        pad_w = 24 - direction_mask.shape[3]
        mask = F.pad(direction_mask, (0, pad_w, 0, pad_h), mode="constant", value=1)
        zero_layer = torch.zeros(mask.shape[0], 1, 24, 24).to(self.device)
        direction_mask = torch.cat((mask, zero_layer), dim=1)
        direction_mask = direction_mask * -1e9

        return square_mask, direction_mask

    def forward(self, obs, mask, teacher_cells=None):
        obs = self.normalize_observations(obs)
        x = self.backbone(obs)
        # value = self.value_head(x)

        square_mask, direction_mask = self.prepare_masks(obs, mask)
        square_logits = self.square_head(x)
        square_logits = (square_logits + square_mask).flatten(1)

        if teacher_cells is not None:
            square = teacher_cells
        else:
            square = torch.argmax(square_logits, dim=1).int()
        square_reshaped = (
            F.one_hot(square.long(), num_classes=24 * 24).float().reshape(-1, 1, 24, 24)
        )
        representation_with_square = torch.cat((x, square_reshaped), dim=1)
        direction = self.direction_head(representation_with_square)
        direction = direction + direction_mask

        i, j = square // 24, square % 24
        direction = direction[torch.arange(direction.shape[0]), :, i, j]
        return square_logits, direction

    def training_step(self, batch, batch_idx):
        obs, mask, values, actions = batch
        target_i = actions[:, 1]
        target_j = actions[:, 2]
        target_cell = target_i * 24 + target_j

        square, direction = self(obs, mask, target_cell)

        square_loss = self.square_loss(square, target_cell.long())
        direction_loss = self.direction_loss(direction, actions[:, 3])
        # value_loss = self.value_loss(value.flatten(), values)

        loss = square_loss + direction_loss
        # loss
        # self.log("value_loss", value_loss, on_step=True, prog_bar=True)
        self.log("square_loss", square_loss, on_step=True, prog_bar=True)
        self.log("dir_loss", direction_loss, on_step=True, prog_bar=True)
        self.log("loss", loss, on_step=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, amsgrad=True)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.n_steps, eta_min=5e-6
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "name": "learning_rate",
            },
        }

    def on_after_backward(self):
        # Track gradients for high-level modules: backbone, value_head, square_head, direction_head
        high_level_modules = {
            "backbone": self.backbone,
            "value_head": self.value_head,
            "square_head": self.square_head,
            "direction_head": self.direction_head,
        }

        for name, module in high_level_modules.items():
            # Calculate the norm of the gradients for each module
            grad_norm = 0
            for param in module.parameters():
                if param.grad is not None:
                    grad_norm += param.grad.norm().item() ** 2
            grad_norm = grad_norm**0.5  # Take the square root to get the total norm

            # Log the gradient norm for this module
            self.log(f"grad_norm/{name}", grad_norm, on_step=True, prog_bar=True)

        # also store learning rate
        self.log("learning_rate", self.trainer.optimizers[0].param_groups[0]["lr"])


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
                in_channels = (
                    stage_channels[i - 1] if j == 0 and i > 0 else stage_channels[i]
                )
                out_channels = stage_channels[i]
                stage.append(ConvResBlock(in_channels, out_channels))
                skip_channels.append(in_channels)
            if i < len(stage_blocks) - 1:  # Downsample
                channels = stage_channels[i]
                stage.append(ConvResBlock(channels, channels, stride=2))
                skip_channels.append(channels)
            self.encoder_stages.append(stage)

        # self.reception_field_mixture = ConvResBlock(
        #     stage_channels[-1], stage_channels[-1]
        # )
        # Decoder stages
        self.decoder_stages = nn.ModuleList()

        for i in range(len(stage_blocks) - 1, -1, -1):
            stage = nn.ModuleList()
            for j in range(stage_blocks[i]):
                in_channels = (
                    stage_channels[i + 1]
                    if j == 0 and i < len(stage_blocks) - 1
                    else stage_channels[i]
                )
                out_channels = stage_channels[i]
                stage.append(
                    DeconvResBlock(in_channels, out_channels, skip_channels.pop())
                )
            if i > 0:  # Upsample
                channels = stage_channels[i]
                stage.append(
                    DeconvResBlock(channels, channels, skip_channels.pop(), stride=2)
                )
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
    def __init__(
        self, in_channels: int, out_channels: int, skip_channels: int, stride: int = 1
    ):
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
            self.residual_conv = nn.ConvTranspose2d(
                in_channels, out_channels, 1, stride=stride, output_padding=1
            )
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
