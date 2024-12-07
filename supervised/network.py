import torch
import torch.nn as nn
from torch.nn import functional as F
import lightning as L


class Network(L.LightningModule):
    def __init__(
        self,
        input_dims: tuple[int, int, int] = (55, 24, 24),
        repeats: list[int] = [2, 2, 2],
        channel_sequence: list[int] = [32, 32, 32],
    ):
        super().__init__()
        c, h, w = input_dims

        self.backbone = Pyramid(c, repeats, channel_sequence)
        final_channels = channel_sequence[0]

        self.value_head = nn.Sequential(
            Pyramid(final_channels, [], []),
            nn.Conv2d(final_channels, 1, kernel_size=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(h * w, 1),
        )

        self.policy_pyramid = Pyramid(final_channels, [1], [final_channels])
        self.square_head = nn.Conv2d(final_channels, 1, kernel_size=1)
        self.direction_head = nn.Sequential(
            nn.Conv2d(final_channels + 1, 1, kernel_size=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(h * w, 4),
        )

    def forward(self, x):
        x = x / 50
        x = self.backbone(x)
        value = self.value_head(x)

        square_logits = self.square_head(self.policy_pyramid(x)).flatten(1)
        square = F.gumbel_softmax(square_logits, dim=1)
        square_reshaped = square.reshape(-1, 24, 24)
        direction = self.direction_head(
            torch.cat((x, square_reshaped.unsqueeze(1)), dim=1)
        )

        return value, square_logits, direction

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        target_i = targets[:, 1]
        target_j = targets[:, 2]
        target_cell = (target_i * 24 + target_j).float()

        _, square, direction = self(inputs) # square is 

        # crossentropy loss for square loss and direction loss
        square_loss = F.cross_entropy(square, target_cell.long())
        direction_loss = F.cross_entropy(direction, targets[:, 3])
        
        loss = square_loss + direction_loss

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.0005)


class Pyramid(nn.Module):
    def __init__(
        self,
        input_channels: int = 55,
        stage_blocks: list[int] = [2, 2, 2],
        stage_channels: list[int] = [256, 320, 384],
    ):
        super().__init__()
        # First convolution to adjust input channels
        first_channels = 32 if stage_channels == [] else stage_channels[0]
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
