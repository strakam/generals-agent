import torch.nn as nn


class Pyramid(nn.Module):
    def __init__(
        self,
        stage_blocks: list[int] = [2, 2],
        stage_channels: list[int] = [256, 320],
    ):
        super().__init__()
        # First convolution to adjust input channels
        self.first_conv = nn.Sequential(
            nn.Conv2d(55, stage_channels[0], kernel_size=3, padding=1), nn.ReLU()
        )
        # Encoder stages
        self.encoder_stages = nn.ModuleList()
        self.skip_connections = []

        for i in range(len(stage_blocks)):
            stage = nn.ModuleList()
            for j in range(stage_blocks[i]):
                in_channels = stage_channels[i-1] if j == 0 and i > 0 else stage_channels[i]
                out_channels = stage_channels[i]
                stage.append(ConvResBlock(in_channels, out_channels))
            if i < len(stage_blocks) - 1: # Downsample
                channels = stage_channels[i]
                stage.append(ConvResBlock(channels, channels, stride=2))
            self.encoder_stages.append(stage)

        # Decoder stages
        self.decoder_stages = nn.ModuleList()

        for i in range(len(stage_blocks) - 1, -1, -1):
            stage = nn.ModuleList()
            for j in range(stage_blocks[i]):
                in_channels = stage_channels[i+1] if j == 0 and i < len(stage_blocks) - 1 else stage_channels[i]
                out_channels = stage_channels[i]
                stage.append(DeconvResBlock(in_channels, out_channels))
            if i > 0: # Upsample
                channels = stage_channels[i]
                stage.append(DeconvResBlock(channels, channels, stride=2))
            self.decoder_stages.append(stage)

    def forward(self, x):
        skip_connections = []
        x = self.first_conv(x)
        for stage in self.encoder_stages:
            for block in stage:
                skip_out, x = block(x)
                skip_connections.append(skip_out)
        skip_connections.append(x)

        print("SKIPS")
        for skip in skip_connections:
            print(skip.shape)

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
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.stride = stride

        if stride == 2:
            self.conv1 = nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels, out_channels // 2, 3, stride=stride, padding=1, output_padding=1
                ),
                nn.ReLU(),
            )
            self.residual_conv = nn.ConvTranspose2d(in_channels, out_channels, 1, stride=stride, output_padding=1)
            self.skip_conv = nn.ConvTranspose2d(in_channels, out_channels//2, 1, stride=stride, output_padding=1)
        else:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels // 2, 3, stride=stride, padding=1),
                nn.ReLU(),
            )
            self.residual_conv = nn.Conv2d(in_channels, out_channels, 1, stride=stride)
            self.skip_conv = nn.Conv2d(in_channels, out_channels//2, 1, stride=stride)

        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels // 2, out_channels, 3, padding=1),
            nn.ReLU(),
        )


    def forward(self, x, skip_in):
        print(f"x {x.shape}, skip {skip_in.shape}, my stride {self.stride}")

        skip = self.residual_conv(x)
        skip_in = self.skip_conv(skip_in)

        x = self.conv1(x)
        print(f"conved {x.shape}")
        x = x + skip_in
        x = self.conv2(x)
        x = x + skip

        return x 
