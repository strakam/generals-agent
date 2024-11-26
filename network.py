import torch
import torch.nn.functional as F


class Network(torch.nn.Module):
    def __init__(self, in_channels: int):
        super(Network, self).__init__()

        self.inc = DoubleConv(in_channels, 16)
        self.down1 = Down(16, 32)
        self.fc = torch.nn.Linear(32 * 4 * 4 + 6, 32 * 4 * 4)
        self.up1 = Up(32, 16)
        self.outc = torch.nn.Conv2d(16, 4, kernel_size=1)

    def forward(self, x, y):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x2 = x2.view(x2.size(0), -1)
        x2 = torch.cat([x2, y], dim=1)
        x2 = self.fc(x2)
        x2 = x2.view(x1.size(0), 32, 4, 4)
        x = self.up1(x2, x1)
        x = self.outc(x)
        return x


class DoubleConv(torch.nn.Module):
    """(conv => Norm => ReLU) * 2"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.double_conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(torch.nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = torch.nn.Sequential(
            torch.nn.MaxPool2d(2), DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(torch.nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.up = torch.nn.ConvTranspose2d(
            in_channels, in_channels // 2, kernel_size=2, stride=2
        )
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
