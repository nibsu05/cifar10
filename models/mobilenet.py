import torch
import torch.nn as nn
import torch.nn.functional as F

class InvertedResidual(nn.Module):
    """Inverted residual block for MobileNetV2."""
    def __init__(self, inp, outp, stride, expand_ratio):
        super().__init__()
        hidden_dim = int(inp * expand_ratio)
        self.use_res_connect = (stride == 1 and inp == outp)

        layers = []
        # expansion phase
        if expand_ratio != 1:
            layers += [
                nn.Conv2d(inp, hidden_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True)
            ]
        # depthwise conv
        layers += [
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1,
                      groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True)
        ]
        # projection phase
        layers += [
            nn.Conv2d(hidden_dim, outp, kernel_size=1, bias=False),
            nn.BatchNorm2d(outp)
        ]
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.conv(x) if self.use_res_connect else self.conv(x)

class MobileNet(nn.Module):
    """Simplified MobileNetV2 (default) for CIFAR-10."""
    def __init__(self, num_classes: int = 10, width_mult: float = 1.0):
        super().__init__()
        # t, c, n, s
        cfg = [
            (1,  16, 1, 1),
            (6,  24, 2, 1),
            (6,  32, 3, 2),
            (6,  64, 4, 2),
            (6,  96, 3, 1),
            (6, 160, 3, 2),
            (6, 320, 1, 1),
        ]
        input_channel = int(32 * width_mult)
        # initial conv
        self.stem = nn.Sequential(
            nn.Conv2d(3, input_channel, 3, 1, 1, bias=False),
            nn.BatchNorm2d(input_channel),
            nn.ReLU6(inplace=True)
        )
        # build inverted residual blocks
        layers = []
        for t, c, n, s in cfg:
            out_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                layers.append(InvertedResidual(input_channel, out_channel, stride, expand_ratio=t))
                input_channel = out_channel
        # last conv
        last_channel = int(1280 * width_mult) if width_mult > 1.0 else 1280
        layers += [
            nn.Conv2d(input_channel, last_channel, 1, bias=False),
            nn.BatchNorm2d(last_channel),
            nn.ReLU6(inplace=True)
        ]
        self.features = nn.Sequential(*layers)
        # classifier
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(last_channel, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

# Factory alias

def mobilenet(num_classes: int = 10, width_mult: float = 1.0) -> MobileNet:
    return MobileNet(num_classes, width_mult)
