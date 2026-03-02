"""CNN 图像识别模型"""
import torch
import torch.nn as nn
import torchvision.models as models


class ConvBlock(nn.Module):
    """Conv → BatchNorm → ReLU，统一封装为可复用块"""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, padding: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ImageClassifier(nn.Module):
    """
    通用 CNN 图像分类器。

    网络结构：
        Block1: Conv(in_ch→64) → Conv(64→64) → MaxPool
        Block2: Conv(64→128) → Conv(128→128) → MaxPool
        Block3: Conv(128→256) → Conv(256→256) → MaxPool
        Head:   AdaptiveAvgPool → FC(256→256) → Dropout → FC(256→num_classes)

    Args:
        in_channels:  输入通道数（RGB=3，灰度=1）
        num_classes:  输出类别数
        dropout_rate: Dropout 概率
    """

    def __init__(self, in_channels: int = 3, num_classes: int = 10, dropout_rate: float = 0.5):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            ConvBlock(in_channels, 64),
            ConvBlock(64, 64),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2
            ConvBlock(64, 128),
            ConvBlock(128, 128),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3
            ConvBlock(128, 256),
            ConvBlock(256, 256),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # 自适应池化→无论输入尺寸如何，输出固定为 (256, 2, 2)
        self.pool = nn.AdaptiveAvgPool2d((2, 2))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 2 * 2, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, num_classes),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Kaiming 初始化卷积层，Xavier 初始化全连接层"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)
