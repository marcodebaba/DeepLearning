"""训练配置（不可变数据类）"""
from dataclasses import dataclass, field
from typing import Tuple


@dataclass(frozen=True)
class TrainConfig:
    """所有超参数集中管理，frozen=True 保证不可变"""

    # 数据
    data_root: str = "../data"
    dataset: str = "CIFAR10"  # 支持 CIFAR10 / FashionMNIST
    num_workers: int = 2

    # 模型
    num_classes: int = 10
    dropout_rate: float = 0.5

    # 训练
    epochs: int = 20
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4

    # 学习率调度（Warmup + CosineAnnealingLR）
    warmup_epochs: int = 3
    label_smoothing: float = 0.1

    # Early stopping
    patience: int = 5

    # 保存
    checkpoint_path: str = "best_model.pth"


# 数据集元信息：(均值, 标准差, 类别名称)
DATASET_META = {
    "CIFAR10": (
        (0.4914, 0.4822, 0.4465),
        (0.2470, 0.2435, 0.2616),
        ["airplane", "automobile", "bird", "cat", "deer",
         "dog", "frog", "horse", "ship", "truck"],
    ),
    "FashionMNIST": (
        (0.2860,),
        (0.3530,),
        ["T-shirt", "Trouser", "Pullover", "Dress", "Coat",
         "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"],
    ),
}
