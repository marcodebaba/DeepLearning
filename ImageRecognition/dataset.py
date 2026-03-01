"""数据集加载与数据增强"""
from typing import Tuple

import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset

from config import DATASET_META, TrainConfig


def _build_transforms(
        mean: Tuple, std: Tuple, is_train: bool
) -> T.Compose:
    """
    构建数据变换管道。
    训练集：随机裁剪 + 水平翻转（数据增强）
    验证/测试集：仅归一化
    """
    normalize = T.Normalize(mean, std)

    if is_train:
        return T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            T.ToTensor(),
            normalize,
        ])

    return T.Compose([T.ToTensor(), normalize])


def load_datasets(cfg: TrainConfig) -> Tuple[Dataset, Dataset, Dataset]:
    """
    下载并返回 (train_dataset, val_dataset, test_dataset)。
    val 复用 test split（小数据集常规做法）。
    """
    if cfg.dataset not in DATASET_META:
        raise ValueError(f"不支持的数据集: {cfg.dataset}，可选: {list(DATASET_META.keys())}")

    mean, std, _ = DATASET_META[cfg.dataset]
    train_tf = _build_transforms(mean, std, is_train=True)
    eval_tf = _build_transforms(mean, std, is_train=False)

    ds_cls = getattr(torchvision.datasets, cfg.dataset)

    train_ds = ds_cls(root=cfg.data_root, train=True, download=True, transform=train_tf)
    val_ds = ds_cls(root=cfg.data_root, train=False, download=True, transform=eval_tf)
    test_ds = ds_cls(root=cfg.data_root, train=False, download=True, transform=eval_tf)

    return train_ds, val_ds, test_ds


def build_loaders(cfg: TrainConfig) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """返回 (train_loader, val_loader, test_loader)"""
    train_ds, val_ds, test_ds = load_datasets(cfg)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
