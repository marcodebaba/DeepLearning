"""入口：组装所有模块，执行训练与评估"""
import torch

from config import DATASET_META, TrainConfig
from dataset import build_loaders
from evaluate import evaluate, print_report
from model import ImageClassifier
from trainer import Trainer


def main() -> None:
    # ── 配置 ──────────────────────────────────────────────────────────────────
    cfg = TrainConfig(
        dataset="CIFAR10",
        epochs=20,
        batch_size=64,
        learning_rate=1e-3,
        patience=5,
        checkpoint_path="best_cifar10.pth",
    )

    # ── 设备 ──────────────────────────────────────────────────────────────────
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )
    print(f"使用设备: {device}")

    # ── 数据 ──────────────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader = build_loaders(cfg)
    _, _, class_names = DATASET_META[cfg.dataset]
    print(f"数据集: {cfg.dataset}，类别: {class_names}\n")

    # ── 模型 ──────────────────────────────────────────────────────────────────
    in_channels = 1 if cfg.dataset == "FashionMNIST" else 3
    model = ImageClassifier(
        in_channels=in_channels,
        num_classes=cfg.num_classes,
        dropout_rate=cfg.dropout_rate,
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}\n")

    if torch.cuda.device_count() > 1:
        print(f"使用 {torch.cuda.device_count()} 块 GPU 并行训练")
        model = torch.nn.DataParallel(model)

    # ── 训练 ──────────────────────────────────────────────────────────────────
    trainer = Trainer(model, cfg, device)
    trainer.fit(train_loader, val_loader)

    # ── 评估 ──────────────────────────────────────────────────────────────────
    print("加载最优模型进行评估...")
    best_state = torch.load(cfg.checkpoint_path, map_location=device, weights_only=True)
    # 兼容 DataParallel 保存的权重（key 可能带 module. 前缀）
    if any(k.startswith("module.") for k in best_state.keys()):
        best_state = {k.removeprefix("module."): v for k, v in best_state.items()}
    eval_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    eval_model.load_state_dict(best_state)

    result = evaluate(eval_model, test_loader, device, class_names)
    print_report(result)


if __name__ == "__main__":
    main()
