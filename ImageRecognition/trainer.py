"""训练与验证循环"""
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from config import TrainConfig


# ── 每轮的统计快照（不可变）─────────────────────────────────────────────────
@dataclass(frozen=True)
class EpochStats:
    epoch: int
    train_loss: float
    train_acc: float
    val_loss: float
    val_acc: float

    def __str__(self) -> str:
        return (
            f"Epoch {self.epoch:03d} | "
            f"Train Loss {self.train_loss:.4f}  Acc {self.train_acc:.2%} | "
            f"Val   Loss {self.val_loss:.4f}  Acc {self.val_acc:.2%}"
        )


# ── 训练器 ────────────────────────────────────────────────────────────────────
class Trainer:
    """
    封装完整训练流程：
        - 混合精度训练（AMP）
        - Warmup + CosineAnnealingLR 学习率调度
        - 标签平滑 CrossEntropyLoss
        - 基于 val_loss 的 Early Stopping + 最优模型保存
        - TensorBoard 日志
    """

    def __init__(self, model: nn.Module, cfg: TrainConfig, device: torch.device):
        self.model = model.to(device)
        self.cfg = cfg
        self.device = device

        # 改进点7：标签平滑
        self.criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)

        self.optimizer = Adam(
            model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )

        # 改进点5+6：Warmup + CosineAnnealingLR
        warmup = LinearLR(
            self.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=cfg.warmup_epochs,
        )
        cosine = CosineAnnealingLR(
            self.optimizer,
            T_max=max(cfg.epochs - cfg.warmup_epochs, 1),
        )
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup, cosine],
            milestones=[cfg.warmup_epochs],
        )

        # 改进点2：混合精度 GradScaler（仅 CUDA 启用）
        self.scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

        # 改进点8：TensorBoard
        self.writer = SummaryWriter(log_dir="runs")

        self.history: List[EpochStats] = []
        self._best_val_loss: float = float("inf")
        self._patience_counter: int = 0

    # ── 内部辅助 ──────────────────────────────────────────────────────────────

    def _run_epoch(self, loader: DataLoader, *, is_train: bool) -> Tuple[float, float]:
        """运行一个 epoch，返回 (avg_loss, accuracy)"""
        self.model.train(is_train)
        total_loss = 0.0
        correct = 0
        total = 0

        ctx = torch.enable_grad() if is_train else torch.no_grad()
        with ctx:
            for images, labels in loader:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                # 改进点2：autocast 混合精度前向传播
                with torch.amp.autocast("cuda", enabled=self.device.type == "cuda"):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)

                if is_train:
                    self.optimizer.zero_grad()
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                total_loss += loss.item() * labels.size(0)
                correct += outputs.argmax(dim=1).eq(labels).sum().item()
                total += labels.size(0)

        return total_loss / total, correct / total

    def _is_improved(self, val_loss: float) -> bool:
        if val_loss < self._best_val_loss:
            self._best_val_loss = val_loss
            self._patience_counter = 0
            return True
        self._patience_counter += 1
        return False

    def _save_checkpoint(self) -> None:
        """保存模型权重，兼容 DataParallel 包装"""
        state = (
            self.model.module.state_dict()
            if isinstance(self.model, nn.DataParallel)
            else self.model.state_dict()
        )
        torch.save(state, self.cfg.checkpoint_path)

    # ── 公开接口 ──────────────────────────────────────────────────────────────

    def fit(
            self,
            train_loader: DataLoader,
            val_loader: DataLoader,
    ) -> List[EpochStats]:
        """执行完整训练流程，返回每轮的 EpochStats 列表。"""
        print(f"开始训练，设备: {self.device}，共 {self.cfg.epochs} 轮\n")

        for epoch in range(1, self.cfg.epochs + 1):
            train_loss, train_acc = self._run_epoch(train_loader, is_train=True)
            val_loss, val_acc = self._run_epoch(val_loader, is_train=False)
            self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]["lr"]
            stats = EpochStats(epoch, train_loss, train_acc, val_loss, val_acc)
            self.history.append(stats)
            print(f"{stats}  LR {current_lr:.2e}")

            # 改进点8：TensorBoard 记录
            self.writer.add_scalar("Loss/train", train_loss, epoch)
            self.writer.add_scalar("Loss/val",   val_loss,   epoch)
            self.writer.add_scalar("Acc/train",  train_acc,  epoch)
            self.writer.add_scalar("Acc/val",    val_acc,    epoch)
            self.writer.add_scalar("LR",         current_lr, epoch)

            # 保存最优模型
            if self._is_improved(val_loss):
                self._save_checkpoint()
                print(f"  ✓ 最优模型已保存 → {self.cfg.checkpoint_path}")

            # Early Stopping
            elif self._patience_counter >= self.cfg.patience:
                print(f"\nEarly Stopping：{self.cfg.patience} 轮内 val_loss 未改善。")
                break

        self.writer.close()
        print("\n训练完成！")
        return self.history
