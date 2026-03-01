"""模型评估：测试准确率 + 每类准确率"""
from typing import List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def evaluate(
        model: nn.Module,
        loader: DataLoader,
        device: torch.device,
        class_names: List[str],
) -> dict:
    """
    在测试集上评估模型，返回整体准确率与每类准确率。

    Returns:
        {
            "overall_acc": float,
            "per_class":   {class_name: acc},
        }
    """
    model.eval()
    num_classes = len(class_names)

    # 每类：正确数 / 总数
    class_correct = [0] * num_classes
    class_total = [0] * num_classes

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            preds = model(images).argmax(dim=1)
            correct_mask = preds.eq(labels)

            for cls_idx in range(num_classes):
                mask = labels.eq(cls_idx)
                class_total[cls_idx] += mask.sum().item()
                class_correct[cls_idx] += (correct_mask & mask).sum().item()

    overall_acc = sum(class_correct) / sum(class_total)

    per_class = {
        class_names[i]: class_correct[i] / class_total[i]
        if class_total[i] > 0 else 0.0
        for i in range(num_classes)
    }

    return {"overall_acc": overall_acc, "per_class": per_class}


def print_report(result: dict) -> None:
    """格式化打印评估报告"""
    print(f"\n{'=' * 40}")
    print(f"  测试集准确率: {result['overall_acc']:.2%}")
    print(f"{'=' * 40}")
    print("  各类别准确率:")
    for name, acc in result["per_class"].items():
        bar = "█" * int(acc * 20)
        print(f"  {name:<15} {acc:.2%}  {bar}")
    print(f"{'=' * 40}\n")
