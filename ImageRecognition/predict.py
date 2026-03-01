"""对单张图片或目录进行推理"""
import sys
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

from config import DATASET_META, TrainConfig
from model import ImageClassifier


def _build_inference_transform(mean: Tuple, std: Tuple) -> T.Compose:
    """推理时只做 Resize + ToTensor + Normalize，不做数据增强"""
    return T.Compose([
        T.Resize((32, 32)),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])


def load_model(cfg: TrainConfig, device: torch.device) -> ImageClassifier:
    """从 checkpoint 加载模型权重"""
    in_channels = 1 if cfg.dataset == "FashionMNIST" else 3
    model = ImageClassifier(
        in_channels=in_channels,
        num_classes=cfg.num_classes,
        dropout_rate=0.0,  # 推理时关闭 Dropout
    )
    state = torch.load(cfg.checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device).eval()
    return model


def predict_image(
        image_path: str,
        cfg: TrainConfig,
        device: torch.device,
        top_k: int = 3,
) -> List[Tuple[str, float]]:
    """
    对单张图片推理，返回 top-k 预测结果。

    Returns:
        [(类别名, 置信度), ...]  按置信度降序排列
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"图片不存在: {image_path}")

    _, _, class_names = DATASET_META[cfg.dataset]
    mean, std, _ = DATASET_META[cfg.dataset]
    transform = _build_inference_transform(mean, std)

    # 加载图片（灰度数据集转为灰度，否则转 RGB）
    mode = "L" if cfg.dataset == "FashionMNIST" else "RGB"
    image = Image.open(path).convert(mode)
    tensor = transform(image).unsqueeze(0).to(device)  # (1, C, H, W)

    model = load_model(cfg, device)
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).squeeze(0)

    top_probs, top_indices = probs.topk(min(top_k, len(class_names)))
    return [
        (class_names[idx.item()], prob.item())
        for idx, prob in zip(top_indices, top_probs)
    ]


def predict_directory(
        dir_path: str,
        cfg: TrainConfig,
        device: torch.device,
) -> None:
    """批量推理目录下所有图片（支持 jpg/png/bmp/gif）"""
    suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
    images = [p for p in Path(dir_path).iterdir() if p.suffix.lower() in suffixes]

    if not images:
        print(f"目录中没有找到图片: {dir_path}")
        return

    print(f"共找到 {len(images)} 张图片\n{'─' * 40}")
    for img_path in sorted(images):
        results = predict_image(str(img_path), cfg, device)
        top_label, top_conf = results[0]
        others = ", ".join(f"{n}({c:.0%})" for n, c in results[1:])
        print(f"{img_path.name:<30} → {top_label:<15} {top_conf:.2%}  [{others}]")


# ── CLI 入口 ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    """
    用法：
        python predict.py path/to/image.jpg          # 单张图片
        python predict.py path/to/images/            # 整个目录
    """
    if len(sys.argv) < 2:
        print("用法: python predict.py <图片路径或目录>")
        sys.exit(1)

    target = sys.argv[1]
    cfg = TrainConfig(checkpoint_path="best_cifar10.pth")
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )

    if Path(target).is_dir():
        predict_directory(target, cfg, device)
    else:
        results = predict_image(target, cfg, device)
        print(f"\n图片: {target}")
        print("─" * 35)
        for rank, (name, conf) in enumerate(results, 1):
            bar = "█" * int(conf * 20)
            print(f"  Top{rank}  {name:<15} {conf:.2%}  {bar}")
