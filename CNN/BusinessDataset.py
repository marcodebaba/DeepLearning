import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset


# 模拟业务场景的数据增强：旋转、噪声、亮度、随机遮挡
business_transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.RandomRotation(30),                          # 模拟拍照角度不正
    transforms.ColorJitter(brightness=0.5, contrast=0.5),  # 模拟光线不均匀
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


class BusinessDigitDataset(Dataset):
    """
    模拟业务场景的手写数字数据集。
    在 MNIST 基础上叠加随机扰动，模拟真实采集数据的噪声和风格差异。

    替换成真实业务数据时：
        - 将 self.base_dataset 替换为你自己的图片目录（用 ImageFolder）
        - 保持 transform 不变，或根据业务图片特点调整
    """

    def __init__(self, train=True, root="/kaggle/working/data"):
        self.base_dataset = torchvision.datasets.MNIST(
            root=root, train=train, download=True,
            transform=None  # 先不加 transform，在 __getitem__ 里加噪声
        )
        self.transform = business_transform

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        image, label = self.base_dataset[idx]

        # 叠加随机高斯噪声（模拟扫描件杂点）
        image = self.transform(image)
        noise = torch.randn_like(image) * 0.1
        image = torch.clamp(image + noise, -1.0, 1.0)

        # 随机遮挡（模拟印章、污渍，10% 概率触发）
        if torch.rand(1).item() < 0.1:
            x = torch.randint(0, 20, (1,)).item()
            y = torch.randint(0, 20, (1,)).item()
            image[:, y:y + 8, x:x + 8] = 0.0

        return image, label


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    dataset = BusinessDigitDataset(train=True)
    fig, axes = plt.subplots(2, 8, figsize=(16, 4))
    fig.suptitle("模拟业务数据样本（含旋转 / 噪声 / 亮度变化 / 随机遮挡）")

    for i, ax in enumerate(axes.flat):
        image, label = dataset[i]
        ax.imshow(image.squeeze(), cmap="gray")
        ax.set_title(str(label))
        ax.axis("off")

    plt.tight_layout()
    plt.savefig("business_dataset_samples.png")
    print("样本预览已保存至 business_dataset_samples.png")
