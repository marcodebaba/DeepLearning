# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import datasets
import torch.nn.functional as F
import os

# 1. 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"设备: {device}")

# 2. 数据预处理
transform = {
    "train": transforms.Compose([
        transforms.Resize((128, 128)),  # 调整大小
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    "test": transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
}

# 3. 定义 CNN 模型
class NWPU_CNN(nn.Module):
    def __init__(self, num_classes=45):
        super(NWPU_CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # 计算卷积层的输出大小
        self._calculate_fc_input_size()

        # 全连接层
        self.fc1 = nn.Linear(self.fc_input_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()

    def _calculate_fc_input_size(self):
        """计算卷积层输出的特征大小，以便动态设置 fc1 的输入大小"""
        with torch.no_grad():  # 关闭梯度计算，加速计算
            x = torch.randn(1, 3, 128, 128)  # 随机输入一个 128x128 的 RGB 图片
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = self.pool(F.relu(self.conv3(x)))
            self.fc_input_size = x.view(1, -1).size(1)  # 自动获取展平后的大小

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # (64, 64, 64)
        x = self.pool(F.relu(self.conv2(x)))  # (128, 32, 32)
        x = self.pool(F.relu(self.conv3(x)))  # (256, 16, 16)

        x = x.view(x.size(0), -1)  # 展平
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == "__main__":
    # 4. 加载数据集
    data_dir = "./data/NWPU-RESISC45/Dataset"  # 替换为你的数据集路径
    batch_size = 32

    train_dataset = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=transform["train"])
    val_dataset = datasets.ImageFolder(os.path.join(data_dir, "test"), transform=transform["test"])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # 5. 实例化模型
    model = NWPU_CNN().to(device)

    # 6. 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 7. 训练模型
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_acc = 100 * correct / total
        print(
            f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}, Train Acc: {train_acc:.2f}%")

    # 8. 保存模型
    torch.save(model.state_dict(), "nwpu_resisc45_cnn.pth")
    print("模型已保存为 nwpu_resisc45_cnn.pth")

    # 9. 评估模型
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_acc = 100 * correct / total
    print(f"Validation Accuracy: {val_acc:.2f}%")
