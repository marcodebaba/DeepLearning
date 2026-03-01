import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# 配置参数
batch_size = 64
learning_rate = 0.001
num_epochs = 3

# 将灰度图转换为3通道 + 归一化
transform = transforms.Compose([
    transforms.Resize((224, 224)),      # ResNet 输入尺寸是 224x224
    transforms.Grayscale(num_output_channels=3),  # 把灰度图复制成3通道
    transforms.ToTensor()
])

# 下载 FashionMNIST 数据
train_dataset = torchvision.datasets.FashionMNIST(
    root='../data', train=True, download=True, transform=transform)

test_dataset = torchvision.datasets.FashionMNIST(
    root='../data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# 使用 torchvision.models 中的预训练 ResNet18
model = resnet18(pretrained=True)

# 替换最后的全连接层：1000 → 10
model.fc = nn.Linear(model.fc.in_features, 10)

# 使用 GPU（如可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# ⏰ 训练模型
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss:.4f}")

# 🎯 测试模型准确率
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {100 * correct / total:.2f}%")

# 👀 打印一些图片内容和标签
classes = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# 从测试集中取一批样本看看
images, labels = next(iter(test_loader))
plt.figure(figsize=(10, 4))
for i in range(8):
    img = images[i].permute(1, 2, 0)  # C,H,W -> H,W,C
    plt.subplot(2, 4, i+1)
    plt.imshow(img.squeeze(), cmap="gray")
    plt.title(f"Label: {classes[labels[i]]}")
    plt.axis("off")
plt.tight_layout()
plt.show()
