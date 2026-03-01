import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# 自定义 Dataset 类
class MulticlassWeatherDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, label


# 加载数据
def load_data(root_dir):
    image_paths = []
    labels = []

    # 获取目录中的所有文件
    for filename in os.listdir(root_dir):
        if filename.endswith(('.jpg', '.png', '.jpeg')):
            # 图像文件的完整路径
            image_paths.append(os.path.join(root_dir, filename))
            # 从文件名中提取标签，例如 img1_sunny.jpg 提取 sunny。
            label = filename.split('_')[-1].split('.')[0]
            labels.append(label)

    # 将类别标签映射为数字索引
    label_to_idx = {label: idx for idx, label in enumerate(set(labels))}
    # 将数字索引映射回类别标签
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}
    # 将标签列表转换为索引列表
    labels = [label_to_idx[label] for label in labels]

    return image_paths, labels, label_to_idx, idx_to_label


# 数据路径
root_dir = "weather_dataset"
image_paths, labels, label_to_idx, idx_to_label = load_data(root_dir)

# 划分训练集和测试集
train_paths, test_paths, train_labels, test_labels = train_test_split(
    image_paths, labels, test_size=0.2, random_state=42
)

# 数据增强（修改后）
transform = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.RandomHorizontalFlip(p=0.5),  # 减少翻转概率
    transforms.RandomRotation(10),  # 减少旋转角度
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 创建训练集 Dataset 和 DataLoader
train_dataset = MulticlassWeatherDataset(train_paths, train_labels, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 创建训练集 Dataset 和 DataLoader
test_dataset = MulticlassWeatherDataset(test_paths, test_labels, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

# 获取一个批次的数据
imgs_batch, labels_batch = next(iter(train_loader))

# 打印图片和标签的形状
# Images shape: torch.Size([16, 3, 256, 256])
print(f'Images shape: {imgs_batch.shape}')
# Labels shape: torch.Size([16])
print(f'Labels shape: {labels_batch.shape}')

'''
# 绘制批次中前6张图片，并显示类别名称
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
for i in range(6):
    ax = axes[i // 3, i % 3]
    img = imgs_batch[i].permute(1, 2, 0).numpy() * 0.5 + 0.5  # 逆归一化
    ax.imshow(img)
    ax.axis('off')
    ax.set_title(f"Label: {idx_to_label[labels_batch[i].item()]}")

plt.tight_layout()
plt.show()
'''


# 定义 CNN 模型
class MultiClassWeather(nn.Module):
    def __init__(self, num_classes):
        super(MultiClassWeather, self).__init__()
        # 输入通道数 (in_channels): 3（对应输入图像的 RGB 通道）
        # 输出通道数 (out_channels): 32（卷积层的滤波器数量），卷积操作通过 32 个不同的卷积核对输入图像进行操作，生成 32 个不同的特征图。因此，通道数从 3 变为 32。
        # 卷积核大小 (kernel_size): 3（3x3 卷积核）
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((6, 6))

        self.fc1 = nn.Linear(128 * 6 * 6, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# 判断是否有 GPU 支持
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用的设备: {device}")

# 获取实际类别数量
num_classes = len(label_to_idx)
print(f"实际类别数量: {num_classes}")

# 初始化模型
model = MultiClassWeather(num_classes=num_classes).to(device)

# 优化器
optimizer = optim.Adam(model.parameters(), lr=0.0001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
loss_fn = nn.CrossEntropyLoss()


# 4. 训练模型
def train_model(model, train_loader, loss_fn, optimizer):
    size = len(train_loader.dataset)
    num_batches = len(train_loader)
    model.train()

    train_loss, correct = 0, 0
    for X, Y in train_loader:
        X, Y = X.to(device), Y.to(device)

        # 计算预测值和损失
        predicted = model(X)
        loss = loss_fn(predicted, Y)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 记录损失和正确预测数
        train_loss += loss.item()
        correct += (predicted.argmax(1) == Y).type(torch.float).sum().item()

    # 平均损失和准确率
    train_loss /= num_batches
    correct /= size
    return train_loss, correct


# 5. 测试模型
def evaluate_model(model, test_loader):
    size = len(test_loader.dataset)
    num_batches = len(test_loader)
    model.eval()

    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, Y in test_loader:
            X, Y = X.to(device), Y.to(device)

            predicted = model(X)
            test_loss += loss_fn(predicted, Y).item()
            correct += (predicted.argmax(1) == Y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    return test_loss, correct


# 训练和测试模型
epochs = 15
train_loss, train_acc = [], []
test_loss, test_acc = [], []

# 训练和测试模型
for epoch in range(epochs):
    epoch_loss, epoch_acc = train_model(model, train_loader, loss_fn, optimizer)
    epoch_test_loss, epoch_test_acc = evaluate_model(model, test_loader)

    # 更新学习率
    scheduler.step()

    # 打印每个 epoch 的结果
    print(f"Epoch {epoch + 1:>2} | Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc * 100:.2f}% | "
          f"Test Loss: {epoch_test_loss:.4f} | Test Acc: {epoch_test_acc * 100:.2f}%")

print("训练结束!")


def predict(model, dataloader, dataset):
    model.eval()
    class_names = dataset.classes
    results = []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            for i in range(images.size(0)):
                image_path = dataset.samples[i][0]
                true_label = labels[i].item()
                pred_label = predicted[i].item()

                results.append({
                    "File Name": os.path.basename(image_path),
                    "True Label": true_label,
                    "True Class": class_names[true_label],
                    "Predicted Label": pred_label,
                    "Predicted Class": class_names[pred_label]
                })

    return results


# 调用预测函数
results = predict(model, test_loader, test_dataset)

# 打印前 10 个预测结果
for i, res in enumerate(results[:10]):
    print(f"Image: {res['File Name']}")
    print(f"  True Label: {res['True Label']} ({res['True Class']})")
    print(f"  Predicted Label: {res['Predicted Label']} ({res['Predicted Class']})")
    print("-" * 50)

# 保存模型
torch.save(model.state_dict(), 'multi-class-weather_cnn.pth')
print("模型已保存为 multi-class-weather_cnn.pth")

# 6. 绘制损失和准确率变化曲线
plt.figure(figsize=(12, 6))

# 绘制损失变化曲线
plt.subplot(1, 2, 1)
plt.plot(range(1, epochs + 1), train_loss, label='Train Loss')
plt.plot(range(1, epochs + 1), test_loss, label='Test Loss', linestyle='--')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss Curve')

# 绘制准确率变化曲线
plt.subplot(1, 2, 2)
plt.plot(range(1, epochs + 1), train_acc, label='Train Accuracy')
plt.plot(range(1, epochs + 1), test_acc, label='Test Accuracy', linestyle='--')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy Curve')

plt.tight_layout()
plt.show()
