import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用的设备: {device}")

# ================================================================
# 1. 数据预处理
# ================================================================
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

business_transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.5, contrast=0.5),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# ================================================================
# 2. 模拟业务数据集（含旋转 / 噪声 / 亮度变化 / 随机遮挡）
# ================================================================
class BusinessDigitDataset(Dataset):
    def __init__(self, train=True, root='/kaggle/working/data'):
        self.base_dataset = torchvision.datasets.MNIST(
            root=root, train=train, download=True, transform=None
        )

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        image, label = self.base_dataset[idx]
        image = business_transform(image)

        # 随机高斯噪声（模拟扫描件杂点）
        noise = torch.randn_like(image) * 0.1
        image = torch.clamp(image + noise, -1.0, 1.0)

        # 随机遮挡（10% 概率，模拟印章污渍）
        if torch.rand(1).item() < 0.1:
            x = torch.randint(0, 20, (1,)).item()
            y = torch.randint(0, 20, (1,)).item()
            image[:, y:y + 8, x:x + 8] = 0.0

        return image, label


# ================================================================
# 3. 定义 CNN 模型
# ================================================================
class CNNDigitClassifier(nn.Module):
    def __init__(self, input_size=(1, 1, 28, 28)):
        super(CNNDigitClassifier, self).__init__()
        # MNIST数据集是灰度图像（单通道），32个3*3的卷积核
        # conv1的输入尺寸为1x28x28，经过卷积后conv1的输出尺寸为26x26x32
        self.conv1 = nn.Conv2d(1, 32, 3)
        # conv2的输入通道数是conv1的输出通道数。输出通道数是一个超参数，通常选择大于前一层的输出通道数，以增加模型容量
        # conv2的输入尺寸为13x13x32，经过卷积后conv2的输出尺寸为11x11x64
        self.conv2 = nn.Conv2d(32, 64, 3)

        # 动态计算卷积和池化后的输出尺寸
        with torch.no_grad():
            x = torch.zeros(input_size)
            x = F.max_pool2d(F.relu(self.conv1(x)), 2)
            x = F.max_pool2d(F.relu(self.conv2(x)), 2)
            _, _, h, w = x.shape
            n_features = 64 * h * w

        # conv2池化后输出尺寸为5x5x64，全连接层1的输出维度为128
        self.fc1 = nn.Linear(n_features, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # 输入：26x26x32，池化后输出尺寸为13x13x32
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        # 输入：11x11x64，池化后输出尺寸为5x5x64
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# ================================================================
# 4. 训练 / 评估 / 预测
# ================================================================
def train_model(model, train_loader, loss_fn, optimizer):
    model.train()
    train_loss, correct = 0, 0
    for X, Y in train_loader:
        X, Y = X.to(device), Y.to(device)
        optimizer.zero_grad()
        output = model(X)
        loss = loss_fn(output, Y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        correct += (output.argmax(1) == Y).type(torch.float).sum().item()
    return train_loss / len(train_loader), correct / len(train_loader.dataset)


def evaluate_model(model, test_loader, loss_fn):
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, Y in test_loader:
            X, Y = X.to(device), Y.to(device)
            output = model(X)
            test_loss += loss_fn(output, Y).item()
            correct += (output.argmax(1) == Y).type(torch.float).sum().item()
    return test_loss / len(test_loader), correct / len(test_loader.dataset)


def predict_digit(image_path, model, t):
    model.eval()
    image = Image.open(image_path).convert('L')
    image = t(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
        predicted = output.argmax(1).item()
    print(f'预测结果: {predicted}')
    return predicted


# ================================================================
# 5. 微调工具函数
# ================================================================
def freeze_conv_layers(model):
    for param in model.conv1.parameters():
        param.requires_grad = False
    for param in model.conv2.parameters():
        param.requires_grad = False
    print("卷积层已冻结，只训练全连接层")


def fine_tune(model, train_loader, val_loader, epochs=10, lr=0.0001, patience=5):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        train_loss, train_acc = train_model(model, train_loader, loss_fn, optimizer)
        val_loss, val_acc = evaluate_model(model, val_loader, loss_fn)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        print(f"Epoch {epoch + 1} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc * 100:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc * 100:.2f}%")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "fine_tuned_cnn.pth")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}, best val loss: {best_val_loss:.4f}")
                break

    print("最优微调模型已保存至 fine_tuned_cnn.pth")
    return history


# ================================================================
# 6. 曲线可视化
# ================================================================
def plot_curves(history, filename="training_curves.png"):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, history["train_loss"], label="Train Loss")
    ax1.plot(epochs, history["val_loss"], label="Val Loss")
    ax1.set_title("Loss Curve")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()

    ax2.plot(epochs, history["train_acc"], label="Train Acc")
    ax2.plot(epochs, history["val_acc"], label="Val Acc")
    ax2.set_title("Accuracy Curve")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(filename)
    print(f"训练曲线已保存至 {filename}")


# ================================================================
# 主流程
# ================================================================
if __name__ == '__main__':
    # --- Step 1: 预训练（MNIST 原始数据）---
    print("===== Step 1: 预训练阶段 =====")
    train_dataset = torchvision.datasets.MNIST(
        root='/kaggle/working/data', train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.MNIST(
        root='/kaggle/working/data', train=False, download=True, transform=transform
    )
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    model = CNNDigitClassifier().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_loss = float("inf")
    patience, patience_counter = 5, 0

    for epoch in range(20):
        train_loss, train_acc = train_model(model, train_loader, loss_fn, optimizer)
        test_loss, test_acc = evaluate_model(model, test_loader, loss_fn)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(test_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(test_acc)
        print(f'Epoch {epoch + 1} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc * 100:.2f}% | '
              f'Test Loss: {test_loss:.4f} | Test Acc: {test_acc * 100:.2f}%')

        if test_loss < best_val_loss:
            best_val_loss = test_loss
            torch.save(model.state_dict(), "mnist_cnn.pth")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}, best val loss: {best_val_loss:.4f}")
                break

    print("预训练完成，权重已保存至 mnist_cnn.pth")
    plot_curves(history, filename="pretrain_curves.png")

    # --- Step 2: 微调（业务数据）---
    print("\n===== Step 2: 微调阶段 =====")
    model.load_state_dict(torch.load("mnist_cnn.pth", map_location=device))
    freeze_conv_layers(model)

    biz_train = BusinessDigitDataset(train=True)
    biz_val = BusinessDigitDataset(train=False)
    biz_train_loader = DataLoader(biz_train, batch_size=64, shuffle=True)
    biz_val_loader = DataLoader(biz_val, batch_size=64, shuffle=False)

    ft_history = fine_tune(model, biz_train_loader, biz_val_loader, epochs=10, lr=0.0001, patience=5)
    plot_curves(ft_history, filename="finetune_curves.png")

    print("\n===== 全部完成 =====")
