import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader

# 设备配置（移至顶部，确保类定义时已可使用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用的设备: {device}")

# 1. 数据预处理
transform = transforms.Compose([
    transforms.Resize((28, 28)),  # 确保输入尺寸为 28x28
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 2. 定义 CNN 模型
class CNNDigitClassifier(nn.Module):
    def __init__(self, input_size=(1, 1, 28, 28)):  # 修正为四维形状 (batch_size, channels, height, width)
        super(CNNDigitClassifier, self).__init__()
        # MNIST数据集是灰度图像（单通道），32个3*3的卷积核
        # conv1的输入尺寸为1x28x28，经过卷积后conv1的输出尺寸为26x26x32
        self.conv1 = nn.Conv2d(1, 32, 3)
        # conv2的输入通道数是conv1的输出通道数。输出通道数是一个超参数，通常选择大于前一层的输出通道数，以增加模型容量
        # conv2的输入尺寸为13x13x32，经过卷积后conv2的输出尺寸为11x11x64
        self.conv2 = nn.Conv2d(32, 64, 3)

        # 动态计算卷积和池化后的输出尺寸
        with torch.no_grad():
            x = torch.zeros(input_size)  # 创建四维张量，保持在 CPU（此时模型权重尚未移至 device）
            x = F.max_pool2d(F.relu(self.conv1(x)), 2)
            x = F.max_pool2d(F.relu(self.conv2(x)), 2)
            _, _, h, w = x.shape  # 确保解包为四个值 (batch_size, channels, height, width)
            n_features = 64 * h * w  # 特征数

        # conv2池化后输出尺寸为5x5x64，全连接层1的输出维度为128
        self.fc1 = nn.Linear(n_features, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # 输入：26x26x32，池化后输出尺寸为13x13x32
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        # 输入：11x11x64，池化后输出尺寸为5x5x64
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(x.size(0), -1)  # 展平
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 3. 训练模型
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

    train_loss /= len(train_loader)
    train_acc = correct / len(train_loader.dataset)
    return train_loss, train_acc

# 4. 评估模型（修复：添加 loss_fn 参数，不再依赖全局变量）
def evaluate_model(model, test_loader, loss_fn):
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, Y in test_loader:
            X, Y = X.to(device), Y.to(device)
            output = model(X)
            test_loss += loss_fn(output, Y).item()
            correct += (output.argmax(1) == Y).type(torch.float).sum().item()

    test_loss /= len(test_loader)
    test_acc = correct / len(test_loader.dataset)
    return test_loss, test_acc

# 5. 预测自己手写的数字（修复：添加 transform 参数，不再依赖全局变量）
def predict_digit(image_path, model, transform):
    model.eval()
    image = Image.open(image_path).convert('L')  # 转换为灰度图
    image = transform(image).unsqueeze(0).to(device)  # 预处理
    with torch.no_grad():
        output = model(image)
        predicted = output.argmax(1).item()
    print(f'预测结果: {predicted}')
    return predicted

if __name__ == '__main__':
    # 加载 MNIST 数据集
    train_dataset = torchvision.datasets.MNIST(
        root='/kaggle/working/data', train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.MNIST(
        root='/kaggle/working/data', train=False, download=True, transform=transform
    )

    # 定义数据加载器
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # 实例化模型
    model = CNNDigitClassifier().to(device)

    # 定义损失函数和优化器
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    # 训练和测试
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(10):
        train_loss, train_acc = train_model(model, train_loader, loss_fn, optimizer)
        test_loss, test_acc = evaluate_model(model, test_loader, loss_fn)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(test_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(test_acc)
        print(f'Epoch {epoch + 1} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc * 100:.2f}% | '
              f'Test Loss: {test_loss:.4f} | Test Acc: {test_acc * 100:.2f}%')

    # 保存模型参数
    torch.save(model.state_dict(), "mnist_cnn.pth")
    print("模型参数已保存！")

    # 画出 loss 和 accuracy 曲线
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
    plt.savefig("training_curves.png")
    print("训练曲线已保存至 training_curves.png")

    # 使用训练好的模型对自己的手写数字进行预测
    #image_path = "../mnist_images/9_19.png"
    #predict_digit(image_path, model, transform)
