import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # CIFAR-10 图片大小
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化
])

# CIFAR-10 类别
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

# 定义 CIFARClassifier 模型
class CIFARClassifier(nn.Module):
    def __init__(self, input_size=(1, 3, 32, 32)):
        super(CIFARClassifier, self).__init__()

        # 定义 ReLU 激活函数，确保在整个初始化中可用
        self.relu = nn.ReLU()

        # CIFAR-10数据集是RGB彩色图像，因此输入图像的深度（通道数）为3
        # 输入图像尺寸为32x32x3，conv1的输出尺寸为30x30x64
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3)
        # 输入尺寸为15x15x64（conv1后池化的输出），conv2的输出尺寸为13x13x128
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
        # 输入尺寸为6x6x128（conv2后池化的输出），conv3的输出尺寸为4x4x256
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # 动态计算卷积和池化后的输出尺寸
        with torch.no_grad():
            x = torch.zeros(input_size).to(device)
            x = self.pool(self.relu(self.conv1(x)))
            x = self.pool(self.relu(self.conv2(x)))
            x = self.pool(self.relu(self.conv3(x)))
            _, _, h, w = x.shape  # 获取输出高度和宽度
            n_features = 256 * h * w  # 特征数

        print(f'n_features: {n_features}')
        # 全连接层1的输入是2x2x256，输出维度为512
        self.fc1 = nn.Linear(n_features, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        # 输入：30x30x64，池化后输出尺寸为15x15x64
        x = self.pool(self.relu(self.conv1(x)))
        # 输入：13x13x128，池化后输出尺寸为6x6x128
        x = self.pool(self.relu(self.conv2(x)))
        # 输入：4x4x256，池化后输出尺寸为2x2x256
        x = self.pool(self.relu(self.conv3(x)))

        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

if __name__ == "__main__":
    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CIFARClassifier().to(device)

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 加载 CIFAR-10 数据集
    trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=0)

    testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=0)

    # 训练模型
    epochs = 10
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch {epoch + 1}, Loss: {running_loss / len(trainloader):.4f}')

    print("Training Complete!")

    # 保存模型
    torch.save(model.state_dict(), "./cifar10_cnn.pth")
    print("模型已保存为 cifar10_cnn.pth")
