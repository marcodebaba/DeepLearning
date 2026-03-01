import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

# 1. 数据预处理（CIFAR-10 训练的归一化方式）
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # CIFAR-10 图片尺寸
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # CIFAR-10 归一化
])

# 2. CIFAR-10 类别列表
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

# 3. 重新定义 CIFARClassifier 结构（必须和训练时一致）
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

# 4. 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 5. 实例化模型并加载 CIFAR-10 训练的权重
model = CIFARClassifier().to(device)
model.load_state_dict(torch.load("cifar10_cnn.pth", map_location=device))  # 加载训练好的模型
model.eval()  # 设为评估模式
print("CIFAR-10 模型已加载！")

# 6. 预处理输入图像
def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')  # 确保是 RGB 图片
    image = transform(image).unsqueeze(0).to(device)  # 预处理
    return image

# 7. 进行预测
def predict_image(model, image_path):
    image = preprocess_image(image_path)
    with torch.no_grad():
        output = model(image)  # 前向传播
        _, predicted = torch.max(output, 1)  # 获取最高得分的类别
    print(f'预测结果: {classes[predicted.item()]}')

# 8. 预测自定义图片
image_path = "../CIFAR10_images/frog.jpg"
predict_image(model, image_path)
