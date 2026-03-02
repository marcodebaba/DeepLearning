import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image

# 1. 定义数据预处理（与训练时保持一致）
transform = transforms.Compose([
    transforms.Resize((28, 28)),  # 确保输入尺寸为 28x28
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 2. 重新定义 CNN 结构（必须和训练时一致）
# 2. 定义 CNN 模型
class CNNDigitClassifier(nn.Module):
    def __init__(self, input_size=(1, 1, 28, 28)):  # 修正为四维形状 (batch_size, channels, height, width)
        super(CNNDigitClassifier, self).__init__()
        # MNIST数据集是灰度图像（单通道），32个3*3的卷积核
        # conv1的输入尺寸为28x28x1，conv1的输出尺寸为26x26x32
        self.conv1 = nn.Conv2d(1, 32, 3)
        # conv2的输入通道数是conv1的输出通道数。输出通道数是一个超参数，通常选择大于前一层的输出通道数，以增加模型容量
        # conv2的输入尺寸为13x13x32，conv2的输出尺寸为11x11x64
        self.conv2 = nn.Conv2d(32, 64, 3)

        # 动态计算卷积和池化后的输出尺寸
        with torch.no_grad():
            x = torch.zeros(input_size).to(device)  # 创建四维张量
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

# 3. 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 4. 实例化模型并加载参数
model = CNNDigitClassifier().to(device)
model.load_state_dict(torch.load("mnist_cnn.pth", map_location=device, weights_only=True))  # 加载训练好的模型
model.eval()  # 设为评估模式
print("MNIST 模型已加载！")

# 5. 预处理输入图像
def preprocess_image(image_path):
    image = Image.open(image_path).convert('L')  # 确保是灰度图
    image = transform(image).unsqueeze(0).to(device)  # 预处理
    return image

# 6. 进行预测
def predict_digit(image_path, model):
    model.eval()  # 进入评估模式
    image = preprocess_image(image_path)  # 预处理图片
    with torch.no_grad():
        output = model(image)  # 前向传播
        _, predicted = torch.max(output, 1)  # 获取最高得分的类别
    print(f'预测结果: {predicted.item()}')
    return predicted.item()

# 7. 预测手写数字
image_path = "../mnist_images/8_31.png"  # 替换成你的图片路径
predict_digit(image_path, model)
