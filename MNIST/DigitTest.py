import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

# 1. 预处理（必须与训练时相同）
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 2. 重新定义模型（必须与训练时一致）
class DigitClassifier(nn.Module):
    def __init__(self):
        super(DigitClassifier, self).__init__()
        self.linear1 = nn.Linear(28 * 28, 120)
        self.linear2 = nn.Linear(120, 64)
        self.linear3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        x = self.linear3(x)
        return x

# 3. 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 4. 加载模型
model = DigitClassifier().to(device)
model.load_state_dict(torch.load("MNIST_classifier.pth", map_location=device))  # 确保路径正确
model.eval()
print("MNIST 模型已加载！")

# 5. 预处理输入图像
def preprocess_image(image_path):
    image = Image.open(image_path).convert('L')  # 确保是灰度图
    image = transform(image).unsqueeze(0).to(device)  # 预处理
    return image

# 6. 进行预测
def predict_image(model, image_path):
    image = preprocess_image(image_path)
    with torch.no_grad():
        output = model(image)  # 前向传播
        _, predicted = torch.max(output, 1)  # 获取最高得分的类别
    print(f'识别结果: {predicted.item()}')

# 7. 预测自定义手写数字
image_path = "../mnist_images/1_14.png"  # 确保路径正确
predict_image(model, image_path)
