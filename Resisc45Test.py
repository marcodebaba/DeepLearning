import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import torch.nn.functional as F

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class_names = [
    "airplane", "airport", "baseball_diamond", "basketball_court", "beach",
    "bridge", "chaparral", "church", "circular_farmland", "cloud",
    "commercial_area", "dense_residential", "desert", "forest", "freeway",
    "golf_course", "ground_track_field", "harbor", "industrial_area", "intersection",
    "island", "lake", "meadow", "medium_residential", "mobile_home_park", "mountain",
    "overpass", "palace", "parking_lot", "railway", "railway_station", "rectangular_farmland",
    "river", "roundabout", "runway", "sea_ice", "ship", "snowberg", "sparse_residential",
    "stadium", "storage_tank", "tennis_court", "terrace", "thermal_power_station",
    "wetland"
]

# 定义与训练时相同的 CNN 模型
class NWPU_CNN(nn.Module):
    def __init__(self, num_classes=45):
        super(NWPU_CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self._calculate_fc_input_size()

        self.fc1 = nn.Linear(self.fc_input_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()

    def _calculate_fc_input_size(self):
        with torch.no_grad():
            x = torch.randn(1, 3, 128, 128)
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = self.pool(F.relu(self.conv3(x)))
            self.fc_input_size = x.view(1, -1).size(1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# 加载模型
model = NWPU_CNN().to(device)
model.load_state_dict(torch.load("nwpu_resisc45_cnn.pth", map_location=device))
model.eval()

# 预处理步骤
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# 读取图片并进行预测
def predict_image(image_path, model):
    image = Image.open(image_path).convert("RGB")
    image = transform(image)
    image = image.unsqueeze(0).to(device)  # 增加 batch 维度

    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)

    return predicted.item()


# 测试自己的图片
image_path = "./resisc_images/church.jpg"  # 替换为自己的图片路径
# 预测并输出类别名称
predicted_class = predict_image(image_path, model)
predicted_label = class_names[predicted_class]  # 获取类别名称

print(f"预测类别: {predicted_class} - {predicted_label}")
