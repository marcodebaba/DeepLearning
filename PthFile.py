import torch

# 加载 .pth 文件
pth_path = "./CNN/mnist_cnn.pth"  # 确保路径正确
state_dict = torch.load(pth_path, map_location="cpu")

# 打印内容
print("📌 mnist_cnn.pth 文件内容：")
for key, value in state_dict.items():
    print(f"🔹 {key} -> Shape: {value.shape}")
