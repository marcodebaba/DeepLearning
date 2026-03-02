import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # noqa: E402

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.optim as optim  # noqa: E402
import torchvision  # noqa: E402
import torchvision.transforms as transforms  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from CNNDigitClassifier import CNNDigitClassifier, train_model, evaluate_model  # noqa: E402
from AdjustCNN import freeze_conv_layers, fine_tune, plot_curves  # noqa: E402
from BusinessDataset import BusinessDigitDataset  # noqa: E402

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用的设备: {device}")

transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# ================================================================
# Step 1: 预训练 CNNDigitClassifier（MNIST 原始数据）
# ================================================================
print("\n===== Step 1: 预训练阶段 =====")

train_dataset = torchvision.datasets.MNIST(
    root="/kaggle/working/data", train=True, download=True, transform=transform
)
test_dataset = torchvision.datasets.MNIST(
    root="/kaggle/working/data", train=False, download=True, transform=transform
)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

model = CNNDigitClassifier().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

best_val_loss = float("inf")
patience, patience_counter = 5, 0

for epoch in range(20):
    train_loss, train_acc = train_model(model, train_loader, loss_fn, optimizer)
    test_loss, test_acc = evaluate_model(model, test_loader, loss_fn)
    print(f"Epoch {epoch + 1} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc * 100:.2f}% | "
          f"Val Loss: {test_loss:.4f} | Val Acc: {test_acc * 100:.2f}%")

    if test_loss < best_val_loss:
        best_val_loss = test_loss
        torch.save(model.state_dict(), "mnist_cnn.pth")
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

print("预训练完成，权重已保存至 mnist_cnn.pth")

# ================================================================
# Step 2: 微调（业务数据）
# ================================================================
print("\n===== Step 2: 微调阶段 =====")

model.load_state_dict(torch.load("mnist_cnn.pth", map_location=device))
freeze_conv_layers(model)

biz_train = BusinessDigitDataset(train=True)
biz_val = BusinessDigitDataset(train=False)
biz_train_loader = DataLoader(biz_train, batch_size=64, shuffle=True)
biz_val_loader = DataLoader(biz_val, batch_size=64, shuffle=False)

history = fine_tune(model, biz_train_loader, biz_val_loader, epochs=10, lr=0.0001, patience=5)
plot_curves(history)

print("\n===== 全部完成 =====")
