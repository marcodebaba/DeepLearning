import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# 1. 数据预处理与加载
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),  # 转换为张量
    transforms.Normalize((0.5,), (0.5,))  # 标准化到[-1, 1]
])

# 加载 MNIST 数据集
train_dataset = torchvision.datasets.MNIST(
    root='../data', train=True, download=True, transform=transform
)
test_dataset = torchvision.datasets.MNIST(
    root='../data', train=False, download=True, transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 2. 定义模型
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

model = DigitClassifier()

# 3. 定义损失函数和优化器
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

# 4. 训练模型
def train_model(model, train_loader, loss_fn, optimizer):
    model.train()
    train_loss, correct = 0, 0
    for X, Y in train_loader:
        optimizer.zero_grad()
        output = model(X)
        loss = loss_fn(output, Y)
        loss.backward()
        optimizer.step()
        correct += (output.argmax(1) == Y).type(torch.float).sum().item()
        train_loss += loss.item()
    return train_loss / len(train_loader), correct / len(train_loader.dataset)

# 5. 评估模型
def evaluate_model(model, test_loader):
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, Y in test_loader:
            output = model(X)
            test_loss += loss_fn(output, Y).item()
            correct += (output.argmax(1) == Y).type(torch.float).sum().item()
    return test_loss / len(test_loader), correct / len(test_loader.dataset)

# 训练和测试
epochs = 15
train_loss, train_acc, test_loss, test_acc = [], [], [], []
for epoch in range(epochs):
    epoch_loss, epoch_acc = train_model(model, train_loader, loss_fn, optimizer)
    epoch_test_loss, epoch_test_acc = evaluate_model(model, test_loader)
    train_loss.append(epoch_loss)
    train_acc.append(epoch_acc)
    test_loss.append(epoch_test_loss)
    test_acc.append(epoch_test_acc)
    print(f'Epoch {epoch + 1} | Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc * 100:.2f}% | Test Loss: {epoch_test_loss:.4f} | Test Acc: {epoch_test_acc * 100:.2f}%')

print("训练结束!")

# 保存模型参数到文件
torch.save(model.state_dict(), '../MNIST/MNIST_classifier.pth')
print("模型参数已保存到 'MNIST_classifier.pth'")

# 可视化损失和正确率
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(range(1, epochs + 1), train_loss, label='Train Loss')
plt.plot(range(1, epochs + 1), test_loss, label='Test Loss', linestyle='--')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss Curve')
plt.subplot(1, 2, 2)
plt.plot(range(1, epochs + 1), train_acc, label='Train Accuracy')
plt.plot(range(1, epochs + 1), test_acc, label='Test Accuracy', linestyle='--')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy Curve')
plt.tight_layout()
plt.show()
