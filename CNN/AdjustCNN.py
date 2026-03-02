import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from CNNDigitClassifier import CNNDigitClassifier
from BusinessDataset import BusinessDigitDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用的设备: {device}")


def freeze_conv_layers(model):
    """冻结卷积层，只训练全连接层"""
    for param in model.conv1.parameters():
        param.requires_grad = False
    for param in model.conv2.parameters():
        param.requires_grad = False
    print("卷积层已冻结，只训练全连接层")


def train_epoch(model, loader, loss_fn, optimizer):
    model.train()
    total_loss, correct = 0, 0
    for X, Y in loader:
        X, Y = X.to(device), Y.to(device)
        optimizer.zero_grad()
        output = model(X)
        loss = loss_fn(output, Y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (output.argmax(1) == Y).type(torch.float).sum().item()
    return total_loss / len(loader), correct / len(loader.dataset)


def evaluate_epoch(model, loader, loss_fn):
    model.eval()
    total_loss, correct = 0, 0
    with torch.no_grad():
        for X, Y in loader:
            X, Y = X.to(device), Y.to(device)
            output = model(X)
            total_loss += loss_fn(output, Y).item()
            correct += (output.argmax(1) == Y).type(torch.float).sum().item()
    return total_loss / len(loader), correct / len(loader.dataset)


def fine_tune(model, train_loader, val_loader, epochs=10, lr=0.0001, patience=5):
    """
    微调入口函数
    - lr 比预训练小 10 倍，避免破坏已有权重
    - Early Stopping 监控 val_loss
    - 保存最优模型到 fine_tuned_cnn.pth
    """
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr
    )

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, loss_fn, optimizer)
        val_loss, val_acc = evaluate_epoch(model, val_loader, loss_fn)

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


def plot_curves(history):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, history["train_loss"], label="Train Loss")
    ax1.plot(epochs, history["val_loss"], label="Val Loss")
    ax1.set_title("Fine-tuning Loss Curve")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()

    ax2.plot(epochs, history["train_acc"], label="Train Acc")
    ax2.plot(epochs, history["val_acc"], label="Val Acc")
    ax2.set_title("Fine-tuning Accuracy Curve")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()

    plt.tight_layout()
    plt.savefig("fine_tuning_curves.png")
    print("微调曲线已保存至 fine_tuning_curves.png")


if __name__ == "__main__":
    # 1. 加载预训练模型权重
    model = CNNDigitClassifier().to(device)
    model.load_state_dict(torch.load("mnist_cnn.pth", map_location=device))
    print("预训练权重加载完成")

    # 2. 冻结卷积层
    freeze_conv_layers(model)

    # 3. 加载模拟业务数据集
    train_dataset = BusinessDigitDataset(train=True)
    val_dataset = BusinessDigitDataset(train=False)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # 4. 开始微调
    history = fine_tune(model, train_loader, val_loader, epochs=10, lr=0.0001, patience=5)

    # 5. 画出微调曲线
    plot_curves(history)
