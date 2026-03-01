import torchvision
import os

os.makedirs("test_images", exist_ok=True)

dataset = torchvision.datasets.CIFAR10(root="../data", train=False, download=False)
class_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

saved = {}
for img, label in dataset:
    name = class_names[label]
    if name not in saved:
        img.save(f"test_images/{name}.png")
        saved[name] = True
    if len(saved) == 10:
        break

print("已保存：", os.listdir("test_images"))
