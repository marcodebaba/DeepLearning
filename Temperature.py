import datetime

import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.pyplot import MultipleLocator
from sklearn import preprocessing

# 读取数据
features = pd.read_csv('./temps.csv')

# 备份 `month, day, year`
years = features['year'].copy()
months = features['month'].copy()
days = features['day'].copy()

# 目标值
labels = np.array(features['actual'])
features = features.drop('actual', axis=1)
feature_list = list(features.columns)

# one-hot 编码
features = pd.get_dummies(features)
print(features.head(5))
print('数据维度:', features.shape)

# 数据标准化
input_features = preprocessing.StandardScaler().fit_transform(features)

# 模型参数
input_size = input_features.shape[1]
batch_size = 16

# PyTorch 神经网络
TemperatureModule = torch.nn.Sequential(
    torch.nn.Linear(input_size, 128),
    torch.nn.Sigmoid(),
    torch.nn.Linear(128, 1),
)

# 损失函数 & 优化器
loss_fn = torch.nn.MSELoss(reduction='mean')
optimizer = torch.optim.Adam(TemperatureModule.parameters(), lr=0.001)

# 训练
losses = []
for i in range(3000):
    batch_loss = []
    for start in range(0, len(input_features), batch_size):
        end = min(start + batch_size, len(input_features))
        xx = torch.tensor(input_features[start:end], dtype=torch.float)
        yy = torch.tensor(labels[start:end], dtype=torch.float).unsqueeze(1)

        loss = loss_fn(TemperatureModule(xx), yy)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        batch_loss.append(loss.item())
    losses.append(np.mean(batch_loss))
    if i % 100 == 0:
        print(f"Epoch {i}, Loss: {losses[-1]:.4f}")

# 预测
x = torch.tensor(input_features, dtype=torch.float)
with torch.no_grad():
    predict = TemperatureModule(x).numpy()

# 获取特征矩阵中 'month' 特征的列
months = features.iloc[:, feature_list.index('month')]
# 获取特征矩阵中 'day' 特征的列
days = features.iloc[:, feature_list.index('day')]
# 获取特征矩阵中 'year' 特征的列
years = features.iloc[:, feature_list.index('year')]

# 遍历年份、月份和日期，将它们组合成日期字符串格式
dates = [str(int(year)) + '-' + str(int(month)) + '-' + str(int(day)) for year,
month, day in zip(years, months, days)]
# 遍历日期字符串列表，将它们解析为 datetime 对象
dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in dates]

# 创建一个包含日期和实际标签的 DataFrame
true_data = pd.DataFrame(data={'date': dates, 'actual': labels})

# 遍历年份、月份和日期，将它们组合成测试日期的字符串格式
test_dates = [str(int(year)) + '-' + str(int(month)) + '-' + str(int(day)) for
              year, month, day in zip(years, months, days)]
# 遍历测试日期字符串列表，将它们解析为 datetime 对象
test_dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in
              test_dates]

# 创建一个包含测试日期和预测值的 DataFrame
predictions_data = pd.DataFrame(data={'date': test_dates, 'prediction':
    predict.reshape(-1)})

print(true_data.head())  # 查看真实数据
print(predictions_data.head())  # 查看预测数据

# 设置matplotlib 的字体为SimHei
matplotlib.rc("font", family='SimHei')

# 创建一个图形，大小为 12×7 英寸，分辨率为 160 dpi
plt.figure(figsize=(16, 7), dpi=160)

# 绘制真实值的曲线，颜色为蓝色，标记为+
plt.plot(true_data['date'], true_data['actual'], 'b+', label='real')

# 绘制预测值的曲线，颜色为红色，标记为 o，并添加标签
plt.plot(predictions_data['date'], predictions_data['prediction'], 'r+',
         label='prediction', marker='o')

plt.autoscale()

# 设置 x 轴的主要刻度间隔为 3
x_major_locator = MultipleLocator(3)

# 设置 y 轴的主要刻度间隔为 5
y_major_locator = MultipleLocator(5)

# 获取当前图形的坐标轴
ax = plt.gca()

# 设置 x 轴的主要刻度定位器
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))  # 每月一个刻度
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))  # 格式化日期

# 设置 y 轴的主要刻度定位器
ax.yaxis.set_major_locator(y_major_locator)

# 添加图例，字体大小为 15
plt.legend(fontsize=15)

# 设置 y 轴标签为'日最高温度'，字体大小为 15
plt.ylabel('日最高温度', size=15)

plt.xticks(rotation=45, ha='right', size=12)  # 旋转 + 靠右对齐

# 显示图形
plt.show()
