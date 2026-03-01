import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn


# 预处理开始
data = pd.read_csv('data/Income1.csv')

educations = torch.from_numpy(data.Education.values.reshape(-1, 1)).type(torch.FloatTensor)
incomes = torch.from_numpy(data.Income.values.reshape(-1, 1)).type(torch.FloatTensor)
#print(educations.size(), incomes.size())
# 预处理结束

# 创建模型开始
class EducationIncomeModel(nn.Module):
    def __init__(self):  # 初始化方法
        # 继承父类的属性
        super(EducationIncomeModel, self).__init__()
        # 创建线性连接层
        self.linear = nn.Linear(1, 1)

    def forward(self, inputs):
        return self.linear(inputs)

# 实例化模型
model = EducationIncomeModel()

# 回归问题使用均方误差损失函数
loss_fn = nn.MSELoss()

# 初始化优化器
opt = torch.optim.SGD(model.parameters(), lr=0.0001)

# 对全部的数据训练5000次
for epoch in range(5000):
    for education, income in zip(educations, incomes):
        # 调用model得到预测输出predict
        predict = model(education)
        # 根据模型预测输出与实际的值y计算损失
        loss = loss_fn(predict, income)
        # 将累计的梯度置为0
        opt.zero_grad()
        # 反向传播损失，计算损失与模型参数之间的梯度
        loss.backward()
        # 根据计算得到梯度优化模型参数
        opt.step()
print("训练结束!")

# 以生成器的形式返回模型参数的名称和值
print(list(model.named_parameters()))

# 绘制原数据分布的散点图
plt.scatter(data.Education, data.Income, label='real data')
# 用我们训练出来的参数，来绘制直线
plt.plot(educations, model(educations).detach().numpy(), c='r', label='predict line')
# 设置x轴的标签
plt.xlabel('Education')
# 设置y轴的标签
plt.ylabel('Income')
# 可以显示图例
plt.legend()
plt.show()
