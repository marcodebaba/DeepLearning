import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 二维变量函数 y = 2 * (x1^2 + x2^2)
def f(x1, x2):
    return 2 * (x1**2 + x2**2)

# 当前点
x0 = np.array([1.0, 2.0])
y0 = f(*x0)

# 梯度
grad = 4 * x0

# 绘制网格数据
x1_vals = np.linspace(-3, 3, 100)
x2_vals = np.linspace(-3, 3, 100)
X1, X2 = np.meshgrid(x1_vals, x2_vals)
Y = f(X1, X2)

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')

# 曲面
ax.plot_surface(X1, X2, Y, alpha=0.6, cmap='viridis')

# 当前点
ax.scatter(x0[0], x0[1], y0, color='r', s=50, label=f'当前点 y={y0:.1f}')

# 梯度方向：在三维中用箭头表示
# z 方向变化率通过梯度与切平面的关系可理解
ax.quiver(x0[0], x0[1], y0, grad[0], grad[1], 0, 
          color='r', length=1.0, normalize=True, label='梯度 ∇y')

ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('y')
ax.set_title('三维图：y=2*(x1^2+x2^2) 与梯度关系')
ax.legend()
plt.show()
