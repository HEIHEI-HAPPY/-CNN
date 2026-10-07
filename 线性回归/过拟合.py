# 过拟合演示：高次多项式拟合二次数据

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def dm01_over_fitting():
    # 随机种子
    np.random.seed(42)
    x = np.random.uniform(-3, 3, 100)
    y = 0.5 * x**2 + x + np.random.normal(0, 1, 100)

    # 数据预处理：构造高次多项式特征（9次）
    x_1d = x.copy()
    x_poly = x_1d.reshape(-1, 1)
    for i in range(2, 10):
        x_poly = np.hstack([x_poly, x_1d.reshape(-1, 1) ** i])

    # 划分训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(
        x_poly, y, test_size=0.2, random_state=42
    )

    # 模型训练（高次多项式 → 过拟合）
    estimator = LinearRegression()
    estimator.fit(x_train, y_train)

    # 预测
    y_train_pred = estimator.predict(x_train)
    y_test_pred = estimator.predict(x_test)

    # 评估
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    print(f"训练集均方误差: {train_mse:.4f}")
    print(f"测试集均方误差: {test_mse:.4f}")
    print(f"过拟合程度（测试MSE / 训练MSE）: {test_mse / train_mse:.2f}倍")

    # 可视化
    # 提取原始 x 用于画图
    x_train_1d = x_train[:, 0]
    x_test_1d = x_test[:, 0]

    # 生成平滑曲线
    x_line = np.linspace(x_1d.min(), x_1d.max(), 300)
    x_line_poly = x_line.reshape(-1, 1)
    for i in range(2, 10):
        x_line_poly = np.hstack([x_line_poly, x_line.reshape(-1, 1) ** i])
    y_line = estimator.predict(x_line_poly)

    plt.figure(figsize=(10, 6))
    plt.scatter(x_train_1d, y_train, color='steelblue', label='训练集', alpha=0.7, edgecolors='k', s=50)
    plt.scatter(x_test_1d, y_test, color='orange', label='测试集', alpha=0.7, edgecolors='k', s=50)
    plt.plot(x_line, y_line, color='red', linewidth=2, label='9次多项式拟合（过拟合）')
    plt.title('过拟合示例：高次多项式拟合二次数据', fontsize=14)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(r'D:\机器学习\线性回归\过拟合.png', dpi=150, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    dm01_over_fitting()
