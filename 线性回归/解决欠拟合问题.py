# 欠拟合演示：用线性模型拟合二次数据

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def dm01_under_fitting():
    # 随机种子
    np.random.seed(42)
    x = np.random.uniform(-3, 3, 100)
    y = 0.5 * x**2 + x + np.random.normal(0, 1, 100)

    # 数据预处理
    x = x.reshape(-1, 1)
    x_poly = np.hstack([x, x**2,])  # 增加二次特征

    # 划分训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(
        x_poly, y, test_size=0.2, random_state=42
    )

    # 模型训练（线性模型拟合二次数据 → 欠拟合）
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

    # 可视化
    # 生成平滑曲线的x值和对应的多项式特征
    x_line = np.linspace(x_train[:, 0].min(), x_train[:, 0].max(), 200).reshape(-1, 1)
    x_line_poly = np.hstack([x_line, x_line**2])
    y_line = estimator.predict(x_line_poly)

    plt.figure(figsize=(10, 6))
    plt.scatter(x_train[:, 0], y_train, color='steelblue', label='训练集', alpha=0.7, edgecolors='k', s=50)
    plt.scatter(x_test[:, 0], y_test, color='orange', label='测试集', alpha=0.7, edgecolors='k', s=50)
    plt.plot(x_line, y_line, color='red', linewidth=2, label='拟合曲线')
    plt.title('欠拟合示例：线性模型拟合二次数据', fontsize=14)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
   # plt.savefig(r'D:\机器学习\线性回归\欠拟合.png', dpi=150, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    dm01_under_fitting()
