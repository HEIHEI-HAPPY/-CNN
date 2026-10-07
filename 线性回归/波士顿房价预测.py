# 随机梯度下降 - 波士顿房价预测

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 1. 加载数据集
# ============================================================
# 波士顿房价数据集在 sklearn 1.2+ 已移除，使用原数据源加载
import pandas as pd
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep=r"\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]

feature_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE',
                 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
X = data
y = target

print("=" * 50)
print("波士顿房价数据集")
print("=" * 50)
print(f"样本数: {X.shape[0]}")
print(f"特征数: {X.shape[1]}")
print(f"特征名: {feature_names}")
print(f"房价范围: {y.min():.1f} ~ {y.max():.1f} 万美元")
print(f"房价均值: {y.mean():.2f} 万美元")

# ============================================================
# 2. 数据预处理
# ============================================================
# 划分训练集和测试集（80%训练，20%测试）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 标准化（梯度下降对特征尺度敏感，必须标准化）
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\n训练集: {X_train_scaled.shape[0]} 样本")
print(f"测试集: {X_test_scaled.shape[0]} 样本")

# ============================================================
# 3. 随机梯度下降实现线性回归
# ============================================================
class SGDLinearRegression:
    """随机梯度下降线性回归"""

    def __init__(self, learning_rate=0.01, n_epochs=100, batch_size=32, random_state=42):
        self.lr = learning_rate          # 学习率
        self.n_epochs = n_epochs         # 迭代轮数
        self.batch_size = batch_size     # 批大小
        self.random_state = random_state
        self.w = None                    # 权重
        self.b = None                    # 偏置
        self.losses = []                 # 记录每轮损失

    def fit(self, X, y):
        n_samples, n_features = X.shape
        np.random.seed(self.random_state)

        # 初始化参数
        self.w = np.zeros(n_features)
        self.b = 0.0

        for epoch in range(self.n_epochs):
            # 每轮打乱数据顺序（SGD的核心）
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            epoch_loss = 0.0
            n_batches = 0

            # 按小批量训练
            for start in range(0, n_samples, self.batch_size):
                end = start + self.batch_size
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                # 前向传播：预测
                y_pred = X_batch.dot(self.w) + self.b

                # 计算梯度（MSE的梯度）
                error = y_pred - y_batch
                dw = (2 / len(y_batch)) * X_batch.T.dot(error)
                db = (2 / len(y_batch)) * np.sum(error)

                # 更新参数
                self.w -= self.lr * dw
                self.b -= self.lr * db

                # 累加损失
                epoch_loss += np.mean(error ** 2)
                n_batches += 1

            # 记录本轮平均损失
            self.losses.append(epoch_loss / n_batches)

        return self

    def predict(self, X):
        return X.dot(self.w) + self.b


# ============================================================
# 4. 训练模型
# ============================================================
model = SGDLinearRegression(
    learning_rate=0.01,
    n_epochs=200,
    batch_size=32
)
model.fit(X_train_scaled, y_train)

# 预测
y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)

# ============================================================
# 5. 评估指标
# ============================================================
print("\n" + "=" * 50)
print("模型评估结果")
print("=" * 50)

train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
train_rmse = np.sqrt(train_mse)
test_rmse = np.sqrt(test_mse)
train_mae = mean_absolute_error(y_train, y_train_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

print(f"{'指标':<15} {'训练集':<15} {'测试集':<15}")
print("-" * 45)
print(f"{'MSE':<15} {train_mse:<15.4f} {test_mse:<15.4f}")
print(f"{'RMSE':<15} {train_rmse:<15.4f} {test_rmse:<15.4f}")
print(f"{'MAE':<15} {train_mae:<15.4f} {test_mae:<15.4f}")
print(f"{'R²':<15} {train_r2:<15.4f} {test_r2:<15.4f}")

# ============================================================
# 6. 特征权重分析
# ============================================================
print("\n" + "=" * 50)
print("特征权重（绝对值排序）")
print("=" * 50)
weights = sorted(zip(feature_names, model.w), key=lambda x: abs(x[1]), reverse=True)
for name, w in weights:
    direction = "↑ 正相关" if w > 0 else "↓ 负相关"
    print(f"  {name:<10} {w:>8.4f}   {direction}")

# ============================================================
# 7. 可视化
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 图1：损失函数下降曲线
ax1 = axes[0, 0]
ax1.plot(model.losses, color='blue', linewidth=1.5)
ax1.set_title('SGD 损失函数下降曲线', fontsize=14)
ax1.set_xlabel('迭代轮数 (Epoch)')
ax1.set_ylabel('MSE Loss')
ax1.grid(True, alpha=0.3)

# 图2：预测值 vs 真实值（测试集）
ax2 = axes[0, 1]
ax2.scatter(y_test, y_test_pred, alpha=0.7, color='steelblue', edgecolors='k', s=50)
min_val = min(y_test.min(), y_test_pred.min())
max_val = max(y_test.max(), y_test_pred.max())
ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='完美预测线')
ax2.set_title(f'预测值 vs 真实值（R²={test_r2:.4f}）', fontsize=14)
ax2.set_xlabel('真实房价（万美元）')
ax2.set_ylabel('预测房价（万美元）')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 图3：残差分布
ax3 = axes[1, 0]
residuals = y_test - y_test_pred
ax3.hist(residuals, bins=20, color='steelblue', edgecolor='k', alpha=0.7)
ax3.axvline(x=0, color='red', linestyle='--', linewidth=2)
ax3.set_title('残差分布（测试集）', fontsize=14)
ax3.set_xlabel('残差（真实值 - 预测值）')
ax3.set_ylabel('频数')
ax3.grid(True, alpha=0.3)

# 图4：特征权重
ax4 = axes[1, 1]
names = [w[0] for w in weights]
values = [w[1] for w in weights]
colors = ['green' if v > 0 else 'red' for v in values]
ax4.barh(names, values, color=colors, edgecolor='k', alpha=0.7)
ax4.set_title('特征权重', fontsize=14)
ax4.set_xlabel('权重值')
ax4.axvline(x=0, color='black', linewidth=0.8)
ax4.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig(r'D:\机器学习\线性回归\波士顿房价预测结果.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# 8. 梯度下降公式说明
# ============================================================
print("\n" + "=" * 50)
print("随机梯度下降（SGD）公式")
print("=" * 50)
print("""
  预测公式:  ŷ = X·w + b
  损失函数:  MSE = (1/n) × Σ(y - ŷ)²
  梯度计算:
      ∂Loss/∂w = (2/n) × Xᵀ·(ŷ - y)
      ∂Loss/∂b = (2/n) × Σ(ŷ - y)
  参数更新:
      w = w - α × ∂Loss/∂w
      b = b - α × ∂Loss/∂b

  SGD 与批量梯度下降的区别:
  - 批量梯度下降(BGD): 每次用全部样本计算梯度，稳定但慢
  - 随机梯度下降(SGD): 每次用1个样本，快但震荡
  - 小批量SGD: 每次用batch_size个样本，兼顾速度和稳定性
""")
