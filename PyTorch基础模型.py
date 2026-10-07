# 实验：基于 PyTorch 的基础模型构建
# 目标：理解 PyTorch 模型构建与训练的基本流程
# 内容：线性回归、逻辑回归、多层前馈神经网络

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

print("=" * 60)
print("基于 PyTorch 的基础模型构建实验")
print("=" * 60)

# ============================================================
# 1. 准备实验数据
# ============================================================
print("\n[1] 准备实验数据")

# 1.1 回归数据（用于线性回归和多层前馈网络）
X_reg, y_reg = make_regression(n_samples=500, n_features=1, noise=15, random_state=42)
y_reg = y_reg.reshape(-1, 1)

# 标准化
scaler_reg = StandardScaler()
X_reg = scaler_reg.fit_transform(X_reg)
scaler_y = StandardScaler()
y_reg = scaler_y.fit_transform(y_reg)

# 划分数据集
X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

# 转为 PyTorch Tensor
X_reg_train_t = torch.FloatTensor(X_reg_train)
y_reg_train_t = torch.FloatTensor(y_reg_train)
X_reg_test_t = torch.FloatTensor(X_reg_test)
y_reg_test_t = torch.FloatTensor(y_reg_test)

print(f"  回归数据: 训练集 {X_reg_train_t.shape}, 测试集 {X_reg_test_t.shape}")

# 1.2 分类数据（用于逻辑回归和多层前馈网络）
X_cls, y_cls = make_classification(
    n_samples=500, n_features=2, n_redundant=0, n_informative=2,
    n_clusters_per_class=1, random_state=42
)

# 标准化
scaler_cls = StandardScaler()
X_cls = scaler_cls.fit_transform(X_cls)

# 划分数据集
X_cls_train, X_cls_test, y_cls_train, y_cls_test = train_test_split(
    X_cls, y_cls, test_size=0.2, random_state=42
)

# 转为 PyTorch Tensor
X_cls_train_t = torch.FloatTensor(X_cls_train)
y_cls_train_t = torch.FloatTensor(y_cls_train).unsqueeze(1)
X_cls_test_t = torch.FloatTensor(X_cls_test)
y_cls_test_t = torch.FloatTensor(y_cls_test).unsqueeze(1)

print(f"  分类数据: 训练集 {X_cls_train_t.shape}, 测试集 {X_cls_test_t.shape}")


# ============================================================
# 2. 模型1：线性回归（nn.Module）
# ============================================================
print("\n" + "=" * 60)
print("[2] 模型1：线性回归")
print("=" * 60)


class LinearRegressionModel(nn.Module):
    """线性回归模型：y = w·x + b"""

    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)


# 训练
model_linear = LinearRegressionModel(input_dim=1)
criterion = nn.MSELoss()
optimizer = optim.SGD(model_linear.parameters(), lr=0.01)

losses_linear = []
n_epochs = 200

for epoch in range(n_epochs):
    # 前向传播
    y_pred = model_linear(X_reg_train_t)
    loss = criterion(y_pred, y_reg_train_t)

    # 反向传播
    optimizer.zero_grad()
    loss.backward()

    # 参数更新
    optimizer.step()

    losses_linear.append(loss.item())

    if (epoch + 1) % 50 == 0:
        print(f"  Epoch {epoch+1:>3d}, Loss: {loss.item():.6f}")

# 评估
model_linear.eval()
with torch.no_grad():
    y_reg_pred = model_linear(X_reg_test_t)
    test_mse = criterion(y_reg_pred, y_reg_test_t).item()

w = model_linear.linear.weight.item()
b = model_linear.linear.bias.item()
print(f"\n  学到的参数: w = {w:.4f}, b = {b:.4f}")
print(f"  测试集 MSE: {test_mse:.6f}")


# ============================================================
# 3. 模型2：逻辑回归（nn.Module）
# ============================================================
print("\n" + "=" * 60)
print("[3] 模型2：逻辑回归")
print("=" * 60)


class LogisticRegressionModel(nn.Module):
    """逻辑回归模型：ŷ = σ(w·x + b)"""

    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.linear(x))


# 训练
model_logistic = LogisticRegressionModel(input_dim=2)
criterion_bce = nn.BCELoss()
optimizer_log = optim.SGD(model_logistic.parameters(), lr=0.1)

losses_logistic = []
n_epochs = 300

for epoch in range(n_epochs):
    # 前向传播
    y_pred = model_logistic(X_cls_train_t)
    loss = criterion_bce(y_pred, y_cls_train_t)

    # 反向传播
    optimizer_log.zero_grad()
    loss.backward()

    # 参数更新
    optimizer_log.step()

    losses_logistic.append(loss.item())

    if (epoch + 1) % 50 == 0:
        print(f"  Epoch {epoch+1:>3d}, Loss: {loss.item():.6f}")

# 评估
model_logistic.eval()
with torch.no_grad():
    y_cls_prob = model_logistic(X_cls_test_t).numpy()
    y_cls_pred = (y_cls_prob >= 0.5).astype(int)

acc = accuracy_score(y_cls_test, y_cls_pred)
prec = precision_score(y_cls_test, y_cls_pred)
rec = recall_score(y_cls_test, y_cls_pred)
f1 = f1_score(y_cls_test, y_cls_pred)
auc = roc_auc_score(y_cls_test, y_cls_prob)

print(f"\n  测试集评估:")
print(f"    准确率: {acc:.4f}")
print(f"    精确率: {prec:.4f}")
print(f"    召回率: {rec:.4f}")
print(f"    F1:     {f1:.4f}")
print(f"    AUC:    {auc:.4f}")


# ============================================================
# 4. 模型3：多层前馈神经网络（nn.Module）
# ============================================================
print("\n" + "=" * 60)
print("[4] 模型3：多层前馈神经网络")
print("=" * 60)


class MLPModel(nn.Module):
    """多层前馈神经网络：输入 → 隐藏层1 → ReLU → 隐藏层2 → ReLU → 输出"""

    def __init__(self, input_dim, hidden1=32, hidden2=16, output_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, output_dim),
        )

    def forward(self, x):
        return self.net(x)


# --- 4.1 MLP 做回归 ---
print("\n  --- MLP 回归任务 ---")
model_mlp_reg = MLPModel(input_dim=1, output_dim=1)
criterion_mse = nn.MSELoss()
optimizer_mlp_reg = optim.Adam(model_mlp_reg.parameters(), lr=0.01)

losses_mlp_reg = []

for epoch in range(200):
    y_pred = model_mlp_reg(X_reg_train_t)
    loss = criterion_mse(y_pred, y_reg_train_t)

    optimizer_mlp_reg.zero_grad()
    loss.backward()
    optimizer_mlp_reg.step()

    losses_mlp_reg.append(loss.item())

    if (epoch + 1) % 50 == 0:
        print(f"  Epoch {epoch+1:>3d}, Loss: {loss.item():.6f}")

model_mlp_reg.eval()
with torch.no_grad():
    y_mlp_reg_pred = model_mlp_reg(X_reg_test_t)
    mlp_reg_mse = criterion_mse(y_mlp_reg_pred, y_reg_test_t).item()

print(f"  测试集 MSE: {mlp_reg_mse:.6f}")

# --- 4.2 MLP 做分类 ---
print("\n  --- MLP 分类任务 ---")
model_mlp_cls = MLPModel(input_dim=2, output_dim=1)
criterion_bce2 = nn.BCELoss()
optimizer_mlp_cls = optim.Adam(model_mlp_cls.parameters(), lr=0.01)

losses_mlp_cls = []

for epoch in range(300):
    y_pred = model_mlp_cls(X_cls_train_t)
    y_pred = torch.sigmoid(y_pred)
    loss = criterion_bce2(y_pred, y_cls_train_t)

    optimizer_mlp_cls.zero_grad()
    loss.backward()
    optimizer_mlp_cls.step()

    losses_mlp_cls.append(loss.item())

    if (epoch + 1) % 50 == 0:
        print(f"  Epoch {epoch+1:>3d}, Loss: {loss.item():.6f}")

model_mlp_cls.eval()
with torch.no_grad():
    y_mlp_cls_prob = torch.sigmoid(model_mlp_cls(X_cls_test_t)).numpy()
    y_mlp_cls_pred = (y_mlp_cls_prob >= 0.5).astype(int)

acc_mlp = accuracy_score(y_cls_test, y_mlp_cls_pred)
prec_mlp = precision_score(y_cls_test, y_mlp_cls_pred)
rec_mlp = recall_score(y_cls_test, y_mlp_cls_pred)
f1_mlp = f1_score(y_cls_test, y_mlp_cls_pred)
auc_mlp = roc_auc_score(y_cls_test, y_mlp_cls_prob)

print(f"\n  测试集评估:")
print(f"    准确率: {acc_mlp:.4f}")
print(f"    精确率: {prec_mlp:.4f}")
print(f"    召回率: {rec_mlp:.4f}")
print(f"    F1:     {f1_mlp:.4f}")
print(f"    AUC:    {auc_mlp:.4f}")


# ============================================================
# 5. 结果可视化
# ============================================================
print("\n" + "=" * 60)
print("[5] 结果可视化")
print("=" * 60)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# --- 图1：线性回归拟合效果 ---
ax1 = axes[0, 0]
ax1.scatter(X_reg_test, y_reg_test, color='steelblue', alpha=0.6, edgecolors='k', s=40, label='真实值')
sort_idx = X_reg_test[:, 0].argsort()
with torch.no_grad():
    y_pred_sorted = model_linear(X_reg_test_t[sort_idx]).numpy()
ax1.plot(X_reg_test[sort_idx], y_pred_sorted, color='red', linewidth=2, label='线性回归')
ax1.set_title('线性回归拟合效果', fontsize=13)
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.legend()
ax1.grid(True, alpha=0.3)

# --- 图2：线性回归损失曲线 ---
ax2 = axes[0, 1]
ax2.plot(losses_linear, color='steelblue', linewidth=1.5)
ax2.set_title('线性回归 Loss 曲线', fontsize=13)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('MSE Loss')
ax2.grid(True, alpha=0.3)

# --- 图3：逻辑回归决策边界 ---
ax3 = axes[0, 2]
h = 0.02
x_min, x_max = X_cls[:, 0].min() - 1, X_cls[:, 0].max() + 1
y_min, y_max = X_cls[:, 1].min() - 1, X_cls[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
grid = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()])
model_logistic.eval()
with torch.no_grad():
    Z = model_logistic(grid).numpy().reshape(xx.shape)
ax3.contourf(xx, yy, Z, alpha=0.3, cmap='RdBu')
ax3.scatter(X_cls_test[:, 0], X_cls_test[:, 1], c=y_cls_test, cmap='RdBu',
            edgecolors='k', s=40, alpha=0.7)
ax3.set_title(f'逻辑回归决策边界 (AUC={auc:.3f})', fontsize=13)
ax3.set_xlabel('特征1')
ax3.set_ylabel('特征2')
ax3.grid(True, alpha=0.3)

# --- 图4：逻辑回归损失曲线 ---
ax4 = axes[1, 0]
ax4.plot(losses_logistic, color='coral', linewidth=1.5)
ax4.set_title('逻辑回归 Loss 曲线', fontsize=13)
ax4.set_xlabel('Epoch')
ax4.set_ylabel('BCE Loss')
ax4.grid(True, alpha=0.3)

# --- 图5：MLP 回归拟合效果 ---
ax5 = axes[1, 1]
ax5.scatter(X_reg_test, y_reg_test, color='steelblue', alpha=0.6, edgecolors='k', s=40, label='真实值')
with torch.no_grad():
    y_mlp_sorted = model_mlp_reg(X_reg_test_t[sort_idx]).numpy()
ax5.plot(X_reg_test[sort_idx], y_mlp_sorted, color='red', linewidth=2, label='MLP')
ax5.set_title(f'MLP 回归拟合 (MSE={mlp_reg_mse:.4f})', fontsize=13)
ax5.set_xlabel('x')
ax5.set_ylabel('y')
ax5.legend()
ax5.grid(True, alpha=0.3)

# --- 图6：MLP 分类决策边界 ---
ax6 = axes[1, 2]
with torch.no_grad():
    Z_mlp = torch.sigmoid(model_mlp_cls(grid)).numpy().reshape(xx.shape)
ax6.contourf(xx, yy, Z_mlp, alpha=0.3, cmap='RdBu')
ax6.scatter(X_cls_test[:, 0], X_cls_test[:, 1], c=y_cls_test, cmap='RdBu',
            edgecolors='k', s=40, alpha=0.7)
ax6.set_title(f'MLP 分类决策边界 (AUC={auc_mlp:.3f})', fontsize=13)
ax6.set_xlabel('特征1')
ax6.set_ylabel('特征2')
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(r'D:\机器学习\PyTorch实验结果.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# 6. 模型对比与总结
# ============================================================
print("\n" + "=" * 60)
print("[6] 模型对比总结")
print("=" * 60)

print("""
┌─────────────────────────────────────────────────────────────────┐
│                    PyTorch 模型构建基本流程                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. 定义模型（继承 nn.Module）                                  │
│     class MyModel(nn.Module):                                   │
│         def __init__(self):                                     │
│             super().__init__()                                  │
│             self.layer = nn.Linear(...)                         │
│         def forward(self, x):                                   │
│             return self.layer(x)                                │
│                                                                 │
│  2. 定义损失函数                                                │
│     回归: nn.MSELoss()                                          │
│     分类: nn.BCELoss() / nn.CrossEntropyLoss()                 │
│                                                                 │
│  3. 定义优化器                                                  │
│     optim.SGD(model.parameters(), lr=0.01)                     │
│     optim.Adam(model.parameters(), lr=0.001)                   │
│                                                                 │
│  4. 训练循环                                                    │
│     for epoch in range(n_epochs):                               │
│         y_pred = model(x)          # 前向传播                   │
│         loss = criterion(y_pred, y) # 计算损失                  │
│         optimizer.zero_grad()       # 清空梯度                  │
│         loss.backward()             # 反向传播（计算梯度）       │
│         optimizer.step()            # 参数更新                  │
│                                                                 │
│  5. 评估                                                        │
│     model.eval()                                                │
│     with torch.no_grad():                                       │
│         y_pred = model(x_test)                                  │
└─────────────────────────────────────────────────────────────────┘
""")

print(f"{'指标':<12} {'逻辑回归':<12} {'MLP 分类':<12}")
print("-" * 36)
print(f"{'准确率':<12} {acc:<12.4f} {acc_mlp:<12.4f}")
print(f"{'精确率':<12} {prec:<12.4f} {prec_mlp:<12.4f}")
print(f"{'召回率':<12} {rec:<12.4f} {rec_mlp:<12.4f}")
print(f"{'F1':<12} {f1:<12.4f} {f1_mlp:<12.4f}")
print(f"{'AUC':<12} {auc:<12.4f} {auc_mlp:<12.4f}")

print(f"\n{'模型':<12} {'测试集 MSE':<15}")
print("-" * 27)
print(f"{'线性回归':<12} {test_mse:<15.6f}")
print(f"{'MLP 回归':<12} {mlp_reg_mse:<15.6f}")

print("""
总结：
1. 线性回归：最简单的模型，只能拟合线性关系
2. 逻辑回归：在线性回归基础上加 Sigmoid，用于二分类
3. MLP：多层网络 + 非线性激活函数，能拟合复杂关系
4. MLP 通常比简单模型效果更好，但需要更多数据和调参
""")