# 正则化演示：L1（Lasso）与 L2（Ridge）

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 1. 生成数据（二次函数 + 噪声）
# ============================================================
np.random.seed(42)
x = np.random.uniform(-3, 3, 100)
y = 0.5 * x**2 + x + np.random.normal(0, 1, 100)

# 数据预处理：构造9次多项式特征（故意过拟合）
x_1d = x.copy().reshape(-1, 1)
poly = PolynomialFeatures(degree=9, include_bias=False)
x_poly = poly.fit_transform(x_1d)

# 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(
    x_poly, y, test_size=0.2, random_state=42
)

print("=" * 60)
print("正则化对比实验：L1（Lasso）vs L2（Ridge）")
print("=" * 60)
print(f"特征数: {x_poly.shape[1]}（9次多项式）")
print(f"训练集: {x_train.shape[0]} 样本")
print(f"测试集: {x_test.shape[0]} 样本")

# ============================================================
# 2. 训练三个模型：无正则 / L1 / L2
# ============================================================
models = {
    '无正则 (OLS)': LinearRegression(),
    'L1 正则 (Lasso, α=0.1)': Lasso(alpha=0.1, max_iter=10000),
    'L2 正则 (Ridge, α=1.0)': Ridge(alpha=1.0),
}

results = {}
for name, model in models.items():
    model.fit(x_train, y_train)

    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)

    results[name] = {
        'model': model,
        'train_mse': train_mse,
        'test_mse': test_mse,
        'coefs': model.coef_,
    }

# ============================================================
# 3. 结果对比
# ============================================================
print("\n" + "=" * 60)
print("MSE 对比")
print("=" * 60)
print(f"{'模型':<30} {'训练MSE':<12} {'测试MSE':<12} {'过拟合比':<10}")
print("-" * 64)
for name, r in results.items():
    ratio = r['test_mse'] / r['train_mse']
    print(f"{name:<30} {r['train_mse']:<12.4f} {r['test_mse']:<12.4f} {ratio:<10.2f}")

# ============================================================
# 4. 系数对比（核心：看 L1 如何让系数变0）
# ============================================================
print("\n" + "=" * 60)
print("系数对比（L1 会将部分系数压缩为 0）")
print("=" * 60)
feature_names = [f'x^{i}' for i in range(1, 10)]
print(f"{'特征':<8}", end='')
for name in models:
    print(f"{name:<22}", end='')
print()
print("-" * 72)
for i, fname in enumerate(feature_names):
    print(f"{fname:<8}", end='')
    for name in results:
        val = results[name]['coefs'][i]
        print(f"{val:<22.4f}", end='')
    print()

# 统计非零系数个数
print("\n" + "=" * 60)
print("非零系数统计")
print("=" * 60)
for name, r in results.items():
    non_zero = np.sum(np.abs(r['coefs']) > 1e-6)
    print(f"{name}: {non_zero}/{len(r['coefs'])} 个非零系数")

# ============================================================
# 5. 可视化
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 生成平滑曲线
x_line = np.linspace(x_1d.min(), x_1d.max(), 300).reshape(-1, 1)
x_line_poly = poly.transform(x_line)

# 提取原始 x 值
x_train_1d = x_train[:, 0]
x_test_1d = x_test[:, 0]
sort_idx = x_train_1d.argsort()

# 图1-3：三个模型的拟合效果
for idx, (name, r) in enumerate(results.items()):
    row, col = divmod(idx, 2)
    ax = axes[row, col]

    y_line = r['model'].predict(x_line_poly)

    ax.scatter(x_train_1d, y_train, color='steelblue', label='训练集', alpha=0.6, edgecolors='k', s=40)
    ax.scatter(x_test_1d, y_test, color='orange', label='测试集', alpha=0.6, edgecolors='k', s=40)
    ax.plot(x_line, y_line, color='red', linewidth=2, label='拟合曲线')
    ax.set_title(f"{name}\n训练MSE={r['train_mse']:.2f}, 测试MSE={r['test_mse']:.2f}", fontsize=12)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(y.min() - 2, y.max() + 2)

# 图4：系数对比柱状图
ax4 = axes[1, 1]
x_pos = np.arange(len(feature_names))
width = 0.25

bars1 = ax4.bar(x_pos - width, results['无正则 (OLS)']['coefs'], width,
                label='无正则', color='steelblue', edgecolor='k')
bars2 = ax4.bar(x_pos, results['L1 正则 (Lasso, α=0.1)']['coefs'], width,
                label='L1 (Lasso)', color='coral', edgecolor='k')
bars3 = ax4.bar(x_pos + width, results['L2 正则 (Ridge, α=1.0)']['coefs'], width,
                label='L2 (Ridge)', color='seagreen', edgecolor='k')

ax4.set_xticks(x_pos)
ax4.set_xticklabels(feature_names)
ax4.set_title('系数对比（L1 将部分系数压缩为 0）', fontsize=12)
ax4.set_ylabel('系数值')
ax4.legend()
ax4.grid(True, alpha=0.3, axis='y')
ax4.axhline(y=0, color='black', linewidth=0.8)

plt.tight_layout()
plt.savefig(r'D:\机器学习\线性回归\L1_L2正则化对比.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# 6. 正则化原理说明
# ============================================================
print("\n" + "=" * 60)
print("正则化原理")
print("=" * 60)
print("""
一、为什么需要正则化？
   高次多项式容易过拟合 → 模型复杂度高 → 训练误差低但测试误差高
   正则化通过惩罚过大的系数，降低模型复杂度

二、L1 正则（Lasso）
   损失函数 = MSE + α × Σ|wᵢ|
   - 使用系数的绝对值之和作为惩罚
   - 效果：将部分系数压缩为 exactly 0 → 自动特征选择
   - 适用场景：特征多但有用特征少（稀疏模型）

三、L2 正则（Ridge）
   损失函数 = MSE + α × Σwᵢ²
   - 使用系数的平方和作为惩罚
   - 效果：将所有系数均匀缩小，但不会变为 0
   - 适用场景：特征之间有相关性，需要保留所有特征

四、L1 vs L2 对比
   特性              L1 (Lasso)          L2 (Ridge)
   ─────────────────────────────────────────────────
   惩罚项            |wᵢ| (绝对值)       wᵢ² (平方)
   系数能否为0       能（自动选特征）    不能（只缩小）
   解的唯一性        可能不唯一          唯一
   几何形状          菱形（棱角）        圆形（光滑）
   适用场景          稀疏模型            所有特征都有用

五、α（alpha）的作用
   α = 0     → 无正则化，等同于普通线性回归
   α 小      → 弱正则化，轻微抑制过拟合
   α 大      → 强正则化，系数趋近于0，可能欠拟合
""")
