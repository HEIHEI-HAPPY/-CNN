# 实验1：机器学习基础流程
# 目标：理解机器学习的基本步骤

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 1. 加载数据集（Iris）
# ============================================================
iris = load_iris()
X = iris.data               # 4个特征：花萼长/宽、花瓣长/宽
y = iris.target              # 真实标签（0,1,2 三类）
feature_names = iris.feature_names
target_names = iris.target_names

print("=" * 50)
print("数据集基本信息")
print("=" * 50)
print(f"样本数: {X.shape[0]}")
print(f"特征数: {X.shape[1]}")
print(f"特征名: {feature_names}")
print(f"类别名: {target_names}")
print(f"各类样本数: {np.bincount(y)}")

# 标准化（无监督学习也需要标准化，避免量纲影响）
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ============================================================
# 2. 模型1：KMeans 聚类（直接在原始4维特征上聚类）
# ============================================================
kmeans_raw = KMeans(n_clusters=3, random_state=42, n_init=10)
y_kmeans_raw = kmeans_raw.fit_predict(X_scaled)

# 用PCA降到2维，方便可视化聚类结果
pca_for_viz = PCA(n_components=2)
X_2d = pca_for_viz.fit_transform(X_scaled)

# 计算聚类准确率（无监督学习的评估：与真实标签对比）
from sklearn.metrics import adjusted_rand_score
from scipy.optimize import linear_sum_assignment

def cluster_accuracy(y_true, y_pred):
    """计算聚类准确率（匈牙利算法匹配标签）"""
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)
    row_ind, col_ind = linear_sum_assignment(-cm)
    return cm[row_ind, col_ind].sum() / len(y_true)

ari_raw = adjusted_rand_score(y, y_kmeans_raw)
acc_raw = cluster_accuracy(y, y_kmeans_raw)

print("\n" + "=" * 50)
print("模型1：KMeans 直接聚类（4维特征）")
print("=" * 50)
print(f"调整兰德指数 (ARI): {ari_raw:.4f}")
print(f"聚类准确率: {acc_raw:.4f}")

# ============================================================
# 3. 模型2：PCA 降维（4维 → 2维，无聚类）
# ============================================================
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print("\n" + "=" * 50)
print("模型2：PCA 降维（4维 → 2维）")
print("=" * 50)
print(f"各主成分方差解释比: {pca.explained_variance_ratio_}")
print(f"累计方差解释比: {sum(pca.explained_variance_ratio_):.4f}")

# ============================================================
# 4. 模型3：组合模型（先 PCA 降维，再 KMeans 聚类）
# ============================================================
kmeans_combined = KMeans(n_clusters=3, random_state=42, n_init=10)
y_kmeans_combined = kmeans_combined.fit_predict(X_pca)

ari_combined = adjusted_rand_score(y, y_kmeans_combined)
acc_combined = cluster_accuracy(y, y_kmeans_combined)

print("\n" + "=" * 50)
print("模型3：PCA + KMeans 组合模型")
print("=" * 50)
print(f"调整兰德指数 (ARI): {ari_combined:.4f}")
print(f"聚类准确率: {acc_combined:.4f}")

# ============================================================
# 5. 可视化对比
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 图1：真实标签
ax1 = axes[0, 0]
scatter1 = ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', edgecolors='k', s=60)
ax1.set_title('真实标签（Ground Truth）', fontsize=14)
ax1.set_xlabel('主成分1')
ax1.set_ylabel('主成分2')
legend1 = ax1.legend(*scatter1.legend_elements(),
                       labels=list(target_names), title="类别")
ax1.add_artist(legend1)

# 图2：KMeans 直接聚类结果
ax2 = axes[0, 1]
scatter2 = ax2.scatter(X_2d[:, 0], X_2d[:, 1], c=y_kmeans_raw, cmap='viridis', edgecolors='k', s=60)
centers_2d = pca_for_viz.transform(kmeans_raw.cluster_centers_)
ax2.scatter(centers_2d[:, 0], centers_2d[:, 1], c='red', marker='X', s=200, edgecolors='k', label='聚类中心')
ax2.set_title(f'KMeans 直接聚类（ACC={acc_raw:.2f}）', fontsize=14)
ax2.set_xlabel('主成分1')
ax2.set_ylabel('主成分2')
ax2.legend()

# 图3：PCA 降维效果（用真实标签着色，展示降维后类别可分性）
ax3 = axes[1, 0]
scatter3 = ax3.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', edgecolors='k', s=60)
ax3.set_title(f'PCA 降维（解释方差={sum(pca.explained_variance_ratio_):.2%}）', fontsize=14)
ax3.set_xlabel(f'主成分1（{pca.explained_variance_ratio_[0]:.2%}）')
ax3.set_ylabel(f'主成分2（{pca.explained_variance_ratio_[1]:.2%}）')
legend3 = ax3.legend(*scatter3.legend_elements(),
                       labels=list(target_names), title="类别")
ax3.add_artist(legend3)

# 图4：PCA + KMeans 组合模型
ax4 = axes[1, 1]
scatter4 = ax4.scatter(X_pca[:, 0], X_pca[:, 1], c=y_kmeans_combined, cmap='viridis', edgecolors='k', s=60)
ax4.scatter(kmeans_combined.cluster_centers_[:, 0], kmeans_combined.cluster_centers_[:, 1],
            c='red', marker='X', s=200, edgecolors='k', label='聚类中心')
ax4.set_title(f'PCA + KMeans 组合（ACC={acc_combined:.2f}）', fontsize=14)
ax4.set_xlabel('主成分1')
ax4.set_ylabel('主成分2')
ax4.legend()

plt.tight_layout()
plt.savefig(r'D:\机器学习\实验1_聚类与降维对比.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# 6. 结果汇总
# ============================================================
print("\n" + "=" * 50)
print("结果汇总")
print("=" * 50)
print(f"{'模型':<25} {'ARI':<10} {'准确率':<10}")
print("-" * 45)
print(f"{'KMeans 直接聚类':<25} {ari_raw:<10.4f} {acc_raw:<10.4f}")
print(f"{'PCA + KMeans 组合':<25} {ari_combined:<10.4f} {acc_combined:<10.4f}")
