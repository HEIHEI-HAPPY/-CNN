"""
交叉验证与网格搜索 - 鸢尾花(Iris)数据集
目标：
  1. 使用交叉验证评估多个模型的稳定性
  2. 使用网格搜索为最优模型自动调参
  3. 对比调参前后的效果
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    StratifiedKFold,
    GridSearchCV,
)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# ============================================================
# 第一步：加载数据并划分
# ============================================================
print("=" * 60)
print("第一步：加载 Iris 数据集")
print("=" * 60)

iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"训练集: {X_train_scaled.shape[0]} 样本")
print(f"测试集: {X_test_scaled.shape[0]} 样本")

# ============================================================
# 第二步：交叉验证对比多个模型
# ============================================================
print(f"\n{'=' * 60}")
print("第二步：5折交叉验证对比模型")
print("=" * 60)

models = {
    "逻辑回归": LogisticRegression(max_iter=200, random_state=42),
    "KNN":      KNeighborsClassifier(),
    "决策树":   DecisionTreeClassifier(random_state=42),
    "随机森林": RandomForestClassifier(random_state=42),
    "SVM":      SVC(random_state=42),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print(f"\n{'模型':<15} {'平均准确率':>10} {'标准差':>10} {'每折结果'}")
print("-" * 75)

cv_results = {}
for name, model in models.items():
    scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring="accuracy")
    cv_results[name] = scores
    print(f"{name:<15} {scores.mean():>9.4f}   {scores.std():>9.4f}   {np.round(scores, 4)}")

# 找出最佳模型
best_name = max(cv_results, key=lambda k: cv_results[k].mean())
print(f"\n>>> 交叉验证最佳模型: {best_name} (平均准确率 {cv_results[best_name].mean():.4f})")

# ============================================================
# 第三步：网格搜索调参
# ============================================================
print(f"\n{'=' * 60}")
print(f"第三步：对「{best_name}」进行网格搜索调参")
print("=" * 60)

# 为每个模型定义参数网格
param_grids = {
    "逻辑回归": {
        "C": [0.01, 0.1, 1, 10, 100],
        "solver": ["lbfgs", "liblinear"],
    },
    "KNN": {
        "n_neighbors": [1, 3, 5, 7, 9, 11],
        "weights": ["uniform", "distance"],
        "metric": ["euclidean", "manhattan"],
    },
    "决策树": {
        "max_depth": [2, 3, 5, 7, None],
        "min_samples_split": [2, 5, 10],
        "criterion": ["gini", "entropy"],
    },
    "随机森林": {
        "n_estimators": [50, 100, 200],
        "max_depth": [3, 5, 7, None],
        "min_samples_split": [2, 5],
    },
    "SVM": {
        "C": [0.1, 1, 10, 100],
        "kernel": ["rbf", "linear", "poly"],
        "gamma": ["scale", "auto", 0.01, 0.1],
    },
}

# 调参前的基准分数（默认参数）
base_model = models[best_name]
base_scores = cross_val_score(base_model, X_train_scaled, y_train, cv=cv)
print(f"\n调参前（默认参数）: 平均准确率 = {base_scores.mean():.4f}")

# 网格搜索
grid_search = GridSearchCV(
    estimator=models[best_name],
    param_grid=param_grids[best_name],
    cv=cv,
    scoring="accuracy",
    n_jobs=-1,       # 使用所有CPU核心
    verbose=0,
)
grid_search.fit(X_train_scaled, y_train)

print(f"调参后（最优参数）: 平均准确率 = {grid_search.best_score_:.4f}")
print(f"\n最优参数:")
for param, value in grid_search.best_params_.items():
    print(f"  {param}: {value}")

# ============================================================
# 第四步：最终测试集评估
# ============================================================
print(f"\n{'=' * 60}")
print("第四步：在测试集上最终评估")
print("=" * 60)

# 调参前模型
y_pred_before = models[best_name].fit(X_train_scaled, y_train).predict(X_test_scaled)
acc_before = accuracy_score(y_test, y_pred_before)

# 调参后模型
best_model = grid_search.best_estimator_
y_pred_after = best_model.predict(X_test_scaled)
acc_after = accuracy_score(y_test, y_pred_after)

print(f"\n{'':>20} {'准确率':>10}")
print("-" * 32)
print(f"{'调参前（默认参数）':<20} {acc_before * 100:>8.2f}%")
print(f"{'调参后（最优参数）':<20} {acc_after * 100:>8.2f}%")
print(f"{'提升':<20} {'+' if acc_after >= acc_before else ''}{(acc_after - acc_before) * 100:>7.2f}%")

print(f"\n最优模型详细分类报告:")
print(classification_report(y_test, y_pred_after, target_names=iris.target_names))

# ============================================================
# 第五步：所有模型网格搜索排名
# ============================================================
print(f"\n{'=' * 60}")
print("第五步：所有模型网格搜索结果排名")
print("=" * 60)

all_grid_results = []
for name, model in models.items():
    gs = GridSearchCV(
        estimator=model,
        param_grid=param_grids[name],
        cv=cv,
        scoring="accuracy",
        n_jobs=-1,
        verbose=0,
    )
    gs.fit(X_train_scaled, y_train)
    all_grid_results.append((name, gs.best_score_, gs.best_params_))

all_grid_results.sort(key=lambda x: x[1], reverse=True)

print(f"\n{'排名':<6} {'模型':<12} {'最佳交叉验证准确率':>18}")
print("-" * 40)
for i, (name, score, _) in enumerate(all_grid_results, 1):
    bar = "█" * int(score * 20)
    print(f" {i:<6} {name:<12} {score * 100:>15.2f}%   {bar}")

print(f"\n>>> 最终推荐模型: {all_grid_results[0][0]}")
print(f"   最优参数: {all_grid_results[0][2]}")
