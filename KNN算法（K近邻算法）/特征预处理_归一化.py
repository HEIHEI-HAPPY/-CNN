"""
目的
利用特征预处理技术对数据进行归一化处理，以提高机器学习模型的性能和训练速度。
步骤
1特征提取
2特征预处理
3特征降维
4特征选择
5特征组合

归一化目的
防止某些特征由于量纲（单位）不同而对模型训练产生较大影响，通过归一化将所有特征缩放到相同范围内，提高模型的稳定性和收敛速度。
公式
x' = (x - min) / (max - min)
x''= x' * (new_max - new_min) + new_min
其中，x为原始特征值，min和max分别为该特征的最小值和最大值，new_min和new_max为归一化后的新范围的最小值和最大值，
x'为归一化后的特征值。

弊端
    容易受到异常值（极大和极小值）的影响，导致归一化结果失真。
    所以一般使用于小数据集或对异常值不敏感的场景。
"""
#导包
from sklearn.preprocessing import MinMaxScaler
import numpy as np
#准备数据
x_train = np.array([[90, 2, 10,40], [60, 4, 15,45], [75, 3, 13,46]])
#创建归一化对象
transfer = MinMaxScaler()
#归一化处理
x_train_new = transfer.fit_transform(x_train)
#查看结果
print(x_train_new)