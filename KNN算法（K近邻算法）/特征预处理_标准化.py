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
所以通过公式把各列的值映射到均值为0，标准差为1的正态分布上。

公式
x'=(当前值-均值)/标准差
x'=(x-mean)/std
其中，x为原始特征值，mean为该特征的均值，std为该特征的标准差，x'为标准化后的特征值。

应用场景
适用于大数据集和对异常值敏感的场景。

"""
#导包
from sklearn.preprocessing import StandardScaler
import numpy as np
#准备数据
x_train = np.array([[90, 2, 10,40], [60, 4, 15,45], [75, 3, 13,46]])
#创建标准化对象
transfer = StandardScaler()
#归一化处理
x_train_new = transfer.fit_transform(x_train)
#查看结果
print('标准化结果：\n')
print(x_train_new)
#查看均值和标准差
print('每列的均值：\n',transfer.mean_)
print('每列的标准差：\n',transfer.scale_)#std
print('每列的方差：\n',transfer.var_)#方差=标准差的平方 var