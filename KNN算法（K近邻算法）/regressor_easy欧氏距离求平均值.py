#导包
from sklearn.neighbors import KNeighborsRegressor
#准备数据
x_train = [[0,0,1], [1,1,0], [3,10,10], [4,11,12]]#多维特征
x_test = [[1,3,10],[4,12,12]]
y_train = [0.1, 0.2, 0.3, 0.4]#回归目标值
#创建模型
estimator = KNeighborsRegressor(n_neighbors=2)
#训练模型
estimator.fit(x_train, y_train)
#模型预测
y_test = estimator.predict(x_test)
print(y_test)
print(estimator.kneighbors(x_test, return_distance=True))