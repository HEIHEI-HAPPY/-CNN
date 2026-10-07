import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import joblib
import os
from collections import Counter
from PIL import Image
# 定义函数，接受图片索引
def show_digit(index):
    # 加载数据集
    digits = pd.read_csv(r'D:\机器学习\KNN算法（K近邻算法）\手写数字识别\data\knn_training_28x28.csv')
    #判断是否越界
    if index<0 or index>=digits.shape[0]:
        print("索引越界，请输入正确的索引")
        return
    # 获取图片数据
    X_trans = digits.iloc[:,1:].values
    y_trans = digits.iloc[:,0].values
    #查看图片是几
    print("图片是：%d"%y_trans[index])
    #查看x的形状
    print("x的形状是：%s"%str(X_trans[index].shape))
    # 将图片数据转换为28x28的二维数组
    image = X_trans[index].reshape(28,28)
    # 显示图片
    plt.imshow(image, cmap='gray')
    plt.title(f'Label: {y_trans[index]}')
    #plt.axis('off')
    plt.show()
#定义函数，训练模型，保存模型
def train_model():
    # 加载数据集
    digits = pd.read_csv(r'D:\机器学习\KNN算法（K近邻算法）\手写数字识别\data\knn_training_28x28.csv')
    # 获取图片数据
    X_trans = digits.iloc[:,1:].values
    y_trans = digits.iloc[:,0].values
    # 打印数据集的形状
    print("数据集的形状是：%s"%str(X_trans.shape))
    print("标签的形状是：%s"%str(y_trans.shape))
    #归一化
    X_trans = X_trans/255
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X_trans, y_trans, test_size=0.2, random_state=42, stratify=y_trans)
    # 创建KNN分类器
    knn = KNeighborsClassifier(n_neighbors=1)
    # 训练模型
    knn.fit(X_train, y_train)
    # 在测试集上进行预测
    y_pred = knn.predict(X_test)
    # 计算准确率
    accuracy = knn.score(X_test, y_test)
    print("模型的准确率是：%s"%str(accuracy))
    # 保存模型
    os.makedirs(r'D:\机器学习\KNN算法（K近邻算法）\手写数字识别\model', exist_ok=True)
    joblib.dump(knn, r'D:\机器学习\KNN算法（K近邻算法）\手写数字识别\model\knn_model.pkl')

#定义函数，识别图片数字
def predict_digit(image_path):
    # 加载模型
    model_path = r'D:\机器学习\KNN算法（K近邻算法）\手写数字识别\model\knn_model.pkl'
    knn = joblib.load(model_path)
    # 读取图片，转为灰度，缩放为28x28
    img = Image.open(image_path).convert('L')
    img = img.resize((28, 28))
    # 转为numpy数组，展平，归一化
    img_array = np.array(img).flatten().reshape(1, -1) / 255.0
    # 预测
    result = knn.predict(img_array)[0]
    # 显示图片和结果
    print("识别结果是：%d" % result)
    plt.imshow(np.array(img), cmap='gray')
    plt.title(f'Predict: {result}')
    plt.axis('off')
    plt.show()
    return result

#测试
if __name__=='__main__':
    #显示图片
    #show_digit(34)
    #训练模型 保存模型
    #train_model()
    #识别图片数字
    predict_digit(r'D:\机器学习\KNN算法（K近邻算法）\手写数字识别\data\test_digit_1.png')