from sklearn.datasets import load_iris
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score 

#1定义函数，加载数据，查看数据库
def dm1_loadiris():
    #1加载数据
    iris_data = load_iris()
    #2查看数据集
    #print(f'数据集描述：\n{iris_data}')
    #print(f'数据集类型：\n{type(iris_data)}')
    #3查看数据集所有键
    print(f'数据集所有键：\n{iris_data.keys()}')
    #4查看数据集键值
    print(f'具体数据：\n{iris_data.data[:5]}')#前5行数据
    print(f'具体标签：\n{iris_data.target[:5]}')#前5行标签
    print(f'具体特征名：\n{iris_data.feature_names}')#特征名
    print(f'具体标签名：\n{iris_data.target_names}')#标签名称
    print(f'具体数据集描述：\n{iris_data.DESCR}')#数据集描述
    print(f'具体数据集框架：\n{iris_data.frame}')#数据集框架
    print(f'：数据集文件名\n{iris_data.filename}')#数据集文件名
    print(f'数据集模型（在那个包下）：\n{iris_data.data_module}')#数据集模块
#2
def dm2_show_iris():
    #1加载数据
    iris_data = load_iris()
    #2转换为DataFrame对象
    iris_df = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)
    #3添加标签列
    iris_df['label'] = iris_data.target
    #print(iris_df)
    #4通过seaborn可视化绘图
    sns.lmplot(data=iris_df, x='sepal length (cm)', y='sepal width (cm)', hue='label', fit_reg=True)
    plt.title('鸢尾花萼片长宽分布（含拟合线）')
    plt.tight_layout()
    plt.show()
    
#3定义函数，切分数据集，训练模型
def dm3_split_train_test():
    #1加载数据
    iris_data = load_iris()
    #2数据预处理：切分数据集为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        iris_data.data, iris_data.target, test_size=0.2, random_state=42
    )
    #打印划分结果
    print(f'训练集特征：\n{X_train}\n个数:{len(X_train)}')
    print(f'测试集特征：\n{X_test}\n个数:{len(X_test)}')
    print(f'训练集标签：\n{y_train}\n个数:{len(y_train)}')
    print(f'测试集标签：\n{y_test}\n个数:{len(y_test)}')

#4定义函数
def dm4_iris_evaluate_test():
    #1加载数据
    iris_data = load_iris()
    #2数据预处理：切分数据集为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        iris_data.data, iris_data.target, test_size=0.2, random_state=42
    )
    #3特征工程
    #提取（标准化）特征（以提取过）
    #标准化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
#5
if __name__ == "__main__":
    #dm1_loadiris()
    #dm2_show_iris()
    #dm3_split_train_test()  #3定义函数，切分数据集，训练模型
    dm4_iris_evaluate_test()