# water-potability1
```python
import warnings 
import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
import missingno as msno 
from warnings import simplefilter 
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression 
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier 
from sklearn.svm import SVC 
from xgboost import XGBClassifier 
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix 
simplefilter("ignore") 
data = pd.read_csv('/kaggle/input/water-potability/water_potability.csv') 
data.head()

data_count = data.isnull().sum()
print(data_count[0:10])
print(f'The dataset has {data.isna().sum().sum()} null values.')
data.describe()

#柱状图筛选缺失严重的列
null_proportion=(data_count/len(data))*100
#转为dataframe并整理索引
null_proportion=null_proportion.reset_index(name='count')
#画布大小
fig = plt.figure(figsize=(12,6))
fig = sns.barplot(null_proportion, x="index", y="count")
fig.set_title('Null Values in the Data', fontsize=30)
fig.set_xlabel('features', fontsize=12)
fig.set_ylabel('% of null values', fontsize=12)
fig.bar_label(fig.containers[0], fmt='%.1f')
#自动优化布局
plt.tight_layout()

#矩阵图查看缺失是否和行绑定决定删除行还是补充列
#组合分析数据清洗更精准后续建模稳
msno.matrix(data, color=(0,0.3,0.5))
plt.title('Null Values Show in Graph', fontsize=20)
plt.show()

#直方图查看数据指标是否正常
#加入正态分布
cols = data.columns
#指标多用循环效率高
for i in range (4):
#创建一行三列子图
    fig, (ax1,ax2,ax3) = plt.subplots(ncols=3, figsize=(12,6))
    ax1 = sns.histplot(data[cols[i*3]], ax=ax1,kde=True)
    ax1.set_title(f"Histogram of '{cols[i*3]}'", size=16)
    if i < 3:
        ax2 = sns.histplot(data[cols[i*3+1]], ax=ax2,kde=True)
        ax2.set_title(f"Histogram of '{cols[i*3+1]}'", size=16)
        ax3 = sns.histplot(data[cols[i*3+2]], ax=ax3,kde=True)
        ax3.set_title(f"Histogram of '{cols[i*3+2]}'", size=16)
#自动优化
    plt.tight_layout()

#绘制散点图查看ph和硬度是否具有关联
plt.figure(figsize=(10,6))
sns.scatterplot(x=data['ph'], y=data['Hardness'], s=80,alpha=0.7, hue=data['Potability'])
plt.title('ScatterPlot Of PH, Hardness', fontsize=15)
plt.xlabel('Ph', fontsize=14)
plt.ylabel('Hardness', fontsize=13)
plt.show()

#利用箱线图查看异常值
plt.figure(figsize=(13,20))
for k, cols in enumerate(data):
    plt.subplot(5,2, k+1)
    sns.boxplot(x=cols,data=data,color='#264D58')
    plt.title(f"Check Outliers- {cols}", fontsize=13)
    plt.tight_layout(pad=4.0)    
plt.show()

#特征关系挖掘，相关性热力图 heatmap分析水质特征间关联关系防止过度拟合
plt.figure(figsize=(10,6))
sns.heatmap(data.corr(), annot=True, cmap='Blues')
plt.title('Correlation Metrics', fontsize=15)
plt.show()

#处理数据缺失值中位数填充分组填充
for col in ['ph', 'Sulfate', 'Trihalomethanes']:
    # 计算分组中位数
    group_median = data.groupby('Potability')[col].transform('median')
    data[col].fillna(group_median, inplace=True)
print(data[['ph', 'Sulfate', 'Trihalomethanes']].isnull().sum())

#划分特征和标签为后续建模准备--特征工程
X= data.drop(columns='Potability', axis=1)
y= data['Potability']

#将特征矩阵 X 和目标向量 y 按照指定比例划分为训练集和测试集
X_train,X_test, y_train,y_test = train_test_split(X,y , test_size=0.3)
print(f"X_train Shape: {X_train.shape}")
print(f"Y_train Shape: {y_train.shape}")
print(f"X_test Shape: {X_test.shape}")
print(f"y_test Shape: {y_test.shape}")

# 创建StandardScaler对象，用于特征标准化
scaler = StandardScaler()
#仅使用训练数据计算均值和标准差
scaler.fit(X_train,)
# 应用标准化变换到训练数据
X_train_scaled  = scaler.transform(X_train)
X_test_scaled  = scaler.transform(X_test)


#储存多种算法模型（逻辑回归，随机森林，集中学习中的bagging
models = {
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier(),
    'Bagging Classifier': BaggingClassifier(),
    'Adaboost Classifier': AdaBoostClassifier(),
    'Gradeint Boosting Classifier': GradientBoostingClassifier(),
    'SVC': SVC(),
    'XGBoost Classifier': XGBClassifier()}

for name, model in models.items():
    # 使用训练数据拟合模型
    model.fit(X_train_scaled, y_train)
    # 对测试数据进行预测
    y_preds = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_preds)
    precision= precision_score(y_test, y_preds, average='macro')
    
    print(f' Model Name: {name}')
    print(f'Accuracy Score: {accuracy:.2f}')
    print(f'Precision Score: {precision}')
    print('_' * 50)
    print()
    # 计算混淆矩阵
    cm = confusion_matrix(y_test, y_preds)
    plt.figure(figsize=(15,5))
    plt.subplot(1, 2, 1)
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='d')  # 添加fmt='d'显示整数
    plt.title(f'{name}- Confusion Matrix', fontsize=15)  # 修正拼写错误
    plt.subplot(1, 2, 2)
    sns.kdeplot(y_test.values, color='royalblue', label='Actual Values')  # 深蓝色
    sns.kdeplot(y_preds, color='cyan', label='Predicted Values')          # 青色
    plt.title(f'{name}- Actual Vs Predicted', fontsize=15)
    plt.legend()
    plt.show()
