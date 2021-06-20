# _*_ coding:utf-8_*_
# 开发人员：**
# 开发时间：2019/11/18 22:05
# 文件名称：test1.py 
# 开发工具：PyCharm
from  sklearn import datasets
import numpy as py

iris = datasets.load_iris()
x=iris.data[: ,[2,3]]
y = iris.target
print(x,y)
from sklearn.model_selection import train_test_split
x_train ,x_test,y_train,y_test =train_test_spilt(x,y,test_size=0.3,random_state=1,stratify=y)
print('Labels counts in y:', np.bincount(y))
print('Labels counts in y_train:', np.bincount(y_train))
print('Labels counts in y_test:', np.bincount(y_test))
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.liner_moder import LogisticRegression
model1 = DecisionTreeClassifier()