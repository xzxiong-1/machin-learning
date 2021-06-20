# _*_ coding:utf-8_*_
# 开发人员：**
# 开发时间：2019/11/19 14:30
# 文件名称：text1.py 
# 开发工具：PyCharm
from   sklearn import  datasets
import numpy as np
iris =datasets.load_iris ()
x=iris.data[:,[2,3]]
y=iris.target
print (x,y)

from sklearn.model_selection import train_test_split

x_train ,x_test,y_train ,y_test=train_test_split(x,y,test_size=0.3,random_state=1,stratify=y)

print ("labels count in y",np.bincount(y))
print ("labels count in y_train",np.bincount(y_train))
print ("labels count in y_test",np.bincount(y_test))

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors  import   KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

model1=DecisionTreeClassifier( )
model2=KNeighborsClassifier( )
model3=LogisticRegression( )

model1.fit(x_train,y_train)
model2.fit(x_train,y_train)
model3.fit(x_train,y_train)

print(model1.fit(x_train,y_train))

pred1=model1.predict_proba(x_test)
pred2=model2.predict_proba(x_test)
pred3=model3.predict_proba(x_test)
finalpred=(pred1+pred2+pred3)/3
print (finalpred)
