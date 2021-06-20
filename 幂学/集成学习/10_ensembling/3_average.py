# 求概率平均值

# In[]:
from sklearn import datasets
import numpy as np

iris = datasets.load_iris()
x = iris.data[:, [2, 3]]
y = iris.target

x.shape, y.shape

# In[]:
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=1, stratify=y)

print('Labels counts in y:', np.bincount(y))
print('Labels counts in y_train:', np.bincount(y_train))
print('Labels counts in y_test:', np.bincount(y_test))

# In[]:
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

model1 = DecisionTreeClassifier()
model2 = KNeighborsClassifier()
model3= LogisticRegression()

# In[]:
model1.fit(x_train,y_train)
model2.fit(x_train,y_train)
model3.fit(x_train,y_train)

# In[]:
pred1=model1.predict_proba(x_test)

pred2=model2.predict_proba(x_test)

pred3=model3.predict_proba(x_test)
print(pred1)
finalpred=(pred1+pred2+pred3)/3

# In[]:
finalpred
print(finalpred)

## 加权平均法
# In[]:
finalpred2 = (pred1*0.3+pred2*0.3+pred3*0.4)
finalpred2