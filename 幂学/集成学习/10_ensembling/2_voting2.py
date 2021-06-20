# 使用sklearn自带库进行投票
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
    x, y, test_size=0.5, random_state=1, stratify=y)

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
from sklearn.ensemble import VotingClassifier
model = VotingClassifier(estimators=[('dt', model1), ('knn', model2), ('lr', model3)],
                     voting='hard')

model.fit(x_train,y_train)
pred=model.score(x_test,y_test)

print(pred)



