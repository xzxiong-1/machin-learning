# 使用mode函数进行最后投票
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
pred1=model1.predict(x_test)
pred2=model2.predict(x_test)
pred3=model3.predict(x_test)

# In[]:
pred1
 
# In[]:
import numpy as np
from scipy.stats import mode
# 测试一个样本
list = ['a', 'a', 'a', 'b', 'b', 'b', 'a']
print("# Print mode(list):", mode(list))
print("# list中最常见的成员为：{}，出现了{}次。".format(mode(list)[0][0], mode(list)[1][0]))

# In[]:
# 观察一次
final_pred = np.array([])
for i in range(0,len(x_test)):
    final_pred =np.append(final_pred, mode([pred1[i], pred2[i], pred3[i]])[0][0])
#    break

#final_pred

# In[]:
# 全部运行
final_pred = np.array([])
for i in range(0,len(x_test)):
    final_pred =np.append(final_pred, mode([pred1[i], pred2[i],pred3[i]])[0][0])

#final_pred

# In[]:
# 统计正确率
r = y_test==final_pred
r

# In[]:
print("准确率",np.sum(r==True)/y_test.shape[0])
