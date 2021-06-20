# In[]:
import pandas as pd
wine_data = r'D:/wine.data'
df_wine = pd.read_csv(wine_data, header=None)

df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash',
                   'Alcalinity of ash', 'Magnesium', 'Total phenols',
                   'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
                   'Color intensity', 'Hue', 'OD280/OD315 of diluted wines',
                   'Proline']
df_wine.head()

# In[]:
set(df_wine['Class label'])

print ("set:",set(df_wine['Class label']))
# In[]:
# 去掉一个类别，保留2类
df_wine = df_wine[df_wine['Class label'] != 1]
set(df_wine['Class label'])

# In[]:
y = df_wine['Class label'].values
X = df_wine[['Alcohol', 'OD280/OD315 of diluted wines']].values


# In[]:
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# 对label值进行编码将值从2，3转化为0，1
le = LabelEncoder()
y = le.fit_transform(y)
y

# In[]:
X_train, X_test, y_train, y_test =            train_test_split(X, y, 
                             test_size=0.2, 
                             random_state=1,
                             stratify=y)
X_train.shape, X_test.shape

# In[]:
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(criterion='entropy',
                              max_depth=1,
                              random_state=1)

ada = AdaBoostClassifier(base_estimator=tree,
                         n_estimators=500, 
                         learning_rate=0.1,
                         random_state=1)


# In[]:
from sklearn.metrics import accuracy_score
tree = tree.fit(X_train, y_train)
y_train_pred = tree.predict(X_train)
y_test_pred = tree.predict(X_test)

tree_train = accuracy_score(y_train, y_train_pred)
tree_test = accuracy_score(y_test, y_test_pred)
print('Decision tree train/test accuracies %.3f/%.3f'
      % (tree_train, tree_test))

ada = ada.fit(X_train, y_train)
y_train_pred = ada.predict(X_train)
y_test_pred = ada.predict(X_test)

print("y_train_pred",y_train_pred)
ada_train = accuracy_score(y_train, y_train_pred) 
ada_test = accuracy_score(y_test, y_test_pred) 
print('AdaBoost train/test accuracies %.3f/%.3f'
      % (ada_train, ada_test))


