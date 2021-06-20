
# In[]:
import pandas as pd
wine_data = 'D:/lyp/2017_2018_2/python_code/MTrain/MachineLearn/3_ML/10_ensembling/wine.data'
df_wine = pd.read_csv(wine_data, header=None)

df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash',
                   'Alcalinity of ash', 'Magnesium', 'Total phenols',
                   'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
                   'Color intensity', 'Hue', 'OD280/OD315 of diluted wines',
                   'Proline']
df_wine.head()

# In[]:
set(df_wine['Class label'])

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

# 对label值进行编码
le = LabelEncoder()
y = le.fit_transform(y)
y

# In[]:
X_train, X_test, y_train, y_test =            train_test_split(X, y, 
                             test_size=0.2, 
                             random_state=1,
                             stratify=y)


# In[]:
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(criterion='entropy', 
                              max_depth=None,
                              random_state=1)

bag = BaggingClassifier(base_estimator=tree,
                        n_estimators=500, 
                        n_jobs=1, 
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

# In[]:
bag = bag.fit(X_train, y_train)
y_train_pred = bag.predict(X_train)
y_test_pred = bag.predict(X_test)

bag_train = accuracy_score(y_train, y_train_pred) 
bag_test = accuracy_score(y_test, y_test_pred) 
print('Bagging train/test accuracies %.3f/%.3f'
      % (bag_train, bag_test))
