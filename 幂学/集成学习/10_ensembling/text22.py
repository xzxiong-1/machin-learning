# _*_ coding:utf-8_*_
# 开发人员：**
# 开发时间：2019/11/21 13:57
# 文件名称：text22.py 
# 开发工具：PyCharm
import pandas as pd
wine_data =r'D:/wine.data'
df_wine=pd.read_csv(wine_data,header=None)

df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash',
                   'Alcalinity of ash', 'Magnesium', 'Total phenols',
                   'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
                   'Color intensity', 'Hue', 'OD280/OD315 of diluted wines',
                   'Proline']
df_wine.head()

set(df_wine["Class label"])
df_wine=df_wine[df_wine["Class label"]!=1]
set(df_wine["Class label"])

y=df_wine["Class label"].values
X = df_wine[['Alcohol', 'OD280/OD315 of diluted wines']].values

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selecting import train_test_spilt

le=LabelEncoder()
y=le.fit_transform()

X_train,X_test,y_train,y_test=train_test_spilt (X,  y,
                                                test_size=0.2,
                                                random_state=1,
                                                stratify=y)
print("X,Y",(X_train.shape,X_test.shape))

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

tree =DecisionTreeClassifier(criterion='entropy',
                              max_depth=1,
                              random_state=1)

ada = AdaBoostClassifier(base_estimator=tree,
                         n_estimators=500,
                         learning_rate=0.1,
                         random_state=1)

from  sklearn.metrics import accuracy_score
tree=tree.fit(X_train,y_train)
y_train_prd=tree.predict(X_train)
y_test_prd=tree.predict(X_test)

tree_train = accuracy_score(y_train, y_train_pred)
tree_test = accuracy_score(y_test, y_test_pred)
print('Decision tree train/test accuracies %.3f/%.3f'
      % (tree_train, tree_test))

ada=ada.fit(X_train,y_train)
y_train_pred =ada.predict(X_train)
y_test_pred=ada.predict(X_test)

ada_train = accuracy_score(y_train, y_train_pred)
ada_test = accuracy_score(y_test, y_test_pred)
print('AdaBoost train/test accuracies %.3f/%.3f'
      % (ada_train, ada_test))
