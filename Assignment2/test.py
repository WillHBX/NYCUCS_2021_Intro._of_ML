######################
# data preprocessing #
######################

import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.decomposition import PCA 

# read and shuffle
origin_df = pd.read_csv("student-mat.csv",sep = ';')
origin_df = origin_df.sample(frac=1, random_state=42).reset_index(drop=True)

categorical_col = list(filter(lambda col: type(origin_df[col][0])== str, origin_df.columns))
numeric_col = list(filter(lambda col: type(origin_df[col][0])== np.int64, origin_df.columns))

# one-hot encoding
categorical_df = pd.get_dummies(origin_df.loc[:,categorical_col])

# drop lable and normalize
numeric_df = origin_df.loc[:,numeric_col].drop('G3',axis = 'columns')
#numeric_df = (numeric_df-numeric_df.min())/(numeric_df.max()-numeric_df.min())

all_x = categorical_df.join(numeric_df)
all_y = origin_df['G3']
y_binary = []
y_5level = []

for num in  all_y:
    y_binary.append(num >= 10)
    if num < 10:
        y_5level.append('F')
    elif num < 12:
        y_5level.append('D')
    elif num < 14:
        y_5level.append('C')
    elif num < 16:
        y_5level.append('B')
    elif num <= 20:
        y_5level.append('A')

split = int(len(all_x.index)*7/10)
X_train = all_x[:split]
X_test = all_x[split:]
y_train_binary = y_binary[:split]
y_test_binary = y_binary[split:]
y_train_5level = y_5level[:split]
y_test_5level = y_5level[split:]

# pca ( only for numeric data )
pca = PCA(n_components='mle')
pca_numeric_df = pca.fit_transform(numeric_df)
pca_all_x = categorical_df.join(numeric_df)

X_train_pca = pca_all_x[:int(len(pca_all_x)*7/10)]
X_test_pca = pca_all_x[int(len(pca_all_x)*7/10):]
# numeric_df
# len(pca_numeric_df[0])

# decision tree
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from matplotlib import pyplot as plt

clf = DecisionTreeClassifier(max_leaf_nodes = 4, random_state=0)
clf.fit(X_train, y_train_binary)
dt_y_predict_binary = clf.predict(X_test)
print(clf.score(X_test,y_test_binary))
tree.plot_tree(clf)
plt.show()
