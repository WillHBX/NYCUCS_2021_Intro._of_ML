from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import re
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


df = pd.read_csv('ionosphere.data',sep = ',',header=None)
df = df.sample(frac=1) # shuffle

split =  int(len(df.index)*7/10)

train_X = df.iloc[:split,:-1]
test_X = df.iloc[split:,:-1]

train_y = df.iloc[:split,-1]
test_y = df.iloc[split:,-1]



parameters = {'kernel': ['poly'],
                'degree': [3,9,27],
                'gamma': [1, 0.1, 0.001],
                'coef0': [1,10,100]
                }

clf = GridSearchCV(SVC(max_iter = 1000), parameters)
clf.fit(train_X,train_y)

print(clf.best_estimator_)
grid_predictions = clf.predict(test_X)
print(confusion_matrix(test_y,grid_predictions))
print(classification_report(test_y,grid_predictions))#Output