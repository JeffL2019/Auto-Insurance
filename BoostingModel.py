import time
from operator import itemgetter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics, neural_network
from sklearn.model_selection import train_test_split
from sklearn.ensemble import (AdaBoostClassifier, GradientBoostingClassifier,
                              RandomForestClassifier, RandomForestRegressor)
df=pd.read_csv("Modelling.csv")
# print(df.head())

start = time.time()

X = df.drop('OUTCOME',axis=1)
y = df['OUTCOME']

scorers = {'Accuracy': 'accuracy'}

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

Classifier = neural_network.MLPClassifier(activation='logistic',
                                          solver='lbfgs',
                                          alpha=0.0001,
                                          max_iter=100000000,
                                          hidden_layer_sizes=(10, ),
                                          random_state=1)
Classifier.fit(X_train, y_train)

# feat_val= [[ 2, 1, 0, 1, 2, 1, 1, 0, 1, 0, 0, 1, 1, 0 ]]
# feat_val= [[ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ]]
feat_val= [[ 3, 1, 1, 3, 2, 3, 1, 1, 1, 1, 1, 1, 1, 1 ]]
predict = Classifier.predict(feat_val)
print(predict)

print(Classifier.score(X_train, y_train))
print("runtime :", time.time() - start)

# scores = cross_validate(estimator=Classifier,
#                         X=X_train,
#                         y=y_train,
#                         scoring=scorers,
#                         cv=5)

# scores_Acc = scores['test_Accuracy']
# print("Random Forest Acc: %0.2f (+/- %0.2f)" %
#         (scores_Acc.mean(), scores_Acc.std() * 2))
