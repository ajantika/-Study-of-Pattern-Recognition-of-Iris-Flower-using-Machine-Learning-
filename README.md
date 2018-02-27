# Study-of-Pattern-Recognition-of-Iris-Flower-using-Machine-Learning-
#Using Python


# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 00:39:24 2018

@author: ajant
"""

#Loading the Library

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.cross_validation import train_test_split

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score

from sklearn.cross_validation import cross_val_score

from matplotlib.colors import ListedColormap

from sklearn import neighbors

#loading the Dataset

from sklearn.datasets import load_iris

df=load_iris()


# create X (features) and y (response)

x = df.iloc[:, [0, 3]].values

y = df.iloc[:, 4].values

# Splitting the dataset into the Training set and Test set

from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.33, random_state = 0)

# Fitting classifier to the Training set


from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier (n_neighbors=5 ,metric='minkowski', p=2 )

classifier.fit(X_train , y_train)


#  Predict the response
pred = classifier.predict(X_test)

# Evaluate accuracy

acc = accuracy_score(y_test, pred) * 100

print('\nThe accuracy of the knn classifier for k = 5 is %d%%' % acc)

# Creating odd list of K for KNN

myList = list(range(0,50))

neighbors = list(filter(lambda x: x % 2 != 0, myList))


# Empty list that will hold cv scores

cv_scores = []

# Performing 10-fold cross-validation with K=5 for KNN (the n_neighbors parameter)

# k = 5 for KNeighborsClassifier

# scoring='accuracy' for evaluation metric


knn=KNeighborsClassifier(n_neighbors=5)

scores=cross_val_score(knn,x,y,cv=10,scoring='accuracy')

print (scores)

#In the first iteration, the accuracy is 100%

#Second iteration, the accuracy is 93% and so on


# perform 10-fold cross validation

for k in neighbors:

    knn = KNeighborsClassifier(n_neighbors=k)
    
    scores = cross_val_score(knn, X_train, y_train, cv=10, scoring='accuracy')
    
    cv_scores.append(scores.mean())
    
    print (cv_scores)
    
    
# changing to misclassification error

MSE = [1 - x for x in cv_scores]
MSE

# Determining best k
optimal_k = neighbors[MSE.index(min(MSE))]

print('\nThe optimal number of neighbors is %d.' % optimal_k)




# Plot misclassification error vs k 

plt.plot(neighbors, MSE)

plt.xlabel('Number of Neighbors K')

plt.ylabel('Misclassification Error')

plt.show()
