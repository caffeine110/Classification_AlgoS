#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on  Fri Aug 10 23:43:32 2018

@author : gaurav gahukar
        : caffeine110

AIM     : (p) TO Impliment UDACITY MiniProject ( K-Nearest Neighbours Algorithms)
        : (s) Email Classification to predict Author of the Email

        : --- Inspired by the story of J K Roling ---
          The Sunday (UK) Times recently revealed that J.K. Rowling wrote the
          detective novel The Cuckoo's Calling under the pen name Robert Galbraith.
          After classification using classification algorithms it was confirmed that
          J K Rolling wrote that book. 
"""


""" 
    This is the code to accompany the Lesson 1 ( KNN ) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1
"""


#importing required libraries
import sys
from time import time

#Set the working directory
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()



#import Knn classifier
from sklearn.neighbors import KNeighborsClassifier

#initialising classifier
knn_classifier = KNeighborsClassifier()



#fitting data to the classifer
knn_classifier.fit(features_train, labels_train)



from sklearn.metrics import accuracy_score
#prediction
author_pred = knn_classifier.predict(features_test)


#printing the accuracy
print(accuracy_score(labels_test, author_pred))
print(knn_classifier.score(features_test, labels_test))
