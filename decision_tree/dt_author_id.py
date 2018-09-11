#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Thu  Aug 9 23:43:32 2018

@author : gaurav gahukar
        : caffeine110

AIM     : (p) TO Impliment UDACITY MiniProject ( Naive Bayes )
        : (s) Email Classification to predict Author of the Email

        : --- Inspired by the story of J K Roling ---
          The Sunday (UK) Times recently revealed that J.K. Rowling wrote the
          detective novel The Cuckoo's Calling under the pen name Robert Galbraith.
          After classification using classification algorithms it was confirmed that
          J K Rolling wrote that book. 
"""

""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
    Sara has label 0
    Chris has label 1
"""

    
import sys
from time import time

sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()



#importing sklearn
from sklearn import tree

#initialising dtc classifier
dtc_classifier = tree.DecisionTreeClassifier()

#fitting data to the classifier
dtc_classifier.fit(features_train, labels_train)


#prediction of author
author_pred = dtc_classifier.predict(features_test)

#importing accuracy score
from sklearn.metrics import accuracy_score

#printing the accuracy scores
print(accuracy_score(labels_test, author_pred))
print(dtc_classifier.score(features_test, labels_test))