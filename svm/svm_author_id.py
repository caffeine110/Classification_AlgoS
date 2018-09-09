#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu  Aug  9 23:43:32 2018

@author : gaurav gahukar
        : caffeine110

AIM     : (p) TO Impliment UDACITY MiniProject ( SVM classifier )
        : (s) Email Classification to predict Author of the Email

        : --- Inspired by the story of J K Roling ---
          The Sunday (UK) Times recently revealed that J.K. Rowling wrote the
          detective novel The Cuckoo's Calling under the pen name Robert Galbraith.
          After classification using classification algorithms it was confirmed that
          J K Rolling wrote that book. 
"""


""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
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


#import
from sklearn import svm

#initialising support vector classification 
svm_classifier = svm.SVC()

#fitting data to the model classifier
svm_classifier.fit(features_train, labels_train)


#predict the author
author_pred = svm_classifier.predict(features_test)


#import accuracy_score
from sklearn.metrics import accuracy_score

#priniting score
print(accuracy_score(labels_test, author_pred))
print(svm_classifier.score(features_test, labels_test))