#!/usr/bin/python

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

#########################################################
### your code goes here ###
#features_train = features_train[:int(len(features_train)/100)]
#labels_train = labels_train[:int(len(labels_train)/100)]

def class_SVC(features_train, labels_train):
    from sklearn.svm import SVC
    t0 = time()
    clf = SVC(kernel="rbf", C=10000)
    clf.fit(features_train, labels_train)
    print("training time:", round(time() - t0, 3), "s")
    return clf

SVC_test = class_SVC(features_train, labels_train)
t0 = time()
terrain_pred = SVC_test.predict(features_test)
print("prediction time:", round(time() - t0, 3), "s")

print(sum(terrain_pred))
#print(terrain_pred[10])
#print(terrain_pred[26])
#print(terrain_pred[50])

from sklearn.metrics import accuracy_score
acc = accuracy_score(labels_test, terrain_pred)

def submitAccuracy():
    print(acc)

submitAccuracy()

#########################################################


