import sys
from class_vis import prettyPicture
from prep_terrain_data import makeTerrainData
from time import time

import matplotlib.pyplot as plt
import copy
import numpy as np
import pylab as pl


features_train, labels_train, features_test, labels_test = makeTerrainData()


########################## SVM #################################
### we handle the import statement and SVC creation for you here
from sklearn.svm import SVC

def class_SVC(features_train, labels_train):
    from sklearn.svm import SVC
    t0 = time()
    clf = SVC(kernel="linear")
    clf.fit(features_train, labels_train)
    print("training time:", round(time() - t0, 3), "s")
    return clf

#### now your job is to fit the classifier
#### using the training features/labels, and to
#### make a set of predictions on the test data

SVC_test = class_SVC(features_train, labels_train)
t0 = time()
terrain_pred = SVC_test.predict(features_test)
print("prediction time:", round(time() - t0, 3), "s")

#### store your predictions in a list named pred

from sklearn.metrics import accuracy_score
acc = accuracy_score(labels_test, terrain_pred)

def submitAccuracy():
    print(acc)

submitAccuracy()