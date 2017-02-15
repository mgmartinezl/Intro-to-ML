import sys
from class_vis import prettyPicture
from prep_terrain_data import makeTerrainData
from time import time

import matplotlib.pyplot as plt
import copy
import numpy as np
import pylab as pl

def classify(features_train, labels_train):
    ### your code goes here--should return a trained decision tree classifer
    from sklearn import tree
    clf=tree.DecisionTreeClassifier(min_samples_split=2)
    clf=clf.fit(features_train,labels_train)
    return clf

features_train, labels_train, features_test, labels_test = makeTerrainData()

Tree_Test = classify(features_train, labels_train)
t0 = time()
terrain_pred = Tree_Test.predict(features_test)
print("prediction time:", round(time() - t0, 3), "s")
print(terrain_pred)

from sklearn.metrics import accuracy_score
acc = accuracy_score(labels_test, terrain_pred)

def submitAccuracies():
    return {"acc": round(acc, 3)}

print(submitAccuracies())