import sys
import time
from class_vis import prettyPicture
from prep_terrain_data import makeTerrainData

import numpy as np
import pylab as pl

features_train, labels_train, features_test, labels_test = makeTerrainData()

#################################################################################

########################## DECISION TREE #################################

#### your code goes here
from classifyDT import classify
Tree_Test = classify(features_train, labels_train)
t0 = time()
terrain_pred = Tree_Test.predict(features_test)
print("prediction time:", round(time() - t0, 3), "s")
print(terrain_pred)

from sklearn.metrics import accuracy_score
acc = accuracy_score(labels_test, terrain_pred)

### be sure to compute the accuracy on the test set

def submitAccuracies():
    return {"acc": round(acc, 3)}
