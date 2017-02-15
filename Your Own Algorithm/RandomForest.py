from prep_terrain_data import makeTerrainData
from sklearn.metrics import accuracy_score
from time import time

def ClassR(features_train, labels_train):
    from sklearn.ensemble import RandomForestClassifier
    clf=RandomForestClassifier(n_estimators=10).fit(features_train, labels_train)
    return clf

features_train, labels_train, features_test, labels_test = makeTerrainData()

random_test=ClassR(features_train, labels_train)
t0=time()
terrain_pred=random_test.predict(features_test)
print("prediction time:", round(time() - t0, 3), "s")
#print(terrain_pred)

acc=accuracy_score(labels_test,terrain_pred)
print(acc)