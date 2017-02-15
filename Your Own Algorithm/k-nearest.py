from prep_terrain_data import makeTerrainData
from sklearn.metrics import accuracy_score
from time import time

def Class(features_train, labels_train):
    from sklearn.neighbors import KNeighborsClassifier
    clf=KNeighborsClassifier(n_neighbors=2).fit(features_train, labels_train)
    return clf

features_train, labels_train, features_test, labels_test = makeTerrainData()

knear_test=Class(features_train, labels_train)
t0=time()
terrain_pred=knear_test.predict(features_test)
print("prediction time:", round(time() - t0, 3), "s")
#print(terrain_pred)

acc=accuracy_score(labels_test,terrain_pred)
print(acc)

