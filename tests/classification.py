import numpy as np
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC, SVC
from sklearn.feature_selection import SelectFromModel

from GCTDA.images import generate_img_dataset
from GCTDA.baseline import count_nodes, histogram_degrees, laplacian_spectrum, heat_kernel_trace


np.random.seed(41)
X,y = generate_img_dataset(dataset="MUTAG",filtration="node_betweenness",extended=True,spread = 0.1, pixels=[7,7],dimensions=[0,1],weighting_type="uniform")
#X,y = laplacian_spectrum("NCI1")
y = np.array(y)
#images2,y2 = generate_img_dataset(dataset="REDDIT-MULTI-5K",filtration="node_betweenness",extended=False,spread = 0.15, pixels=[7,7],dimensions=[0,1])
#X2,y2 = histogram_degrees("NCI1")

#X = [np.concatenate([X[i],X2[i]]) for i in range(len(X))]

"""
from sklearn.feature_selection import VarianceThreshold
sel = VarianceThreshold(threshold=0.001)
X =sel.fit_transform(X)"""

"""
from sklearn.decomposition import PCA
pca = PCA(n_components=5)
X = pca.fit_transform(X)"""


"""
import matplotlib.pyplot as plt
for i,img in enumerate(images):
  #img = img[:49]
  img = img[49*0:49*1]
  image = img.reshape((7,7))
  print(y[i])
  plt.imshow(image)
  plt.show()"""

scoring_metric = "accuracy"


param_grid = { "n_estimators": [ 100], "max_depth":[3,5,10,20,40,60]}
#param_grid = {"C":[0.1,1,10]}

import time
start = time.time()
X = np.array(X)
mean_accuracies=[]
for i in range(10):
  cv = StratifiedKFold(n_splits=10, shuffle=True,random_state=i)
  scores=[]
  for train_index, test_index in cv.split(X, y):
    clf = GridSearchCV(RandomForestClassifier(),param_grid=param_grid,scoring = scoring_metric,n_jobs=-1,cv=5)
    #clf = GridSearchCV(SVC(kernel="linear"),param_grid=param_grid,scoring = scoring_metric,n_jobs=-1,cv=5)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    res = clf.fit(X_train, y_train)
    #print(clf.best_params_)
    #print(clf.cv_results_)
    scores.append(clf.score(X_test,y_test))
  #scores = cross_val_score(clf,X=X,y=y,cv=cv)
  mean_accuracies.append(np.mean(scores))
  print(np.mean(scores))
print(time.time()-start)
print("{:.1f}\u00B1{:.1f}".format(np.mean(mean_accuracies)*100,np.std(mean_accuracies)*100))


