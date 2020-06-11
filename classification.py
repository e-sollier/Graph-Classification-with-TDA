import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.svm import SVC
from images import generate_img_dataset

np.random.seed(41)
images,y = generate_img_dataset(dataset="MUTAG",filtration="jaccard",order="sublevel",spread = 1, pixels=[10,10])

param_grid = [
  {'kernel': ['linear'] , 'C': [1, 10, 100, 1000]},
  {'kernel': ['rbf'], 'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001]},
 ]
scoring_metric = "accuracy"
clf = GridSearchCV(SVC(),param_grid=param_grid,scoring = scoring_metric,n_jobs=-1,cv=5)

nested_cv=True

if not nested_cv:
  #Perform only one train/test split. Faster, but the score is less stable.
  X_train, X_test, y_train, y_test = train_test_split(images, y, test_size=0.25)
  clf.fit(X_train,y_train)
  svm = SVC(**clf.best_params_)
  svm.fit(X_train,y_train)
  print(svm.score(X_test,y_test))
else:
  scores = cross_val_score(clf,X=images,y=y,cv=5)
  print(scores)
  print(np.mean(scores))