import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from images import generate_img_dataset
from baseline import count_nodes

np.random.seed(41)
images,y = generate_img_dataset(dataset="REDDIT-MULTI-5K",filtration="degree",extended=True,spread = 0.2, pixels=[7,7],dimensions=[0,1])


scoring_metric = "accuracy"

param_grid = [
  { "n_estimators": [ 100], "max_depth":[5,20,None]}
]
clf=GridSearchCV(RandomForestClassifier(),param_grid=param_grid,scoring=scoring_metric,n_jobs=-1,cv=5)


mean_accuracies=[]
for i in range(10):
  clf = GridSearchCV(RandomForestClassifier(),param_grid=param_grid,scoring = scoring_metric,n_jobs=-1,cv=5)
  scores = cross_val_score(clf,X=images,y=y,cv=10)
  mean_accuracies.append(np.mean(scores))
print("{:.1f}\u00B1{:.1f}".format(np.mean(mean_accuracies)*100,np.std(mean_accuracies)*100))
