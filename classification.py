from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from images import generate_img_dataset

images,y = generate_img_dataset(dataset="MUTAG",spread = 1, pixels=[10,10])

X_train, X_test, y_train, y_test = train_test_split(images, y, test_size=0.20, random_state=42)

param_grid = [
  {'kernel': ['linear'] , 'C': [1, 10, 100, 1000]},
  {'kernel': ['rbf'], 'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001]},
 ]

scoring_metric = "accuracy"

clf = GridSearchCV(SVC(),param_grid=param_grid,scoring = scoring_metric,n_jobs=-1,cv=5)
clf.fit(X_train,y_train)

svm = SVC(**clf.best_params_)
svm.fit(X_train,y_train)
print(clf.score(X_test,y_test))