import sklearn
from sklearn import datasets
from sklearn import tree
from sklearn.model_selection import cross_val_score

#load iris
iris = datasets.load_iris()

#mendefinisikan atribut dan label pada dataset
x = iris.data
y = iris.target

#membuat model dengan decision tree classifier
clf = tree.DecisionTreeClassifier()

#mengevaluasi performa model dengan cross_val_score
scores = cross_val_score(clf, x, y, cv=5)

print(scores)
