from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.decomposition import PCA

iris = datasets.load_iris()
atribut = iris.data
label = iris.target

#bagi dataset menjadi train set dan test set
X_train, X_test, y_train, y_test = train_test_split(atribut, label, test_size=0.2)

decision_tree = tree.DecisionTreeClassifier()
model_pertama = decision_tree.fit(X_train, y_train)
print(model_pertama.score(X_test, y_test))

#membuat object PCA dengan 4 principal component
pca = PCA(n_components=4)

#mengaplikasikan PCA pada dataset
pca_attributs = pca.fit_transform(X_train)

#melihat variance dari setiap attribut
print(pca.explained_variance_ratio_)

#PCA dengan 2 principal component 
pca = PCA(n_components = 2)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.fit_transform(X_test)

#uji akuransi classifier
model2 = decision_tree.fit(X_train_pca, y_train)
print(model2.score(X_test_pca, y_test))