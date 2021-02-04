import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz

iris = pd.read_csv('Iris.csv')
#print(iris.head())

#menghilangkan kolom yang tidak penting
iris.drop('Id', axis=1, inplace=True)
#print(iris.head())

#memisahkan atribut dan label 
X = iris[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = iris['Species']

#membuat model Decision Tree
tree_model = DecisionTreeClassifier()

#melakukan pelatihan model terhadap data
tree_model.fit(X, y)

#prediksi model dengan tree_model.predict([[SepalLength,  SepalWidthCm, PetalLengthCm, PetalWidthCm]])
out = tree_model.predict([[6.2, 3.4, 5.2, 2.3]])

print(out)

#menghasilkan file dot untuk lihat visualisasi dari decision tree dan untuk melihat harus di convert ke png online
export_graphviz(
    tree_model,
    out_file = "Iris_tree.dot",
    feature_names = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm'],
    class_names = ['Iris-setosa', 'Iris-versioncolor', 'Iris-virginica'],
    rounded = True,
    filled = True
)