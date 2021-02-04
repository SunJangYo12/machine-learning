import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
import matplotlib.pyplot as plt

data = pd.read_csv('Salary_Data.csv')
#print(data.head())

#memisahkan atribut dan label
X = data['YearsExperience']
y = data['Salary']

#mengubah bentuk atribut
X = X[:, np.newaxis]

#membangun model dengan parameter C, gamma dan kernel
model = SVR()
parameters = {
    'kernel': ['rbf'],
    'C'     : [1000, 10000, 100000],
    'gamma' : [0.5, 0.05, 0.005]
}
grid_search = GridSearchCV(model, parameters)

#melatih model dengan fungsi dengan fungsi fit
grid_search.fit(X, y)

#menampilkan parameter terbaik dari object grid_search
#print(grid_search.best_params_)

#membuat model SVM baru dengan parameter terbaik hasil grid search
model_baru = SVR(C=100000, gamma=0.005, kernel='rbf')
model_baru.fit(X, y)

plt.scatter(X, y)
plt.plot(X, model_baru.predict(X))
plt.show()