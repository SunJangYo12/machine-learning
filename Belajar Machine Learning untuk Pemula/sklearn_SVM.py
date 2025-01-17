import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

df = pd.read_csv('diabetes.csv')
#print(df.info())

#memisahkan atribut pada dataset dan menyimpanya pada sebuah variabel
X = df[df.columns[:8]]

#memisahkan label pada dataset dan menyimpanya pada sebuah variabel
y = df['Outcome']

#standarisasi nilai-nilai dari dataset
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

#memisahkan data untuk training dan testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)

#membuat object SVC dan memanggil fungsi fit untuk melatih model
clf = SVC()
clf.fit(X_train, y_train)

#menampilkan skor akuransi prediksi
print(clf.score(X_test, y_test))