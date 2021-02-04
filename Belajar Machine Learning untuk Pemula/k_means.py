import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

#ubah file csv menjadi dataframe
df = pd.read_csv("Mall_Customers.csv")

#tampilkan tiga baris pertama
print(df.head(3))

#ubah nama kolom
df = df.rename(columns={'Gender':'gender',
                        'Age':'age',
                        'Annual Income (k$)':'annual_income',
                        'Spending Score (1-100)':'spending_score'
                        })

#ubah data katagorik menjadi data numerik
df['gender'].replace(['Female', 'Male'], [0, 1], inplace=True)

#tampilkan data yang sudah di preprocess
print(df.head(3))

#menghilangkan kolom customer id dan gender
X = df.drop(['CustomerID', 'gender'], axis=1)

#membuat list yang berisi inertia
clusters = []
for i in range(1, 11):
    km = KMeans(n_clusters=i).fit(X)
    clusters.append(km.inertia_)

#membuat plot inertia
fig, ax = plt.subplots(figsize=(8, 4))
sns.lineplot(x=list(range(1, 11)), y=clusters, ax=ax)
ax.set_title('Cari Elbow')
ax.set_xlabel('Clusters')
ax.set_ylabel('Inertia')

#membuat object KMeans
km5 = KMeans(n_clusters=5).fit(X)

#menambahkan kolom label pada dataset
X['Labels'] = km5.labels_

#membuat plot KMeans dengan 5 klatser
plt.figure(figsize=(8, 4))
sns.scatterplot(X['annual_income'], X['spending_score'], hue=X['Labels'], 
                palette=sns.color_palette('hls', 5))
plt.title('Kmeans dengan 5 Clutser')
plt.show()
