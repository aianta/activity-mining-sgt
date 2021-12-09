import pandas as pd
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

cluster_data = pd.read_csv('clustering-data-f79815d4-2a67-419c-8025-d310d59b486f.csv', header=None)

#print(cluster_data)

embeddings = cluster_data.iloc[:,2:-1]
labels = cluster_data.iloc[:,0:1]

print("labels")
print(labels)

print(embeddings)

pca = PCA(n_components=2)
pca.fit(embeddings)


X1 = pca.transform(embeddings)

#print(X1)
#print(type(X1))
print(pca.explained_variance_)
print(np.sum(pca.explained_variance_))

fig = plt.figure(figsize=(5,5))

plt.scatter(X1[:,0],X1[:,1], c=labels[0])
plt.show()
