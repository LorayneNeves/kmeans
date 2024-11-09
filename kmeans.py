import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans

iris = load_iris()
X = iris.data 

kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)

print(f"Número de clusters: {kmeans.n_clusters}")
print(f"Classificação final (rótulos dos clusters): {kmeans.labels_}")
print(f"Centróides dos clusters: {kmeans.cluster_centers_}")

plt.figure(figsize=(8, 6))

plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap='viridis', s=50)

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='red', marker='X')

plt.title('KMeans Clustering - Iris Dataset')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')

plt.show()
