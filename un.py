# Import necessary libraries
# dataset https://www.kaggle.com/datasets/binovi/wholesale-customers-data-set
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans,DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load the Wholesale Customer Dataset
file_path = '/content/Wholesale customers data.csv'
data = pd.read_csv(file_path)

# Drop the 'Channel' and 'Region' columns, as they are not relevant for clustering
data.drop(['Channel', 'Region'], axis=1, inplace=True)

# Standardize the data to have zero mean and unit variance
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

# Choose the number of clusters (K) using the Elbow method
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(scaled_data)
    inertia.append(kmeans.inertia_)

# Plot the Elbow method graph to select K
plt.plot(range(1, 11), inertia, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of clusters (K)')
plt.ylabel('Inertia')
plt.show()

# Based on the Elbow method, select an appropriate K and perform K-Means clustering
k = 4  # Choose an appropriate value for K based on the Elbow method
kmeans = KMeans(n_clusters=k, random_state=0)
clusters = kmeans.fit_predict(scaled_data)

# Apply PCA for dimensionality reduction (for visualization)
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(scaled_data)

# Visualize the clusters
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=clusters, cmap='viridis')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('K-Means Clustering')
plt.show()



dbscan = DBSCAN(eps=0.4, min_samples=10)
clusters = dbscan.fit_predict(scaled_data)

# Apply PCA for dimensionality reduction (for visualization)
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(scaled_data)

# Visualize the clusters
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=clusters, cmap='viridis')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('DBSCAN Clustering')
plt.show()
