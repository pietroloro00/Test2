import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import DistanceMetric
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.manifold import TSNE
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import silhouette_score
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform, cdist
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
#import pacmap
import kmedoids
from sklearn.decomposition import FastICA
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import SpectralClustering
from scipy.linalg import issymmetric
from numpy import linalg
# Implementation of Mix Distance function
def mix_distance_matrix(X,Y):

    # Find the number of categorical and numerical features
    # The idea is that categorical variables are encoded,
    # so they are represented by dummy/binary variables,
    # and the sum of the possible values == 1
    #print("Shape of X:", X.shape)
    # Find the number of categorical and numerical features
    nFeatures = X.shape[1]
    print(nFeatures)
    nCat = 0

    for i in range(nFeatures):
        if np.sum(np.unique(X.iloc[:, i])) == 1:
            nCat += 1

    nNum = nFeatures - nCat

    # Compute distances, separately
    DCat_condensed = pdist(X.iloc[:, :nCat], metric='hamming')
    DNum_condensed = pdist(X.iloc[:, nCat:], metric='cityblock')

    # Convert condensed distance matrices to squareform
    DCat = squareform(DCat_condensed)
    DNum = squareform(DNum_condensed)

    wCat = nCat / (nCat + nNum)
    D = wCat * DCat + (1 - wCat) * DNum

    return D

def mix_distance(X, Y):
    nFeatures = X.shape[0]

    # Automatically determine the number of categorical features
    nCat = np.sum([len(np.unique(X[i])) == 1 for i in range(nFeatures)])
    print(nCat)

    nNum = nFeatures - nCat

    DCat = 0
    for i in range(nCat):
        if X[i] != Y[i]:
            DCat += 1

    DCat = DCat / nCat

    DNum = 0
    for i in range(nNum):
        DNum = DNum + np.abs(X[i] - Y[i])

    wCat = nCat / (nCat + nNum)
    D = wCat * DCat + (1 - wCat) * DNum

    return D

# Import and visualize dataset

data = pd.read_excel('BankClients.xlsx')
print(data.head())

# Distinguish between categorical and numerical data

# Numerical Features, delete first col = id

num_columns = ['Age', 'FamilySize','Income','Wealth','Debt','FinEdu', 'ESG','Digital','BankFriend','LifeStyle','Luxury','Saving']
# data.iloc[:, 1:].select_dtypes(include='number').columns
X_num = data[num_columns].values


# Categorical Features

cat_columns = ['Gender', 'Job', 'Area', 'CitySize', 'Investments']
X_cat = data[cat_columns].values

# Standardize numerical features

scaler = MinMaxScaler()
X_num_scaled = scaler.fit_transform(X_num)

# Transform categorical features into dummy variables
# Encode categorical features into dummy variables if needed

encoder = OneHotEncoder(drop='first')
X_cat_encoded = encoder.fit_transform(X_cat)

# Combine rescaled numerical and encoded categorical features

X_combined = pd.concat([pd.DataFrame(X_num_scaled, columns=num_columns),
                           pd.DataFrame(X_cat_encoded.toarray(), columns=encoder.get_feature_names_out(cat_columns))],
                          axis=1)

# Take a random subsample to make easier computation
# Set the seed for reproducibility

nSubSample = 1950
np.random.seed(42)
randRows = np.random.randint(X_combined.shape[0], size=nSubSample)
#np.random.permutation(X_combined.shape[0])[:nSubSample]
#X = X_combined.iloc[randRows, :]
X = X_combined.iloc[randRows, :].copy()
#mix_dist = DistanceMetric.get_metric(metric='pyfunc', func=mix_distance)
#distances = ['cityblock', 'cosine', 'euclidean', mix_dist]

# 2D t-SNE

distances = ['cityblock', 'cosine', 'euclidean', mix_distance]
distance_names = ['Cityblock', 'Cosine', 'Euclidean', 'Mix Distance']

for i in range(4):
    plt.subplot(2, 2, i + 1)

    tsne = TSNE(n_components=2, method='exact', metric=distances[i], random_state=42)
    Y = tsne.fit_transform(X)

    plt.scatter(Y[:, 0], Y[:, 1], s=5)
    plt.title(f'Distance: {distance_names[i]}')

plt.tight_layout()
plt.show()

# # 3D t-SNE
#
# tsne = TSNE(n_components=3, method='exact', metric=mix_distance, random_state=42)
# Y_3D = tsne.fit_transform(X)
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# scatter = ax.scatter(Y_3D[:, 0], Y_3D[:, 1], Y_3D[:, 2], c=[0, .75, .75], edgecolors='k')
#
# ax.set_title('3-D Embedding')
# ax.view_init(-30, 15)
#
# plt.show()
#
# # Perform PCA
#
# pca = PCA(n_components=3)
# pca.fit(X)
#
# # Transform the data
#
# score = pca.transform(X)
#
# # Access the components and explained variance
#
# coeff = pca.components_
# latent = pca.explained_variance_
#
#
# # 2-D Embedding (2D visualization)
# plt.figure()
# plt.scatter(score[:, 0], score[:, 1], color='r', s=4)
# plt.title('2-D Embedding with PCA')
#
# # 3-D Embedding (3D visualization)
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(score[:, 0], score[:, 1], score[:, 2], marker='o', edgecolors='c', facecolors=(0, 0.25, 0.85))
# ax.set_title('3-D Embedding')
# ax.view_init(-30, 15)
#
# plt.show()
#
# # Train the model
#
# # 2 dimensions
#
# ica = FastICA(n_components=2)
# S_ = ica.fit_transform(X)  # estimated independent sources
#
# fig, ax = plt.subplots(1, 1, figsize=(6, 6))
#
# ax.scatter(S_[:, 0], S_[:, 1], s=20, alpha=0.8)
# ax.set_title('Scatter plot of Estimated Independent Sources')
# ax.set_xlabel('Independent Source 1')
# ax.set_ylabel('Independent Source 2')
#
# plt.show()
#
# # 3 dimensions
#
# ica = FastICA(n_components=3)  # Change to n_components=3 for 3D
# S_ = ica.fit_transform(X)  # estimated independent sources
#
# # Plot the estimated independent sources in 3D
#
# fig = plt.figure(figsize=(8, 8))
# ax = fig.add_subplot(111, projection='3d')
#
# ax.scatter(S_[:, 0], S_[:, 1], S_[:, 2], s=20, alpha=0.8)
# ax.set_title('Scatter plot of Estimated Independent Sources (3D)')
# ax.set_xlabel('Independent Source 1')
# ax.set_ylabel('Independent Source 2')
# ax.set_zlabel('Independent Source 3')
#
# plt.show()
#
# # Train the model
#
# pacmap = pacmap.PaCMAP(n_components=2, n_neighbors=None)
# X_transformed = pacmap.fit_transform(X, init="pca")
# cluster_labels = np.random.randint(0, 3, size=1950)
#
# # Visualize the embedding with different colors for each cluster
#
# fig, ax = plt.subplots(1, 1, figsize=(6, 6))
#
# # Scatter plot with different colors for each cluster
#
# scatter = ax.scatter(X_transformed[:, 0], X_transformed[:, 1], c=cluster_labels, cmap="Spectral", s=20, alpha=0.8)
# cbar = plt.colorbar(scatter)
# plt.show()
#
# # Convert DataFrame to NumPy array
#
# X_np = X.values
# cluster_values = [3, 4, 5, 6]
# cluster_idx = []
#
# # Perform KMedoids clustering
#
# for n_clusters in cluster_values:
#     kmedoids_instance = kmedoids.KMedoids(n_clusters=n_clusters, init='random', random_state=42, metric=mix_distance)
#     IDX = kmedoids_instance.fit_predict(X_np)
#
#     # Save the cluster indexes
#     cluster_idx.append(IDX)
#
#     # Plotting
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     scatter_opts = {'edgecolors': 'k', 's': 50}
#
#     # Extracting columns from DataFrame for plotting
#     col_0, col_1, col_2 = X.columns[0], X.columns[1], X.columns[2]
#     ax.scatter(X_np[IDX == 0, 0], X_np[IDX == 0, 1], X_np[IDX == 0, 2], facecolors=[0, 0, 1], **scatter_opts, label='Cluster 1')
#     ax.scatter(X_np[IDX == 1, 0], X_np[IDX == 1, 1], X_np[IDX == 1, 2], facecolors=[0.3010, 0.7450, 0.9330], **scatter_opts, label='Cluster 2')
#     ax.scatter(X_np[IDX == 2, 0], X_np[IDX == 2, 1], X_np[IDX == 2, 2], facecolors=[0.4660, 0.6740, 0.1880], **scatter_opts, label='Cluster 3')
#     ax.set_title('3-D Embedding of  clusters')
#     ax.legend()
#     plt.show()
#
# # Evluation of clusters:
#
# calinski_scores = []
# davies_scores = []
# silhouette_scores = []
#
# for idx in cluster_idx:
#     score_calinski = calinski_harabasz_score(X, idx)
#     calinski_scores.append(score_calinski)
#     score_davies = davies_bouldin_score(X,idx)
#     davies_scores.append(score_davies)
#     score_silhouette = silhouette_score(X, idx)
#     silhouette_scores.append(score_silhouette)
#
# # Plotting
#
# fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))
#
# ax1.plot(range(3, 7), calinski_scores, marker='o', linestyle='-', color='b')
# ax1.set_xlabel('Number of Clusters')
# ax1.set_ylabel('Calinski-Harabasz Score')
# ax1.set_title('Calinski-Harabasz Criterion Evaluation')
#
# ax2.plot(range(3, 7), davies_scores, marker='o', linestyle='-', color='r')
# ax2.set_xlabel('Number of Clusters')
# ax2.set_ylabel('Davies-Bouldin Score')
# ax2.set_title('Davies-Bouldin Criterion Evaluation')
#
# ax3.plot(range(3, 7), silhouette_scores, marker='o', linestyle='-', color='g')
# ax3.set_xlabel('Number of Clusters')
# ax3.set_ylabel('Silhouette Score')
# ax3.set_title('Silhouette Criterion Evaluation')
#
# plt.tight_layout()
# plt.show()
#
# ## DB-SCAN ##
#
# minpts = 100
#
# Distance = mix_distance_matrix(X,X)
# print(Distance.shape)
#
# #orderedDistance =Distance.sort()
# orderedDistance = np.sort(Distance, axis=1)
# #orderedDistance = np.sort(Distance, 'ascend')
# minpts_sorted_row = np.sort(orderedDistance[minpts, :])
#
# # Plot the sorted values
# plt.plot(minpts_sorted_row)
# plt.show()
# epsilon = 2.5  # Adjust as needed
#
# # Perform DBSCAN clustering
# dbscan = DBSCAN(eps=epsilon, min_samples=minpts, metric='precomputed')
# DBSCANidx = dbscan.fit_predict(Distance)
#
# # Extract core points
# corepts = np.where(DBSCANidx != -1)[0]
#
# # Get the number of DBSCAN clusters
# nDBSCANCluster = np.unique(DBSCANidx)
# print(nDBSCANCluster)
# print(corepts)
# print(DBSCANidx )
# # Perform TSNE on DBSCAN with 2 components
#
# tsne = TSNE(n_components=2, method='exact', metric=mix_distance, random_state=42)
# Y = tsne.fit_transform(X)
#
# # Plot each DBSCAN cluster with random colors
#
# plt.figure()
# for i in range(1, nDBSCANCluster.shape[0] + 1):
#     pointer = nDBSCANCluster[i - 1]
#     cluster_points = Y[DBSCANidx == pointer]
#
#     plt.scatter(cluster_points[:, 0], cluster_points[:, 1],
#                 marker='o', edgecolors='k',
#                 facecolors=np.random.rand(3,))
#
# plt.title('2-D t-SNE embedding of DBSCAN clusters')
# plt.legend()
# plt.show()
#
# # Perform TSNE on DBSCAN with 3 components
#
# tsne = TSNE(n_components=3, method='exact', metric=mix_distance, random_state=42)
# Y = tsne.fit_transform(X)
#
# # Create a 3D scatter plot for each DBSCAN cluster with random colors
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
#
# for i in range(1, nDBSCANCluster.shape[0] + 1):
#     pointer = nDBSCANCluster[i - 1]
#     cluster_points = Y[DBSCANidx == pointer]
#
#     ax.scatter(cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2],
#                marker='o', edgecolors='k',
#                facecolors=np.random.rand(3,))
#
# ax.set_title('3-D t-SNE embedding of DBSCAN clusters')
# ax.legend()
# plt.show()
#
# Distance = mix_distance_matrix(X,X)
#
# # Distance= squareform(pdist(X, metric='euclidean'))
# Similarity = np.exp(-np.square(Distance))  # Kernel transformation
# print(Similarity.shape)
# is_symmetric = issymmetric(Similarity)
# print(is_symmetric)
#
# nCluster = 3
# #spectral = SpectralClustering(n_clusters=nCluster, affinity='precomputed', n_init=100, random_state=42, eigen_solver='arpack', assign_labels='discretize')
# #SpectralIdx = spectral.fit(Similarity)
#
# spectral_cluster = SpectralClustering(n_clusters=nCluster, affinity='precomputed', assign_labels='kmeans', n_init=100)
#
# # Fit the spectral clustering algorithm to your data
# labels = spectral_cluster.fit_predict(Similarity)
#
# # SpectralIdx represents the cluster indices for each data point
# SpectralIdx = labels
#
# # Compute Laplacian
# #Laplacian = np.eye(len(Similarity)) - Similarity
# W = np.eye(len(Similarity))
# #for i in range(len(Similarity)):
#  # W[i,i] = np.sum(Similarity[:,i])
# #Laplacian = W - Similarity
# W = np.diag(np.sum(Similarity, axis=1))
#
# # Compute the Laplacian matrix
# Laplacian = W - Similarity
# # Compute eigenvalues and eigenvectors
# D, V = np.linalg.eigh(Laplacian)
#
# # Sort eigenvalues and eigenvectors
# idx = D.argsort()
# D = D[idx]
# V = V[:, idx]
#
# # Retain only the first 3 eigenvalues and corresponding eigenvectors
# D = D[:nCluster]
# V = V[:, :nCluster]
#
# print(V)
# print(D)
# print(SpectralIdx.shape)
#
# plt.figure()
# idx=[0,1,2]
# plt.bar(idx, D)
# plt.title('Eigenvalues')
# plt.show()

## NEAREST NEIGHNORS ##
neigh = NearestNeighbors(n_neighbors=2, radius=0.4)
neigh.fit(X)

# Find nearest neighbors
distances, indices = neigh.kneighbors(X)

# Plot the results
fig, ax = plt.subplots(figsize=(8, 8))

# Plot data points
ax.scatter(X['Debt'], X['Income'], s=20, alpha=0.8, label='Data Points')

# Plot connections between nearest neighbors
for i in range(len(X)):
    neighbor_indices = indices[i, 1:]  # Exclude the point itself
    neighbor_points = X.iloc[neighbor_indices]
    for _, neighbor_point in neighbor_points.iterrows():
        ax.plot([X.loc[i, 'Debt'], neighbor_point['Debt']],
                [X.loc[i, 'Income'], neighbor_point['Income']],
                color='gray', linestyle='-', linewidth=0.5)

ax.set_title('Nearest Neighbor Search Results')
ax.set_xlabel('Debt')
ax.set_ylabel('Income')

plt.legend()
plt.show()