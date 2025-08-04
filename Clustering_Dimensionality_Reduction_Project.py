# Week 02 - Clustering & Dimensionality Reduction in ML
# Mall Customer Segmentation Project

# Step A: Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score

# %matplotlib inline (Uncomment if using Jupyter)
# %matplotlib inline

# Step B: Load Dataset
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/mall_customers.csv"
df = pd.read_csv(url)
print("First 5 rows:")
print(df.head())

# Step C: Preprocessing
df.drop_duplicates(inplace=True)
df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})  # Encode Gender
X = df.drop('CustomerID', axis=1)

# Standardize Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step D: Dimensionality Reduction using PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print(f"Explained variance by PCA components: {pca.explained_variance_ratio_}")

# Scatter plot after PCA
plt.figure(figsize=(8,6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.6)
plt.title("PCA - 2D Projection of Customers")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()

# Step E: Clustering
# KMeans Clustering
kmeans = KMeans(n_clusters=5, random_state=42)
labels_kmeans = kmeans.fit_predict(X_pca)

# DBSCAN Clustering
dbscan = DBSCAN(eps=0.5, min_samples=5)
labels_dbscan = dbscan.fit_predict(X_pca)

# Step F: Evaluation
print("KMeans Silhouette Score:", silhouette_score(X_pca, labels_kmeans))
print("KMeans Davies-Bouldin Score:", davies_bouldin_score(X_pca, labels_kmeans))

# DBSCAN may include -1 as noise
if len(set(labels_dbscan)) > 1:
    print("DBSCAN Silhouette Score:", silhouette_score(X_pca, labels_dbscan))
    print("DBSCAN Davies-Bouldin Score:", davies_bouldin_score(X_pca, labels_dbscan))
else:
    print("DBSCAN failed to form multiple clusters.")

# Step G: Visualize Clusters
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=labels_kmeans, palette='Set1')
plt.title("KMeans Clusters")

plt.subplot(1, 2, 2)
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=labels_dbscan, palette='Set2')
plt.title("DBSCAN Clusters")

plt.tight_layout()
plt.show()
