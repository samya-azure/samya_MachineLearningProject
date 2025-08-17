# in this experiment, k=5

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Step 1: Sample heights in cm
heights = np.array([
    [145], [148], [150], [152], [154],     # Very Short
    [157], [159], [160], [162], [164],     # Short
    [167], [169], [171], [172], [174],     # Medium
    [177], [179], [181], [183],            # Tall
    [185], [188], [190], [193], [195]      # Very Tall
])

# Step 2: Apply KMeans with k=5
k = 5
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(heights)

# Step 3: Get results
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# Step 4: Display output
print("\nCluster assignments:")
for i, height in enumerate(heights):
    print(f"Height: {height[0]} cm --> Cluster: {labels[i]}")

# Step 5: Visualize the result
plt.figure(figsize=(10, 4))
plt.scatter(heights, np.zeros_like(heights), c=labels, cmap='tab10', s=100)
plt.scatter(centroids, np.zeros_like(centroids), c='red', marker='X', s=200, label='Centroids')
plt.title("K-Means Clustering of People by Height (k=5)")
plt.xlabel("Height (cm)")
plt.yticks([])
plt.grid(True)
plt.legend()
plt.show()
