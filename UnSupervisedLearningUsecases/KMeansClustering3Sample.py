# Here we are using 3 clustering groups in party data:
# Cluster 0 → Short, Cluster 1 → Medium, Cluster 2 → Tall
# k=3

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Step 1: Heights of people in cm (Party attendees)
heights = np.array([
    [150], [152], [153], [155], [157],  # Short
    [160], [162], [165], [167], [168],  # Medium
    [170], [172], [175], [178], [180], [183], [185], [188]  # Tall
])

# Step 2: Apply K-Means clustering with 3 clusters
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(heights)

# Step 3: Get cluster labels and centers
labels = kmeans.labels_
centers = kmeans.cluster_centers_

# Step 4: Display clustering result
for i, height in enumerate(heights):
    print(f"Height: {height[0]} cm --> Cluster: {labels[i]}")

# Step 5: Visualize the clusters
plt.figure(figsize=(8, 4))
plt.scatter(heights, np.zeros_like(heights), c=labels, cmap='viridis', s=100)
plt.scatter(centers, np.zeros_like(centers), c='red', s=200, marker='X', label='Centroids')
plt.title("K-Means Clustering of Party Attendees by Height")
plt.xlabel("Height (cm)")
plt.yticks([])
plt.legend()
plt.grid(True)
plt.show()
