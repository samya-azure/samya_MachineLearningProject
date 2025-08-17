
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

# Step 1: Sample height data for 20 people
heights = np.array([
    150, 152, 153, 155, 157,   # Group 1: Short
    160, 162, 164, 165, 167,   # Group 2: Medium-Short
    170, 172, 174, 176, 178,   # Group 3: Medium-Tall
    180, 182, 185, 188, 190    # Group 4: Tall
]).reshape(-1, 1)

# Step 2: Perform hierarchical clustering
linked = linkage(heights, method='ward')

# Step 3: Plot the dendrogram
plt.figure(figsize=(12, 6))
dendrogram(linked,
           orientation='top',
           labels=heights.flatten(),
           distance_sort='ascending',
           show_leaf_counts=True)
plt.axhline(y=25, c='red', linestyle='--', label='Cut at distance = 25 (3 clusters)')
plt.title("Dendrogram for 20 People's Heights")
plt.xlabel("Height (cm)")
plt.ylabel("Distance")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Step 4: Form exactly 3 clusters using the 'maxclust' criterion
cluster_labels = fcluster(linked, t=3, criterion='maxclust')

# Step 5: Print height and corresponding cluster
print("Height and Cluster Assignment (3 Clusters):")
for height, cluster in zip(heights.flatten(), cluster_labels):
    print(f"Height: {height} cm --> Cluster: {cluster}")
