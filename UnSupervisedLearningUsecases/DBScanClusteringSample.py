
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# Step 1: Simulate party guests (3 main clusters + scattered guests)
np.random.seed(42)

group1 = np.random.normal(loc=[2, 8], scale=0.4, size=(10, 2))
group2 = np.random.normal(loc=[6, 5], scale=0.4, size=(10, 2))
group3 = np.random.normal(loc=[9, 2], scale=0.4, size=(10, 2))
noise_points = np.array([[1, 1], [4, 9], [10, 10], [3, 4]])  # 🕴 Scattered guests (noise)

# Combine all
X = np.vstack((group1, group2, group3, noise_points))

# Step 2: Scale the data
X_scaled = StandardScaler().fit_transform(X)

# Step 3: Apply DBSCAN with settings that produce border & noise
eps = 0.6           # slightly larger neighborhood
min_samples = 5     # increase density requirement

dbscan = DBSCAN(eps=eps, min_samples=min_samples)
dbscan.fit(X_scaled)

labels = dbscan.labels_
core_samples_mask = np.zeros_like(labels, dtype=bool)
core_samples_mask[dbscan.core_sample_indices_] = True

# Step 4: Count each type of point
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)
n_core = np.sum(core_samples_mask)
n_border = len(labels) - n_core - n_noise

# Step 5: Show clustering summary
print("DBSCAN Results Summary:")
print(f"  ε (eps): {eps}")
print(f"  MinPts: {min_samples}")
print(f"  Total clusters found: {n_clusters}")
print(f"  Core points: {n_core}")
print(f"  Border points: {n_border}")
print(f"  Noise points: {n_noise}")

# Step 6: Plot clusters with point types
unique_labels = set(labels)
plt.figure(figsize=(8, 6))

for i in range(len(X_scaled)):
    label = labels[i]
    point = X_scaled[i]

    # Default: Noise
    color = 'black'
    marker = 'o'
    size = 60

    if label != -1:
        if core_samples_mask[i]:
            color = 'green'   # 🟢 Core
            size = 100
        else:
            color = 'orange'  # 🟠 Border
            size = 60

    plt.plot(point[0], point[1], marker,
             markerfacecolor=color,
             markeredgecolor='k',
             markersize=size/10)

plt.title("DBSCAN Clustering - Core [Green], Border [Orange], Noise [Black]")
plt.xlabel("Spice Preference")
plt.ylabel("Cheese Preference")
plt.grid(True)
plt.tight_layout()
plt.show()
