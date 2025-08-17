
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from kneed import KneeLocator  # Install it via: pip install kneed

# Step 1: Simulate pizza taste preferences of 30 guests
np.random.seed(42)

# Each group has 10 guests
# Group 1: Veg lovers → low spice (3), high cheese (8)
# Group 2: Chicken fans → medium spice (6), medium cheese (5)
# Group 3: Spicy meat lovers → high spice (9), low cheese (2)
# here size(10,2) means, each group has 10 people and each people has 2 preferences like
# spice preference and cheese preference
group1 = np.random.normal(loc=[3, 8], scale=1.0, size=(10, 2))
group2 = np.random.normal(loc=[6, 5], scale=1.0, size=(10, 2))
group3 = np.random.normal(loc=[9, 2], scale=1.0, size=(10, 2))

X = np.vstack((group1, group2, group3))

# Step 2: Calculate WCSS for different k
wcss = []
K = range(1, 11)

print("WCSS values for different number of clusters (k):")
for k in K:
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
    kmeans.fit(X)
    inertia = kmeans.inertia_
    wcss.append(inertia)
    print(f"  k = {k} ➜ WCSS = {inertia:.2f}")

# Step 3: Automatically find the elbow point
knee_locator = KneeLocator(K, wcss, curve='convex', direction='decreasing')
elbow_point = knee_locator.elbow
print(f"\nAutomatically detected optimal number of clusters (elbow point) is: {elbow_point}")

# Step 4: Plot the Elbow Curve
plt.figure(figsize=(8, 5))
plt.plot(K, wcss, 'bo-', marker='o')
plt.axvline(x=elbow_point, color='red', linestyle='--', label=f'Elbow Point (k={elbow_point})')
plt.title('Elbow Method for Optimal Pizza Types (Clusters)')
plt.xlabel('Number of Pizza Types (Clusters)')
plt.ylabel('WCSS (Within-Cluster Sum of Squares)')
plt.xticks(K)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
