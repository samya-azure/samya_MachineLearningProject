
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd

# Step 1: Generate sample data (Height in cm, Weight in kg)
# 3 groups of 20 people each
np.random.seed(42)
heights = np.concatenate((np.random.normal(160, 5, 20),
                          np.random.normal(170, 5, 20),
                          np.random.normal(180, 5, 20)))
weights = np.concatenate((np.random.normal(55, 5, 20),
                          np.random.normal(70, 5, 20),
                          np.random.normal(85, 5, 20)))

data = pd.DataFrame({'Height': heights, 'Weight': weights})
X = data[['Height', 'Weight']].values

# Step 2: Calculate WCSS for different values of k
wcss = []
K = range(1, 11)

print("WCSS values for different number of clusters (k):")
for k in K:
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
    kmeans.fit(X)
    wcss_value = kmeans.inertia_
    wcss.append(wcss_value)
    print(f"k = {k} --> WCSS = {wcss_value:.2f}")

# 📊 Step 3: Plot the Elbow Graph
plt.figure(figsize=(8, 5))
plt.plot(K, wcss, 'bo-', marker='o')
plt.title('Elbow Method to Find Optimal k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('WCSS (Within-Cluster Sum of Squares)')
plt.xticks(K)
plt.grid(True)
plt.axvline(x=3, linestyle='--', color='red', label='Suggested Elbow at k=3')
plt.legend()
plt.tight_layout()
plt.show()
