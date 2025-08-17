
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Step 1: Load Images and Preprocess
def load_images_from_folder(folder, img_size=(64, 64)):
    images = []
    filenames = []
    for filename in os.listdir(folder):
        filepath = os.path.join(folder, filename)
        img = cv2.imread(filepath)
        if img is not None:
            img = cv2.resize(img, img_size)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
            images.append(img.flatten())  # Flatten image
            filenames.append(filename)
    return np.array(images), filenames

# 🔧 Set the correct relative path from this script to Train_Images
base_dir = os.path.dirname(os.path.abspath(__file__))
image_folder = os.path.join(base_dir, '..', 'Images', 'Train_Images')

data, filenames = load_images_from_folder(image_folder)

# Step 2: Apply K-Means Clustering
k = 2  # We assume 2 clusters: dogs and cats
kmeans = KMeans(n_clusters=k, random_state=42)
clusters = kmeans.fit_predict(data)

# Step 3: Show Results
for i, label in enumerate(clusters):
    print(f"Image: {filenames[i]} --> Cluster: {label}")

# Optional: Visualize images in each cluster
def plot_images_by_cluster(images, clusters, filenames, target_cluster):
    indices = [i for i, c in enumerate(clusters) if c == target_cluster]
    plt.figure(figsize=(10, 3))
    for i, idx in enumerate(indices[:5]):  # Show first 5 images in this cluster
        img = images[idx].reshape(64, 64)
        plt.subplot(1, 5, i+1)
        plt.imshow(img, cmap='gray')
        plt.title(filenames[idx])
        plt.axis('off')
    plt.suptitle(f"Cluster {target_cluster}")
    plt.show()

# Plot for both clusters
plot_images_by_cluster(data, clusters, filenames, 0)
plot_images_by_cluster(data, clusters, filenames, 1)
