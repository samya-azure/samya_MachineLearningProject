
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

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
            images.append(img.flatten())  # Flatten to 1D vector
            filenames.append(filename)
    return np.array(images), filenames

# ✅ Your image folder path
base_dir = os.path.dirname(os.path.abspath(__file__))
image_folder = os.path.join(base_dir, '..', 'Images', 'Train_Images')

# Load and preprocess images
data, filenames = load_images_from_folder(image_folder)

# Step 2: Apply PCA to reduce dimensionality
print("Applying PCA to reduce dimensions...")
pca = PCA(n_components=100, random_state=42)
data_pca = pca.fit_transform(data)

# Step 3: Apply K-Means Clustering on PCA features
print("Clustering using K-Means...")
kmeans = KMeans(n_clusters=2, random_state=42)
clusters = kmeans.fit_predict(data_pca)

# Step 4: Show Results
print("\n📂 Cluster Assignments:")
for i, label in enumerate(clusters):
    print(f"Image: {filenames[i]} --> Cluster: {label}")

# Optional: Visualize sample images from each cluster
def plot_images_by_cluster(original_data, clusters, filenames, target_cluster):
    indices = [i for i, c in enumerate(clusters) if c == target_cluster]
    plt.figure(figsize=(10, 3))
    for i, idx in enumerate(indices[:5]):  # Show 5 images from this cluster
        img = original_data[idx].reshape(64, 64)
        plt.subplot(1, 5, i+1)
        plt.imshow(img, cmap='gray')
        plt.title(filenames[idx], fontsize=8)
        plt.axis('off')
    plt.suptitle(f"Cluster {target_cluster}")
    plt.tight_layout()
    plt.show()

# Show images from each cluster
plot_images_by_cluster(data, clusters, filenames, 0)
plot_images_by_cluster(data, clusters, filenames, 1)
