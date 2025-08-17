
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D

# Step 1: Load and preprocess images
def load_images(folder, target_size=(224, 224)):
    images = []
    display_images = []
    filenames = []
    for filename in os.listdir(folder):
        filepath = os.path.join(folder, filename)
        img = cv2.imread(filepath)
        if img is not None:
            img_resized = cv2.resize(img, target_size)
            img_array = img_to_array(img_resized)
            img_preprocessed = preprocess_input(img_array)
            images.append(img_preprocessed)
            display_images.append(cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (64, 64)))  # for display
            filenames.append(filename)
    return np.array(images), np.array(display_images), filenames

# Get absolute paths
base_dir = os.path.dirname(os.path.abspath(__file__))
image_folder = os.path.join(base_dir, '..', 'Images', 'Train_Images')
weights_path = os.path.join(base_dir, '..', 'models', 'mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5')

# Load images
print("🔄 Loading and preprocessing images...")
images, display_images, filenames = load_images(image_folder)

# Load MobileNetV2 from local weights
print("📥 Loading MobileNetV2 model from local weights...")
base_model = MobileNetV2(weights=weights_path, include_top=False, input_shape=(224, 224, 3))
x = GlobalAveragePooling2D()(base_model.output)
model = Model(inputs=base_model.input, outputs=x)

# Extract features
print("📤 Extracting CNN features...")
features = model.predict(images, verbose=1)

# Apply KMeans clustering
print("📊 Clustering images with KMeans...")
kmeans = KMeans(n_clusters=2, random_state=42)
clusters = kmeans.fit_predict(features)

# Output clustering results
print("\n📂 Cluster Assignments:")
for i, label in enumerate(clusters):
    print(f"Image: {filenames[i]} --> Cluster: {label}")

# Optional: Visualize sample images per cluster
def plot_cluster(cluster_id):
    indices = [i for i, c in enumerate(clusters) if c == cluster_id]
    plt.figure(figsize=(10, 3))
    for i, idx in enumerate(indices[:5]):
        img = display_images[idx]
        plt.subplot(1, 5, i+1)
        plt.imshow(img, cmap='gray')
        plt.title(filenames[idx], fontsize=8)
        plt.axis('off')
    plt.suptitle(f"Cluster {cluster_id}")
    plt.tight_layout()
    plt.show()

plot_cluster(0)
plot_cluster(1)

# Optional: Move clustered images to folders (organized output)
def move_images_by_cluster():
    output_root = os.path.join(base_dir, '..', 'Images', 'Clustered')
    for i, label in enumerate(clusters):
        cluster_folder = os.path.join(output_root, f"Cluster_{label}")
        os.makedirs(cluster_folder, exist_ok=True)
        src = os.path.join(image_folder, filenames[i])
        dst = os.path.join(cluster_folder, filenames[i])
        if not os.path.exists(dst):
            cv2.imwrite(dst, cv2.imread(src))
    print("📁 Clustered images saved to 'Images/Clustered/Cluster_0' and 'Cluster_1'")

# Uncomment to enable image moving:
# move_images_by_cluster()
