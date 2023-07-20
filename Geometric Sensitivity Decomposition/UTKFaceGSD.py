import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Define the path to the UTKFace dataset folder
dataset_path = "C:/Users/aashr/OneDrive/Documents/Research Projects/EmoryREU/UTKFace.tar/UTKFace/UTKFace"

# Initialize empty lists for images and labels
images = []
labels = []

# Load and preprocess the images and labels
for image_name in os.listdir(dataset_path):
    image_path = os.path.join(dataset_path, image_name)
    label = int(image_name.split("_")[0])
    
    # Read and resize the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (64, 64))
    
    images.append(image.flatten())
    labels.append(label)

# Convert lists to numpy arrays
X = np.array(images)
y = np.array(labels)

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# Perform PCA
pca = PCA(n_components=0.95)  # Retain 95% of the variance
X_pca = pca.fit_transform(X_scaled)

X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# Create and train the SVM classifier
svm = SVC()
svm.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = svm.predict(X_test)

# Print classification report
print(classification_report(y_test, y_pred))

# Visualize the data
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Geometric Sensitivity Decomposition - UTKFace Dataset')
unique_labels = np.unique(y)
legend_labels = [str(label) for label in unique_labels]
plt.legend(handles=scatter.legend_elements()[0], labels=legend_labels, title='Class Labels')
plt.show()
