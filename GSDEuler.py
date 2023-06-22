import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from imblearn.over_sampling import RandomOverSampler
import requests
from io import BytesIO
import zipfile


# Program Goal: Find the optimal settings for GSD using grid search without requiring a local directory --> Ideal for euler/zuber computation
# Define the URL to the UTKFace dataset zip file
dataset_url = 'https://drive.google.com/uc?export=download&id=0BxYys69jI14kYVM3aVhKS1VhRUk'

# Download and extract the dataset
response = requests.get(dataset_url)
zip_file = zipfile.ZipFile(BytesIO(response.content))
zip_file.extractall()
dataset_path = zip_file.namelist()[0]

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

# Oversampling to reduce imbalance
oversampler = RandomOverSampler(random_state=42)
X_pca_oversampled, y_oversampled = oversampler.fit_resample(X_pca, y)

X_train, X_test, y_train, y_test = train_test_split(X_pca_oversampled, y_oversampled, test_size=0.2, random_state=42)

# Define the parameter grid for the SVM classifier
svm_param_grid = {
    'C': [0.1, 1, 10],
    'gamma': [0.1, 1, 10],
    'kernel': ['linear', 'rbf']
}

# Perform grid search for hyperparameter tuning with SVM
svm_grid_search = GridSearchCV(SVC(), svm_param_grid, cv=5)
svm_grid_search.fit(X_train, y_train)

# Retrieve the best hyperparameters
best_svm = svm_grid_search.best_estimator_

# Initialize and train a Random Forest classifier
rf = RandomForestClassifier()
rf.fit(X_train, y_train)

# Initialize and train a Gradient Boosting classifier
gb = GradientBoostingClassifier()
gb.fit(X_train, y_train)

# Make predictions on the testing data
y_pred_svm = best_svm.predict(X_test)
y_pred_rf = rf.predict(X_test)
y_pred_gb = gb.predict(X_test)

# Print classification reports for each classifier
print("SVM Classification Report:")
print(classification_report(y_test, y_pred_svm))

print("Random Forest Classification Report:")
print(classification_report(y_test, y_pred_rf))

print("Gradient Boosting Classification Report:")
print(classification_report(y_test, y_pred_gb))

# Visualize the data with different colors for male and female
male_color = 'blue'
female_color = 'red'
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=(y == 0), cmap=ListedColormap([female_color, male_color]))
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Geometric Sensitivity Decomposition - UTKFace Dataset')
unique_labels = np.unique(y)
legend_labels = [str(label) for label in unique_labels]
plt.legend(handles=scatter.legend_elements()[0], labels=legend_labels, title='Class Labels')
plt.show()
