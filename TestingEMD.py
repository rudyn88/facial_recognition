import os
import numpy as np
from PIL import Image
from torchvision.transforms import ToTensor
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import wasserstein_distance
import random
import gc

class FacialImageDataset:
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.images = self.load_images()

    def load_images(self):
        image_files = sorted(os.listdir(self.root))
        images = []
        classes = set()
        for filename in image_files:
            if filename.endswith('.jpg') or filename.endswith('.png'):
                image_path = os.path.join(self.root, filename)
                images.append(image_path)
                class_label = self.get_class_label(os.path.basename(image_path))
                classes.add(class_label)
        self.classes = sorted(list(classes))
        return images

    def get_class_label(self, filename):
        age, gender, race = filename.split('_')[:3]
        class_label = 0 if int(gender) == 0 else 1
        return class_label

    def __getitem__(self, index):
        image_path = self.images[index]
        image = Image.open(image_path).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)

        class_label = self.get_class_label(os.path.basename(image_path))

        return image, class_label

    def __len__(self):
        return len(self.images)


def extract_features(image):
    return np.array(image).flatten()


def calculate_emd(image_features, training_features):
    emd = wasserstein_distance(image_features, training_features)
    return emd


def train_model(training_images, training_labels):
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(training_images, training_labels)
    return model


def classify_images(test_images, training_images, training_labels):
    predictions = []
    for test_image in test_images:
        test_features = extract_features(test_image)
        min_emd = float('inf')
        predicted_label = None
        for i, training_image in enumerate(training_images):
            training_features = extract_features(training_image)
            emd = calculate_emd(test_features, training_features)
            if emd < min_emd:
                min_emd = emd
                predicted_label = training_labels[i]
        predictions.append(predicted_label)
    return predictions


# Set the paths for your training and testing datasets
training_dataset_path = 'C:/Users/lucab/Downloads/UTKFace'
testing_dataset_path = 'C:/Users/lucab/Downloads/fairface-img-margin025-trainval/train'

# Set the target image size
target_size = (32, 32)

# Load the training images and labels
training_images = []
training_labels = []
transform = ToTensor()
dataset = FacialImageDataset(training_dataset_path, transform=transform)
quarter_size = len(dataset) // 4  # Use only half of the training dataset
selected_indices = random.sample(range(len(dataset)), quarter_size)
for i in selected_indices:
    image, label = dataset[i]
    training_images.append(extract_features(image))
    training_labels.append(label)
    gc.collect()

# Load the testing images
testing_images = []
transform = ToTensor()
dataset = FacialImageDataset(testing_dataset_path, transform=transform)
eighth_size = len(dataset) // 4  # Use only a quarter of the test dataset
selected_indices = random.sample(range(len(dataset)), eighth_size)
for i in selected_indices:
    image, _ = dataset[i]
    testing_images.append(image)
    gc.collect()

# Train the machine learning model
model = train_model(training_images, training_labels)

# Perform image classification using Earth Mover's Distance
predictions = classify_images(testing_images, training_images, training_labels)

# Print the predicted labels for the testing images
print(predictions)
