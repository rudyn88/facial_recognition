import os
import shutil
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from torchvision.models import resnet18
from PIL import Image
import torch.nn as nn
import numpy as np

# Set the device to run the model on
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define data transformations (you can modify these if needed)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


# Define the dataset class
class FacialImageDataset(Dataset):
        # Initializes dataset by setting root directory and transformation
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.images = self.load_images()

    # reads images files from root directory and reutnrs list of image paths where the file ends with .jpg or .png
    def load_images(self):
        image_files = sorted(os.listdir(self.root))
        images = []
        classes = set()
        for filename in image_files:
            if filename.endswith('.jpg') or filename.endswith('.png'):
                image_path = os.path.join(self.root, filename)
                images.append(image_path)
                class_label = self.get_class_label(os.path.basename(image_path))
                classes.add(class_label)  # New line
        self.classes = sorted(list(classes))
        return images

    def get_class_label(self, filename):
        # Extract the age, gender, and race from the filename
        age, gender, race = filename.split('_')[:3]
        # Assign class label based on gender (0 for male, 1 for female)
        class_label = 0 if int(gender) == 0 else 1
        #class_label = int(gender)  # just change this to age, gender, or race
        # class_label = int(age)

        return class_label

    # Indexes dataset and applies transformation and returns image and tuple
    def __getitem__(self, index):
        image_path = self.images[index]
        image = Image.open(image_path).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)

        class_label = self.get_class_label(os.path.basename(image_path))

        return image, class_label

    # returns total number of images
    def __len__(self):
        return len(self.images)
        
# Define the neural network
class Net(nn.Module):
    num_features = 512
    def __init__(self, num_classes):
        super(Net, self).__init__()
        self.resnet = resnet18(pretrained=True)
        
        self.resnet.fc = nn.Linear(512, self.num_features)
        self.linear = nn.Linear(self.num_features, num_classes)

    def forward(self, x):
        x = self.resnet(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x

# Initialize datasets and data loaders
dataset1 = FacialImageDataset(root='C:/Users/lucab/Downloads/UTKFace/UTKFace', transform=transform)
dataset2 = FacialImageDataset(root='C:/Users/lucab/Downloads/fairface-img-margin025-trainval/fairface-img-margin025-trainval/train', transform=transform)

dataloader1 = DataLoader(dataset1, batch_size=32, shuffle=False)
dataloader2 = DataLoader(dataset2, batch_size=32, shuffle=False)

# Initialize the neural network
num_classes = 2  # Assuming binary classification
model = Net(num_classes).to(device)
model.eval()  # Set the model to evaluation mode

# Function to extract activation features
def extract_features(dataset, dataloader):
    features = []
    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)
            outputs = model(images)
            features.append(outputs)
    features = torch.cat(features, dim=0)
    return features

# Extract activation features for both datasets
features1 = extract_features(dataset1, dataloader1)
features2 = extract_features(dataset2, dataloader2)

# Calculate the difference in activation features
feature_diff = torch.rot90(torch.abs(features1 - features2.mean(dim=0)))
print(feature_diff)
print(torch.abs(features1 - features2.mean(dim=0)))

# Calculate the top 20% outliers based on feature difference
num_outliers = int(len(dataset1) * 0.20)
print(len(feature_diff), '\n', num_outliers)
top_outliers_indices = torch.topk(feature_diff, num_outliers, largest=True).indices.cpu().numpy()
print(top_outliers_indices)
# Create the ActivationImages folder if it doesn't exist
output_folder = 'C:/Users/lucab/Downloads/ActivationImages'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

i=0
# Copy the outlier images to the ActivationImages folder


for idx in top_outliers_indices:
    print(idx)
    for i in idx:
        print(i)
        image_path = dataset1.images[i]
        filename = os.path.basename(image_path)
        shutil.copy(image_path, os.path.join(output_folder, filename))


print(f"{num_outliers} outlier images copied to {output_folder}.")
