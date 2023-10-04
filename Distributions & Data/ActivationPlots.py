import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from torchvision.models import resnet18
from PIL import Image
import torch.nn.functional as F
from torch.distributions import kl_divergence
import matplotlib.pyplot as plt
import torch.distributions as D
import numpy as np


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
        # class_label = int(gender)  # just change this to age, gender, or race
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


# Define the main function
def main():
    # Set the paths to your datasets
    dataset1_path = 'C:/Users/lucab/Downloads/fairface-img-margin025-trainval/fairface-img-margin025-trainval/train'
    dataset2_path = 'C:/Users/lucab/Downloads/celebA/celebA'
    dataset3_path = 'C:/Users/lucab/Downloads/UTKFace/UTKFace'

    # Define the transformations for image preprocessing
    transform = transforms.Compose([
        transforms.Resize((100, 100)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Create instances of the datasets
    dataset1 = FacialImageDataset(dataset1_path, transform=transform)
#    dataset2 = FacialImageDataset(dataset2_path, transform=transform)
    dataset3 = FacialImageDataset(dataset3_path, transform=transform)

    # Create data loaders for the datasets
    dataloader1 = DataLoader(dataset1, batch_size=32, shuffle=True)
#    dataloader2 = DataLoader(dataset2, batch_size=32, shuffle=True)
    dataloader3= DataLoader(dataset3, batch_size=32, shuffle=True)

    # Initialize the neural network model
    # num_classes = len(dataset2.classes)
    # print(num_classes)
    num_classes = 2
    model = Net(num_classes)

    # Modify the initialization of the resnet18 model
    # model.resnet = resnet18()

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    activations1 = []
    activations2 = []
    activations3 = []
    activation_features1 = []
    activation_features2 = []
    activation_features3 = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in dataloader1:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Append intermediate activations to the list
            activations1.append(model.resnet.fc.weight.data.clone())
            activation_features1.append(model.resnet.fc.weight.data.clone().flatten().numpy())

        epoch_loss = running_loss / len(dataloader1)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")

    # Save the intermediate activations from dataset2
    model.eval()
#    with torch.no_grad():
#        for images, _ in dataloader2:
#            images = images.to(device)
#            activations2.append(model(images).detach().clone())
#            activation_features2.append(model.resnet.fc.weight.data.clone().flatten().numpy())
    with torch.no_grad():
        for images, _ in dataloader3:
            images = images.to(device)
            activations3.append(model(images).detach().clone())
            activation_features3.append(model.resnet.fc.weight.data.clone().flatten().numpy())

    # After training, you can use the model to calculate the KL divergence or perform other tasks
    print('Moving on')
    activations1 = torch.cat(activations1, dim=0)
#    activations2 = torch.cat(activations2, dim=0)
    activations3 = torch.cat(activations3, dim=0)
    activation_features1 = np.concatenate(activation_features1, axis=0)
#    activation_features2 = np.concatenate(activation_features2, axis=0)
    activation_features3 = np.concatenate(activation_features3, axis=0)

    print('Features found')
    # Convert activations to probability distributions
#    activations1 = F.softmax(activations1, dim=1)
#    activations2 = F.softmax(activations2, dim=1)

    unique_features1, counts1 = np.unique(activation_features1, return_counts=True)
#    unique_features2, counts2 = np.unique(activation_features2, return_counts=True)
    unique_features3, counts3 = np.unique(activation_features3, return_counts=True)
#    print(unique_features1, counts1)
#    print(unique_features2, counts2)
#    print(unique_features3, counts3)

    print('Plotting time')
    # Plot the activation features and their frequencies for each dataset
    plt.figure()
    plt.bar(unique_features1, counts1, color='blue', label='FairFace')
    plt.bar(unique_features3, counts3, color='orange', label='UTKFace')
    plt.xlabel('Activation Feature', fontsize=18)
    plt.ylabel('Frequency', fontsize=18)
    plt.title('Activation Features', fontsize=20)
    plt.legend(fontsize=18)
    plt.savefig('C:/Users/lucab/OneDrive/Desktop/REU Graphs/KL Test 1')

'''
   plt.figure()
    plt.bar(unique_features2, counts2, color='blue', label='CelebA')
    plt.xlabel('Activation Feature', fontsize=16)
    plt.ylabel('Frequency', fontsize=16)
    plt.title('CelebA Activation Features', fontsize=18)
    plt.legend()
    plt.savefig('C:/Users/lucab/OneDrive/Desktop/REU Graphs/KL Test 2')

    plt.figure()
    plt.xlabel('Activation Feature', fontsize=16)
    plt.ylabel('Frequency', fontsize=16)
    plt.title('CelebA Activation Features', fontsize=18)
    plt.legend()
    plt.savefig('C:/Users/lucab/OneDrive/Desktop/REU Graphs/KL Test 3')
# Plot for Dataset 1 Activation Features
'''

# Plot for Dataset 2 Activation Features





if __name__ == '__main__':
    main()
