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
import shutil


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
    

# Define the main function
def main():
    # Set the paths to your datasets
    dataset1_path = 'C:/Users/lucab/Downloads/UTKFace'
    dataset2_path = 'C:/Users/lucab/Downloads/fairface-img-margin025-trainval/train'

    # Define the transformations for image preprocessing
    transform = transforms.Compose([
        transforms.Resize((100, 100)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Create instances of the datasets
    dataset1 = FacialImageDataset(dataset1_path, transform=transform)
    dataset2 = FacialImageDataset(dataset2_path, transform=transform)

    # Create data loaders for the datasets
    dataloader1 = DataLoader(dataset1, batch_size=32, shuffle=True)
    dataloader2 = DataLoader(dataset2, batch_size=32, shuffle=True)

    # Initialize the neural network model
    #num_classes = len(dataset2.classes)
    num_classes = 2
    model = Net(num_classes)

    # Modify the initialization of the resnet18 model
    #model.resnet = resnet18()

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    activations1 = []
    activations2 = []
    
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

        epoch_loss = running_loss / len(dataloader1)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

    # Save the intermediate activations from dataset2
        model.eval()
        with torch.no_grad():
            for images, _ in dataloader2:
                images = images.to(device)
                activations2.append(model(images).detach().clone())

    # After training, you can use the model to calculate the KL divergence or perform other tasks

    activations1 = torch.cat(activations1, dim=0)
    activations2 = torch.cat(activations2, dim=0)

    # Convert activations to probability distributions
    activations1 = F.softmax(activations1, dim=1)
    activations2 = F.softmax(activations2, dim=1)


    kl_divs = []
    batch_size = 32  # Adjust batch size as per available memory

    for i in range(0, activations2.size(0), batch_size):
        batch_activations1 = activations1.clone()  # Clone the tensor
        batch_activations1 = batch_activations1[i:i + batch_size]

        activations2_batch = activations2[i:i + batch_size]

        if i + batch_size > activations2.size(0):
            batch_size = activations2.size(0) - i
            batch_activations1 = batch_activations1[:batch_size]
            activations2_batch = activations2_batch[:batch_size]

    # Calculate KL divergence
        batch_activations1 = batch_activations1.view(batch_size, 1, -1).expand(-1, num_classes, -1)
        batch_kl_div = F.kl_div(batch_activations1.log_softmax(dim=2), activations2_batch.softmax(dim=1).unsqueeze(2), reduction='batchmean').item()
        kl_divs.append(batch_kl_div)

    kl_div = np.mean(kl_divs)
    print(f"KL Divergence: {kl_div:.4f}")  

    num_outliers = int(len(kl_divs) * 0.2)
    largest_outliers = kl_divs[:num_outliers]

    output_folder = 'C:/Users/lucab/Downloads/FeatureOutliers'
    os.makedirs(output_folder, exist_ok=True)

    # Copy identified outlier images to the output folder
    for image_file, _ in largest_outliers:
        source_path = os.path.join(dataset1_path, image_file)
        destination_path = os.path.join(output_folder, image_file)
        shutil.copy(source_path, destination_path)
    """
    activations1_np = activations1[:, 1].cpu().numpy()
    activations2_np = activations2[:, 1].cpu().numpy()

#    activations1_np = activations1_np.flatten()
#    activations2_np = activations2_np.flatten()

    min_length = min(len(activations1_np), len(activations2_np))
    activations1_np_truncated = activations1_np[:min_length]
    activations2_np_truncated = activations2_np[:min_length]

    fisher_div = np.mean(np.sum(np.sqrt(activations1_np_truncated) * np.log((np.sqrt(activations1_np_truncated) + np.sqrt(activations2_np_truncated)) / 2)))
    print(f"Fisher Divergence: {fisher_div:.4f}")

    plt.hist(activations1_np.flatten(), bins=50, alpha=0.5, label='Dataset 1')
    plt.xlabel('Activation Values After Conversion (Dataset 1)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

    plt.hist(activations2_np.flatten(), bins=50, alpha=0.5, label='Dataset 2')
    plt.xlabel('Activation Values After Conversion (Dataset 2)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()



"""
if __name__ == '__main__':
    main()
