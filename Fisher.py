import os
from pickletools import read_stringnl_noescape
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from torchvision.models import resnet18
from PIL import Image
import torch.nn.functional as F
import numpy as np
from torch.autograd import grad
from scipy.spatial.distance import directed_hausdorff


# Define the dataset class
class FacialImageDataset(Dataset):
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


# Define the neural network
class Net(nn.Module):
    num_features = 512

    def __init__(self, num_classes):
        super(Net, self).__init__()
        self.resnet = resnet18(pretrained=True)

        self.resnet.fc = nn.Linear(512, self.num_features)
        self.linear = nn.Linear(Net.num_features, num_classes)

    def forward(self, x):
        x = self.resnet(x)
        x = x.view(x.size(0), -1)
        x_copy = x.detach().clone()
        x = self.linear(x_copy)
        return x


# Define the main function
def main():
    dataset1_path = 'C:/Users/lucab/Downloads/UTKFace'
    dataset2_path = 'C:/Users/lucab/Downloads/fairface-img-margin025-trainval/train'

    transform = transforms.Compose([
        transforms.Resize((100, 100)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset1 = FacialImageDataset(dataset1_path, transform=transform)
    dataset2 = FacialImageDataset(dataset2_path, transform=transform)

    dataloader1 = DataLoader(dataset1, batch_size=32, shuffle=True)
    dataloader2 = DataLoader(dataset2, batch_size=32, shuffle=True)

    num_classes = 2
    model = Net(num_classes)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    torch.autograd.set_detect_anomaly(True)

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
            loss.backward(retain_graph=True)
            optimizer.step()

            running_loss += loss.item()

            # Append intermediate gradients to the list
            gradients = grad(loss, model.parameters(), retain_graph=True, allow_unused=True)
            gradients = [g.detach().clone() for g in gradients if g is not None]
            activations1.append(gradients)

        epoch_loss = running_loss / len(dataloader1)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

    model.eval()
    with torch.no_grad():
        for images, _ in dataloader2:
            if images.size(0) != 32:  # Skip the iteration if batch size is not 32
                continue
            images = images.to(device)
            outputs = model(images)
            loss = criterion(outputs, torch.zeros(images.size(0)).long().to(device))  # Compute the loss with dummy labels
            #loss.backward(retain_graph=True)  # Compute gradients for model parameters
            gradients = [param.grad.detach().clone() if param.grad is not None else torch.zeros_like(param) for param in model.parameters()]
            activations2.append(gradients)



    # Calculate the Fisher divergence
    divergence = 0.0
    for grads1, grads2 in zip(activations1, activations2):
        diff = torch.cat([(g1 - g2).flatten() for g1, g2 in zip(grads1, grads2)])
        squared_diff = torch.sum(diff ** 2)
        divergence += squared_diff.item()

    print("Fisher Divergence:", divergence)


if __name__ == '__main__':
    main()
