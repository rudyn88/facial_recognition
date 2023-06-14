import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import tensorflow as tf
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import Dataset
from torchvision import datasets
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from matplotlib.legend_handler import HandlerBase
import matplotlib.patches as mpatches
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve, balanced_accuracy_score, RocCurveDisplay
import matplotlib.patches as mpatches
import matplotlib.lines as mlines


#state = {k: v for k, v in args._get_kwargs()}


class UTKFaceDataset(Dataset):
    # Initializes dataset by setting root directory and transformation
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.images = self.load_images()

    # reads images files from root directory and reutnrs list of image paths where the file ends with .jpg or .png
    def load_images(self):
        image_files = sorted(os.listdir(self.root))
        images = []
        for filename in image_files:
            if filename.endswith('.jpg') or filename.endswith('.png'):
                image_path = os.path.join(self.root, filename)
                images.append(image_path)
        return images

    def get_class_label(self, filename):
        # Extract the age, gender, and race from the filename
        age, gender, race = filename.split('_')[:3]
        # Assign class label based on gender (0 for male, 1 for female)
        #class_label = 0 if int(gender) == 0 else 1
        class_label = int(race) #just change this to age, gender, or race

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

class Net(nn.Module):
    # Initializes layers of network by defining convolutional layers, max-pooling layers, and fully connected layers with appropriate and output sizes
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 5)  # UTKFace dataset has 2 classes: male and female, 5 race classes, a lot of age classes

    # defines forward pass and returns an output
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x




# Set the root directory of the UTKFace dataset
root_dir = 'C:/Users/lucab/Downloads/UTKFace'
root_dir2 = 'C:/Users/lucab/Downloads/fairface-img-margin025-trainval/train'

# Defines transformation pipeline by resizing, converting to tensors, and normalizing
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Create an instance of the UTKFaceDataset
utkface_dataset = UTKFaceDataset(root_dir, transform=transform)
#External dataset
other_dataset = datasets.CIFAR10('../data', train=True, transform=transform)

# Create a DataLoader for the UTKFace dataset
utkface_loader = torch.utils.data.DataLoader(utkface_dataset, batch_size=4, shuffle=True, num_workers=2)
other_loader = torch.utils.data.DataLoader(other_dataset, batch_size=4, shuffle=True, num_workers=2)

classes = ('white', 'Black', 'Asian', 'Indian', 'Other')
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
running_loss = 0
i = 0

if __name__ == '__main__':
    for epoch in range(2):
        net.train()  # enter train mode
        loss_avg = 0.0

        other_loader.dataset.offset = np.random.randint(len(other_loader.dataset))
        for in_set, out_set in zip(utkface_loader, other_loader):
            data = torch.cat((in_set[0], out_set[0]), 0)
            target = in_set[1]

            x = net(data)

            # backward
            optimizer.zero_grad()

            loss = F.cross_entropy(x[:len(in_set[0])], target)
            # cross-entropy from softmax distribution to uniform distribution
            loss += 0.5 * -(x[len(in_set[0]):].mean(1) - torch.logsumexp(x[len(in_set[0]):], dim=1)).mean()

            loss.backward()
            optimizer.step()
            i += 1
            # exponential moving average
            loss_avg = loss_avg * 0.8 + float(loss) * 0.2
            running_loss += loss_avg
            if i % 2000 == 1999:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print(loss_avg)

#    state['train_loss'] = loss_avg
    print('Finished Training')
    truth = torch.tensor([])
    pred = torch.tensor([])
    # Test the network on the test data
    correct = 0
    total = 0
    with torch.no_grad():
        for data in utkface_loader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            truth = torch.cat((truth, labels), 0)
            pred = torch.cat((pred, predicted), 0)

            for label, prediction in zip(labels, predicted):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1
    print(total_pred)
    print(correct_pred)



    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')
    print('Accuracy of the network on the test images: %.2f %%' % (100 * correct / total))
