# importing all packages that may be needed
import csv
import random

from matplotlib import image as mpimg
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
# import tensorflow as tf
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
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from matplotlib.legend_handler import HandlerBase
import matplotlib.patches as mpatches
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve, balanced_accuracy_score, \
    RocCurveDisplay
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import argparse


# Defining a dataset class
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
        # class_label = 0 if int(gender) == 0 else 1
        class_label = int(race)  # just change this to age, gender, or race
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


class FairfaceDataset(Dataset):
    # Initializes dataset by setting root directory and transformation
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.images = self.load_images()[:num_images]

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
        # class_label = 0 if int(gender) == 0 else 1
        class_label = int(race)  # just change this to age, gender, or race
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


# Set the root directory of the UTKFace dataset

# root_dir = 'C:/Users/lucab/Downloads/UTKFace'
# root_dir_fair = 'C:/Users/lucab/Downloads/fairface-img-margin025-trainval/train'
# csv_root_dir = 'C:/Users/lucab/Downloads/fairface-img-margin025-trainval/train/fairface_label_train.csv'

root_dir = '/Users/rudy/Documents/Academic/Research/Facial_Recognition/Data_Sets/UTKFace'
root_dir_fair = '/Users/rudy/Documents/Academic/Research/Facial_Recognition/Data_Sets/outliers'
csv_root_dir = '/Users/rudy/Documents/Academic/Research/Facial_Recognition/Data_Sets/fairface_label_train'

# Defines transformation pipeline by resizing, converting to tensors, and normalizing
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Create an instance of the UTKFaceDataset
utkface_dataset = UTKFaceDataset(root_dir, transform=transform)

# Create a DataLoader for the UTKFace dataset
UTKFace_max = 23706
num_images = UTKFace_max
utkface_loader = torch.utils.data.DataLoader(utkface_dataset, batch_size=4, shuffle=True, num_workers=2)

fairfair_dataset = FairfaceDataset(root_dir_fair, transform=transform)
fairface_loader = torch.utils.data.DataLoader(fairfair_dataset, batch_size=4, num_workers=2)


class Net(nn.Module):
    # Initializes layers of network by defining convolutional layers, max-pooling layers, and fully connected layers with appropriate and output sizes
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 128, 5)
        self.fc1 = nn.Linear(128 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 30)
        self.dp = nn.Dropout(0.5)
        self.fc3 = nn.Linear(30,
                             5)  # UTKFace dataset has 2 classes: male and female, 5 race classes, a lot of age classes

    # defines forward pass and returns an output
    def forward(self, x):
        x = self.pool(F.elu(self.conv1(x)))
        x = self.pool(F.elu(self.conv2(x)))
        x = x.view(-1, 128 * 5 * 5)
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        x = self.dp(x)
        x = self.fc3(x)
        return x

    def get_feature_map(self, x, layer_index):
        # Run the input image through the convolutional layers until the desired layer
        for index in range(layer_index):
            x = F.elu(self.conv1(x))
            x = self.pool(x)
            x = F.elu(self.conv2(x))
            x = self.pool(x)

        # Return the feature map of the desired layer
        return x


# trainset = ImageFolder(root='C:/Users/aashr/OneDrive/Documents/Research Projects/EmoryREU/UTKFace.tar/UTKFace/UTKFace', transform=transform)
# trainloader = DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

# testset = ImageFolder(root='C:/Users/aashr/OneDrive/Documents/Research Projects/EmoryREU/UTKFace.tar/UTKFace/UTKFace', transform=transform)
# testloader = DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

# classes = ('male', 'female')
classes = ('White', 'Black', 'Asian', 'Indian', 'Other')

correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}
net = Net()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.5, 0.9), amsgrad=True)
graph_lossE1 = []
step_pointE1 = []
graph_lossE2, step_pointE2 = [], []
graph_lossE3, step_pointE3 = [], []

fairg, fairr = [], []
image_list = []
sample_dict = {}
header_label = []


# Train
def cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max - lr_min) * 0.5 * (
            1 + np.cos(step / total_steps * np.pi))


if __name__ == '__main__':
    loss_avg = 0.0
    running_loss = 0.0
    i = 0
    for epoch in range(5):
        for in_set, out_set in zip(utkface_loader, fairface_loader):
            data = torch.cat((in_set[0], out_set[0]), 0)
            target = in_set[1]
            outputs = net(data)

            optimizer.zero_grad()
            loss = F.cross_entropy(outputs[:len(in_set[0])], target)
            loss += 0.1 * -(outputs[len(in_set[0]):].mean(1) - torch.logsumexp(outputs[len(in_set[0]):], dim=1)).mean()
            loss.backward()
            optimizer.step()
            i += 1
            running_loss += loss.item()
            # cross-entropy from softmax distribution to uniform distribution

            # exponential moving average
            loss_avg = loss_avg * 0.8 + float(loss) * 0.2

            if i % 200 == 199:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, loss_avg / 200))
                if epoch == 0:
                    graph_lossE1.append(loss_avg / 200)
                    step_pointE1.append(i)

                if epoch == 2:
                    graph_lossE2.append(loss_avg / 200)
                    step_pointE2.append(i)

                if epoch == 4:
                    graph_lossE3.append(loss_avg / 200)
                    step_pointE3.append(i)
                loss_avg = 0.0

    #    for epoch in range(5):
    #        running_loss = 0.0
    #        for i, data in enumerate(utkface_loader, 0):
    #            inputs, labels = data
    #            optimizer.zero_grad()
    #            outputs = net(inputs)
    #            loss = criterion(outputs, labels)
    #            loss.backward()
    #            optimizer.step()
    #
    #            running_loss += loss.item()
    #
    #            if i % 200 == 199:
    #                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 200))
    #                if epoch == 0:
    #                    graph_lossE1.append(running_loss / 200)
    #                    step_pointE1.append(i)
    #
    #                if epoch == 2:
    #                    graph_lossE2.append(running_loss / 200)
    #                    step_pointE2.append(i)
    #
    #                if epoch == 4:
    #                    graph_lossE3.append(running_loss / 200)
    #                    step_pointE3.append(i)
    #                running_loss = 0.0

    print('Finished Training')
    truth = torch.tensor([])
    pred = torch.tensor([])
    oodtruth = torch.tensor([])
    oodpred = torch.tensor([])
    # Test the network on the test data
    correct = 0
    total = 0
    oodcorrect, oodtotal = 0, 0
    with torch.no_grad():
        #       for data in utkface_loader:
        #            images, labels = data
        #            outputs = net(images)
        #            _, predicted = torch.max(outputs.data, 1)
        #            total += labels.size(0)
        #            correct += (predicted == labels).sum().item()
        #            truth = torch.cat((truth, labels), 0)
        #            pred = torch.cat((pred, predicted), 0)

        #            for label, prediction in zip(labels, predicted):
        #                if label == prediction:
        #                    correct_pred[classes[label]] += 1
        #                total_pred[classes[label]] += 1
        #        print('Finished Testing on UTKFace!')

        for data in fairface_loader:
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
        print('Finished Testing on Fairface!')

    print(total_pred)
    print(correct_pred)

    # print accuracy for each class
    plt.figure()
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / (total_pred[classname] + (1 * 10 ^ -11))
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')
        width = 0.3
        plt.bar(classname, float(total_pred[classname]), color='#84c2e8', width=0.2, align='edge', label='Predicted')
        plt.bar(classname, float(correct_count), color='#f86ffc', width=-0.2, align='edge', label='Actual')

    print('Accuracy of the network on the test images: %.2f %%' % (100 * correct / total))
    plt.xlabel("Races")
    plt.ylabel("No. of People")
    plt.legend(['Predicted', 'Actual'])
    plt.show()

    plt.figure
    plt.plot(step_pointE1, graph_lossE1, label='Epoch 1')
    plt.plot(step_pointE2, graph_lossE2, label='Epoch 2')
    plt.plot(step_pointE3, graph_lossE3, label='Epoch 3')
    plt.xlabel("Number of iteration")
    plt.ylabel("Loss")
    plt.title("Loss vs Number of iteration")
    plt.legend()
    plt.show()

    # Create an empty confusion matrix
    num_classes = len(classes)
    conf_matrix = torch.zeros(num_classes, num_classes)

    # Perform testing and calculate the confusion matrix
    for data in fairface_loader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)

        # Update the confusion matrix
        for i in range(labels.size(0)):
            true_label = labels[i]
            predicted_label = predicted[i]
            conf_matrix[true_label, predicted_label] += 1

    # Print the confusion matrix
    print('Confusion Matrix Metrics:')


    def calculate_metrics(confusion_matrix):
        num_classes = confusion_matrix.size(0)
        precision = torch.zeros(num_classes)
        recall = torch.zeros(num_classes)
        accuracy = torch.trace(confusion_matrix) / torch.sum(confusion_matrix)

        for class_idx in range(num_classes):
            true_positives = confusion_matrix[class_idx, class_idx]
            false_positives = torch.sum(confusion_matrix[:, class_idx]) - true_positives
            false_negatives = torch.sum(confusion_matrix[class_idx, :]) - true_positives

            precision[class_idx] = true_positives / ((true_positives + false_positives) + (10 ^ -11))
            recall[class_idx] = true_positives / ((true_positives + false_negatives) + (10 ^ -11))

            # Calculate F1 score
            f1_score = 2 * (precision[class_idx] * recall[class_idx]) / (
                        (precision[class_idx] + recall[class_idx]) + (10 ^ -11))

        return precision, recall, f1_score, accuracy


    precision, recall, f1_score, accuracy = calculate_metrics(conf_matrix)
    print('Confusion Matrix Metrics:')
    print(conf_matrix)
    print("Precision: " + str(precision))
    print("Recall: " + str(recall))
    print("F1 Score: " + str(f1_score))
    print("Accuracy: " + str(accuracy))

    # Test the network on a specific image and obtain the feature map
    specific_image_index = 3  # Index of the specific image in the dataset
    specific_image, _ = utkface_dataset[specific_image_index]  # Get the image and ignore the label
    specific_image = specific_image.unsqueeze(0)  # Add a batch dimension
    layer_index = 1  # Index of the desired layer to visualize

    feature_map = net.get_feature_map(specific_image, layer_index)

    # Print the shape of the feature map
    rows = 8  # Number of rows in the grid
    cols = 128 // rows  # Number of columns in the grid

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))

    # Iterate over the channels and display each channel as an image
    for channel, ax in enumerate(axes.flat):
        # Select the channel
        channel_image = feature_map[0, channel, :, :].detach().numpy()

        # Display the channel as an image
        ax.imshow(channel_image, cmap='gray')
        ax.set_title(f'Channel {channel}')
        ax.axis('off')

    # Adjust the spacing between subplots
    plt.tight_layout()

    # Show the figure with all the feature map images
    plt.show()
    print("Finished Outputting Convolution")
#    correct_pred = 0
#    total_pred = 0

#    for classname, correct_count in correct_pred.items():
#        accuracy = 100 * float(correct_count) / total_pred[classname]
#        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')

#    RocCurveDisplay.from_predictions(truth, pred, color="darkorange")
#    plt.plot([0, 1], [0, 1], "k--", label="chance level (AUC = 0.5)")
#    plt.axis("square")
#    plt.xlabel("False Positive Rate")
#    plt.ylabel("True Positive Rate")
#    plt.title("ROC curve")
#    plt.legend()
#    plt.show()
