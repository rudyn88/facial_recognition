# importing all packages that may be needed
import time
import os
from sklearn import metrics
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.metrics import RocCurveDisplay
from scipy.stats import entropy


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_LAUNCH_BLOCKING"]="1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
        #class_label = 0 if int(gender) == 0 else 1
        class_label = int(gender) #just change this to age, gender, or race
        #class_label = int(age)

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
        class_label = int(gender)  # just change this to age, gender, or race
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
root_dir = 'C:/Users/lucab/Downloads/UTKFace'
root_dir_fair = 'C:/Users/lucab/Downloads/fairface-img-margin025-trainval/train'
csv_root_dir = 'C:/Users/lucab/Downloads/fairface-img-margin025-trainval/train/fairface_label_train.csv'
root_dir_oe = 'C:/Users/lucab/Downloads/OutliersCompiled/outliers'
root_dir_oe_utk = 'C:/Users/lucab/Downloads/UTKOutliers/UTKOutliers'
root_dir_oe_combo = 'C:/Users/lucab/Downloads/CombinedOutliers'
root_nobaby = 'C:/Users/lucab/Downloads/NoBabyUTK'
root_dir_lfw = 'C:/Users/lucab/Downloads/LFWSet/Images'

# Defines transformation pipeline by resizing, converting to tensors, and normalizing
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

batchsize = 16
batchsizeoe = 16

# Create an instance of the UTKFaceDataset
utkface_dataset = UTKFaceDataset(root_dir, transform=transform)
# Create a DataLoader for the UTKFace dataset
utkface_loader = torch.utils.data.DataLoader(utkface_dataset, batch_size=batchsize, shuffle=True, num_workers=2)

fairfair_dataset = UTKFaceDataset(root_dir_fair, transform=transform)
fairface_loader = torch.utils.data.DataLoader(fairfair_dataset, batch_size=batchsize, num_workers=2, shuffle=True)

fairface_outlier_dataset = FairfaceDataset(root_dir_oe, transform=transform)
fairface_outlier_loader = torch.utils.data.DataLoader(fairface_outlier_dataset, batch_size=(batchsizeoe), shuffle=True, num_workers=2)

utkface_outlier_dataset = FairfaceDataset(root_dir_oe_utk, transform=transform)
utkface_outlier_loader = torch.utils.data.DataLoader(utkface_outlier_dataset, batch_size=(batchsizeoe), shuffle=True, num_workers=2)

oodset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
oodloader = torch.utils.data.DataLoader(oodset, batch_size=batchsize, shuffle=True, num_workers=2)

combooutlierdataset = UTKFaceDataset(root_dir_oe_combo,transform=transform)
combooutlierloader = torch.utils.data.DataLoader(combooutlierdataset, batch_size=batchsize, shuffle=True, num_workers=2)

lfw_dataset = UTKFaceDataset(root_dir_lfw, transform=transform)
lfw_loader = torch.utils.data.DataLoader(lfw_dataset, batch_size=batchsize, shuffle=True, num_workers=2)


class Net(nn.Module):
    # Initializes layers of network by defining convolutional layers, max-pooling layers, and fully connected layers with appropriate and output sizes
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 5)
        self.pool = nn.AvgPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 128, 5)
        self.fc1 = nn.Linear(128 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 60)
        self.fc4 = nn.Linear(60, 30)
        self.fc3 = nn.Linear(30, 2)  # UTKFace dataset has 2 classes: male and female, 5 race classes, a lot of age classes

    # defines forward pass and returns an output
    def forward(self, x):
        x = self.pool(F.tanh(self.conv1(x)))
        x = self.pool(F.tanh(self.conv2(x)))
        x = x.view(-1, 128 * 5 * 5)
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = F.tanh(self.fc4(x))
        x = self.fc3(x)
        return x

class BetaNet(nn.Module):
    def __init__(self):
        super(BetaNet, self).__init__()
        self.fc = nn.Linear(10, 10)
    def forward(self, x):
        x = self.fc(x)
        return x

def calculate_image_distribution(dataset_path, batchsize):

    # Initialize an array to store the pixel distributions
    pixel_distribution = np.zeros((256,))
    for pixels in dataset_path:
        pixel_values, counts = np.unique(pixels.flatten(), return_counts=True)
        for idx, data in enumerate(counts):
            pixel_distribution[idx] += data


    return pixel_distribution / (batchsize * np.sum(pixel_distribution))

def calculate_kl_divergence(p, q):
    p_sum = np.sum(p)
    q_sum = np.sum(q)

    if p_sum != 0:
        p_normalized = p / p_sum
    else:
        p_normalized = np.ones_like(p)  # Assign default value of 1 if sum is zero

    if q_sum != 0:
        q_normalized = q / q_sum
    else:
        q_normalized = np.ones_like(q)  # Assign default value of 1 if sum is zero
    p_normalized = p_normalized[p_normalized != 0]
    q_normalized = q_normalized[q_normalized != 0]

    if len(p_normalized) < len(q_normalized):
        p_normalized = np.concatenate((p_normalized, np.ones(len(q_normalized) - len(p_normalized))))
    elif len(p_normalized) > len(q_normalized):
        q_normalized = np.concatenate((q_normalized, np.ones(len(p_normalized) - len(q_normalized))))

    return np.sum(p_normalized * np.log(p_normalized/q_normalized))

def calculate_kl_divergence_fullset(p, q):
    p_sum = np.sum(p)
    q_sum = np.sum(q)

    if p_sum != 0:
        p_normalized = p / p_sum
    else:
        p_normalized = np.ones_like(p)  # Assign default value of 1 if sum is zero

    if q_sum != 0:
        q_normalized = q / q_sum
    else:
        q_normalized = np.ones_like(q)  # Assign default value of 1 if sum is zero

    if len(p_normalized) < len(q_normalized):
        p_normalized = np.concatenate((p_normalized, np.zeros(len(q_normalized) - len(p_normalized))))
    elif len(p_normalized) > len(q_normalized):
        q_normalized = np.concatenate((q_normalized, np.zeros(len(p_normalized) - len(q_normalized))))
    print(entropy(p_normalized, q_normalized))
    return entropy(p_normalized, q_normalized)

def calculate_image_distribution_fullset(dataset_path):
    image_files = os.listdir(dataset_path)
    num_images = len(image_files)

    # Initialize an array to store the pixel distributions
    pixel_distribution = np.zeros((256,))

    for image_file in image_files:
        if image_file.endswith('.jpg') or image_file.endswith('.png'):
            image_path = os.path.join(dataset_path, image_file)
            image = Image.open(image_path)
            pixels = np.array(image)
        # Flatten the image and calculate the histogram
            pixel_values, counts = np.unique(pixels.flatten(), return_counts=True)
            pixel_distribution[pixel_values] += counts

    return pixel_distribution / (num_images * np.sum(pixel_distribution))



# trainset = ImageFolder(root='C:/Users/aashr/OneDrive/Documents/Research Projects/EmoryREU/UTKFace.tar/UTKFace/UTKFace', transform=transform)
# trainloader = DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

# testset = ImageFolder(root='C:/Users/aashr/OneDrive/Documents/Research Projects/EmoryREU/UTKFace.tar/UTKFace/UTKFace', transform=transform)
# testloader = DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

classes = ('male', 'female')
#classes = ('White', 'Black', 'Asian', 'Indian', 'Other')



correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}
net = Net().to(device)
weights = torch.FloatTensor([1, 1.5])


criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.5, 0.9), amsgrad=True)
graph_lossE1 = []
step_pointE1 = []
graph_lossE2, step_pointE2 = [], []
graph_lossE3, step_pointE3 = [], []

true_beta_graph = []
image_list = []
sample_dict = {}
header_label = []
beta_graph = []
inset = torch.tensor([])
distpath = torch.tensor([])
true_beta = 0.5
def beta_step(epoch, beta):
    beta_max = np.tanh(beta)
    return (beta_max) * (beta_max) * (1 - np.cos((epoch + 1) / 20 * np.pi))

# Different (linear or mlp) NN for estimating beta, with inputs as distance betweens distribution
if __name__ == '__main__':
    print(device)
    initialT = time.time()
    loss_avg = 0.0
    running_loss = 0.0
    i = 0
    utkdist = calculate_image_distribution_fullset(root_dir_fair)
    fairdist = calculate_image_distribution_fullset(root_dir_lfw)
    true_beta = np.tanh(calculate_kl_divergence_fullset(utkdist, fairdist))
#    true_beta = beta_step(1, true_beta)
    for epoch in range(20):
        for in_set, out_set in zip(fairface_loader, utkface_outlier_loader):
            data = torch.cat((in_set[0], out_set[0]), 0)
            target = torch.cat((in_set[1], out_set[1]), 0)
#            data = torch.cat((in_set[0], out_set[0]), 0)
#            target = in_set[1]
#            utkdist = calculate_image_distribution(in_set[0], batchsize)
#            fairdist = calculate_image_distribution(out_set[0], batchsizeoe)
#            beta_max = calculate_kl_divergence(utkdist, fairdist)
            outputs = net(data)
            optimizer.zero_grad()
#            beta = beta_step(epoch, beta_max)
#            beta_graph.append(beta)
            loss = F.cross_entropy(outputs[:len(data)], target, weight=weights)
#            loss = F.cross_entropy(outputs[:len(in_set[0])], target, weight=weights)
            loss += true_beta * -(outputs[len(in_set[0]):].mean(1) - torch.logsumexp(outputs[len(in_set[0]):], dim=1)).mean()
            loss.backward()
            optimizer.step()
            i += 1
            running_loss += loss.item()
            loss_avg = loss_avg * 0.8 + float(loss) * 0.2

            if i % 200 == 199:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, loss_avg))
                if epoch == 4:
                    graph_lossE1.append(loss_avg)
                    step_pointE1.append(i)

                if epoch == 9:
                    graph_lossE2.append(loss_avg)
                    step_pointE2.append(i)

                if epoch == 14:
                    graph_lossE3.append(loss_avg)
                    step_pointE3.append(i)
                loss_avg = 0.0
#        true_beta = np.average(beta_graph)
#        true_beta_graph.append(true_beta)



        print('Time since last epoch: %.2f' % (time.time() - initialT), 'seconds')
        initialT = time.time()
#        scheduler.step()
        i = 0


    print('Finished Training')
    truth = torch.tensor([])
    pred = torch.tensor([])

    # Test the network on the test data
    correct = 0
    total = 0
    ROCPrediction = torch.tensor([])

    with torch.no_grad():
        for data in lfw_loader:
            images, labels = data
            outputs = net(images)
            ROCPrediction = torch.cat((ROCPrediction, F.softmax((outputs)[:,1])), 0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            truth = torch.cat((truth, labels), 0)
            pred = torch.cat((pred, predicted), 0)

            for label, prediction in zip(labels, predicted):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1
        print('Finished Testing')


    print(total_pred)
    print(correct_pred)


    # print accuracy for each class
    plt.figure()
    for classname, correct_count in correct_pred.items():
         accuracy = 100 * float(correct_count) / total_pred[classname]
         print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')
         width = 0.3
         plt.bar(classname, float(total_pred[classname]), color='#84c2e8', width=0.2, align='edge', label= 'Predicted')
         plt.bar(classname, float(correct_count), color='#f86ffc', width=-0.2, align='edge', label= 'Actual')
    print('Accuracy of the network on the test images: %.2f %%' % (100 * correct / total))
    plt.xlabel("Races")
    plt.ylabel("No. of People")
    plt.legend(['Predicted', 'Actual'])
    plt.show()

    plt.figure
    plt.plot(step_pointE1, graph_lossE1, label= 'Epoch 1')
    plt.plot(step_pointE2, graph_lossE2, label= 'Epoch 2')
    plt.plot(step_pointE3, graph_lossE3, label= 'Epoch 3')
    plt.xlabel("Number of iteration")
    plt.ylabel("Loss")
    plt.title("Loss vs Number of iteration")
    plt.legend()
    plt.show()

#    plt.figure
#    plt.plot(beta_graph, label= 'Beta')
#    plt.show()

    plt.figure
    plt.plot(true_beta_graph, label= 'Beta')
    plt.show()

#    correct_pred = 0
#    total_pred = 0

#    for classname, correct_count in correct_pred.items():
#        accuracy = 100 * float(correct_count) / total_pred[classname]
#        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')

    RocCurveDisplay.from_predictions(truth, ROCPrediction, color="darkorange")
    plt.plot([0, 1], [0, 1], "k--", label="chance level (AUC = 0.5)")
    plt.axis("square")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC curve")
    plt.legend()
    plt.show()

    confusion_matrix = metrics.confusion_matrix(truth, pred)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=['Male', 'Female'])
    cm_display.plot()
    plt.show()



