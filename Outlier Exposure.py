# importing all packages that may be needed

import os
from sklearn import metrics
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import ExponentialLR, MultiStepLR, CosineAnnealingLR
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve, balanced_accuracy_score, \
    RocCurveDisplay
import pathlib
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import torchvision.transforms as TF
from PIL import Image
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d
from tqdm import tqdm
from pytorch_fid.inception import InceptionV3


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


IMAGE_EXTENSIONS = {'bmp', 'jpg', 'jpeg', 'pgm', 'png', 'ppm',
                    'tif', 'tiff', 'webp'}

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('--dims', type=int, default=2048,
                    choices=list(InceptionV3.BLOCK_INDEX_BY_DIM),
                    help=('Dimensionality of Inception features to use. '
                          'By default, uses pool3 features'))


class ImagePathDataset(torch.utils.data.Dataset):
    def __init__(self, files, transforms=None):
        self.files = files
        self.transforms = transforms

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path = self.files[i]
        img = Image.open(path).convert('RGB')
        if self.transforms is not None:
            img = self.transforms(img)
        return img


def get_activations(files, model, batch_size=50, dims=2048, device='cpu',
                    num_workers=1):
    """Calculates the activations of the pool_3 layer for all images.

    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- batch_size  : Batch size of images for the model to process at once.
                     Make sure that the number of samples is a multiple of
                     the batch size, otherwise some samples are ignored. This
                     behavior is retained to match the original FID score
                     implementation.
    -- dims        : Dimensionality of features returned by Inception
    -- device      : Device to run calculations
    -- num_workers : Number of parallel dataloader workers

    Returns:
    -- A numpy array of dimension (num images, dims) that contains the
       activations of the given tensor when feeding inception with the
       query tensor.
    """
    model.eval()

    if batch_size > len(files):
        print(('Warning: batch size is bigger than the data size. '
               'Setting batch size to data size'))
        batch_size = len(files)

    dataset = ImagePathDataset(files, transforms=TF.ToTensor())
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             drop_last=False,
                                             num_workers=num_workers)

    pred_arr = np.empty((len(files), dims))

    start_idx = 0

    for batch in tqdm(dataloader):
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch)[0]

        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

        pred = pred.squeeze(3).squeeze(2).cpu().numpy()

        pred_arr[start_idx:start_idx + pred.shape[0]] = pred

        start_idx = start_idx + pred.shape[0]

    return pred_arr


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1)
            + np.trace(sigma2) - 2 * tr_covmean)


def calculate_activation_statistics(files, model, batch_size=50, dims=2048,
                                    device='cpu', num_workers=1):
    """Calculation of the statistics used by the FID.
    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- batch_size  : The images numpy array is split into batches with
                     batch size batch_size. A reasonable batch size
                     depends on the hardware.
    -- dims        : Dimensionality of features returned by Inception
    -- device      : Device to run calculations
    -- num_workers : Number of parallel dataloader workers

    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the inception model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the inception model.
    """
    act = get_activations(files, model, batch_size, dims, device, num_workers)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def compute_statistics_of_path(path, model, batch_size, dims, device,
                               num_workers=1):
    if path.endswith('.npz'):
        with np.load(path) as f:
            m, s = f['mu'][:], f['sigma'][:]
    else:
        path = pathlib.Path(path)
        files = sorted([file for ext in IMAGE_EXTENSIONS
                        for file in path.glob('*.{}'.format(ext))])
        m, s = calculate_activation_statistics(files, model, batch_size,
                                               dims, device, num_workers)

    return m, s


def calculate_fid_given_paths(paths, batch_size, device, dims, num_workers=1):
    """Calculates the FID of two paths"""
    for p in paths:
        if not os.path.exists(p):
            raise RuntimeError('Invalid path: %s' % p)

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]

    model = InceptionV3([block_idx]).to(device)

    m1, s1 = compute_statistics_of_path(paths[0], model, batch_size,
                                        dims, device, num_workers)
    m2, s2 = compute_statistics_of_path(paths[1], model, batch_size,
                                        dims, device, num_workers)
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)

    return fid_value


def save_fid_stats(paths, batch_size, device, dims, num_workers=1):
    """Calculates the FID of two paths"""
    if not os.path.exists(paths[0]):
        raise RuntimeError('Invalid path: %s' % paths[0])

    if os.path.exists(paths[1]):
        raise RuntimeError('Existing output file: %s' % paths[1])

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]

    model = InceptionV3([block_idx]).to(device)

    print(f"Saving statistics for {paths[0]}")

    m1, s1 = compute_statistics_of_path(paths[0], model, batch_size,
                                        dims, device, num_workers)

    np.savez_compressed(paths[1], mu=m1, sigma=s1)


# Set the root directory of the UTKFace dataset
# root_dir = 'C:/Users/lucab/Downloads/UTKFace'
# root_dir_fair = 'C:/Users/lucab/Downloads/fairface-img-margin025-trainval/train'
# csv_root_dir = 'C:/Users/lucab/Downloads/fairface-img-margin025-trainval/train/fairface_label_train.csv'
# root_dir_oe = 'C:/Users/lucab/Downloads/OutliersCompiled/outliers'

root_dir = '/Users/rudy/Documents/Academic/Research/Facial_Recognition/Data_Sets/UTKFace'
root_dir_fair = '/Users/rudy/Documents/Academic/Research/Facial_Recognition/Data_Sets/fairface-img-margin025-trainval' \
                '/train '
csv_root_dir = '/Users/rudy/Documents/Academic/Research/Facial_Recognition/Data_Sets/fairface_label_train.csv'
root_dir_oe = '/Users/rudy/Documents/Academic/Research/Facial_Recognition/Data_Sets/outliers'

# Defines transformation pipeline by resizing, converting to tensors, and normalizing
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Create an instance of the UTKFaceDataset
utkface_dataset = UTKFaceDataset(root_dir, transform=transform)

# Create a DataLoader for the UTKFace dataset
utkface_loader = torch.utils.data.DataLoader(utkface_dataset, batch_size=4, shuffle=True, num_workers=2)

fairfair_dataset = FairfaceDataset(root_dir_fair, transform=transform)
fairface_loader = torch.utils.data.DataLoader(fairfair_dataset, batch_size=4, shuffle=False, num_workers=2)
fairface_outlier_dataset = FairfaceDataset(root_dir_oe, transform=transform)
fairface_outlier_loader = torch.utils.data.DataLoader(fairface_outlier_dataset, batch_size=4, shuffle=False,
                                                      num_workers=2)


class Net(nn.Module):
    # Initializes layers of network by defining convolutional layers, max-pooling layers, and fully connected layers with appropriate and output sizes
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 5)
        self.pool = nn.AvgPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 128, 5)
        self.fc1 = nn.Linear(128 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 30)
        self.fc3 = nn.Linear(30,
                             2)  # UTKFace dataset has 2 classes: male and female, 5 race classes, a lot of age classes

    # defines forward pass and returns an output
    def forward(self, x):
        x = self.pool(F.tanh(self.conv1(x)))
        x = self.pool(F.tanh(self.conv2(x)))
        x = x.view(-1, 128 * 5 * 5)
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = self.fc3(x)
        return x


# trainset = ImageFolder(root='C:/Users/aashr/OneDrive/Documents/Research Projects/EmoryREU/UTKFace.tar/UTKFace/UTKFace', transform=transform)
# trainloader = DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

# testset = ImageFolder(root='C:/Users/aashr/OneDrive/Documents/Research Projects/EmoryREU/UTKFace.tar/UTKFace/UTKFace', transform=transform)
# testloader = DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

classes = ('male', 'female')
# classes = ('White', 'Black', 'Asian', 'Indian', 'Other')


correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}
net = Net()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.5, 0.9), amsgrad=True)
graph_lossE1 = []
step_pointE1 = []
graph_lossE2, step_pointE2 = [], []
graph_lossE3, step_pointE3 = [], []
scheduler = MultiStepLR(optimizer, milestones=[10, 20], gamma=0.1)

fairg, fairr = [], []
image_list = []
sample_dict = {}
header_label = []
# Train

# beta = 0.5

beta_max = 0.5


def beta_step(epoch):
    return (beta_max) * 0.5 * (1 - np.cos(epoch / 15 * np.pi))


if __name__ == '__main__':
    loss_avg = 0.0
    running_loss = 0.0
    i = 0
    #    fairface_loader.dataset.offset = np.random.randint(len(fairface_loader.dataset))
    for epoch in range(15):
        for in_set, out_set in zip(utkface_loader, fairface_outlier_loader):
            data = torch.cat((in_set[0], out_set[0]), 0)
            target = in_set[1]
            outputs = net(data)
            optimizer.zero_grad()
            beta = beta_step(epoch)
            loss = F.cross_entropy(outputs[:len(in_set[0])], target)
            loss += beta * -(outputs[len(in_set[0]):].mean(1) - torch.logsumexp(outputs[len(in_set[0]):], dim=1)).mean()
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
        #        scheduler.step()
        i = 0

    print('Finished Training')
    truth = torch.tensor([])
    pred = torch.tensor([])

    # Test the network on the test data
    correct = 0
    total = 0
    with torch.no_grad():

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
        accuracy = 100 * float(correct_count) / total_pred[classname]
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

    #    correct_pred = 0
    #    total_pred = 0

    #    for classname, correct_count in correct_pred.items():
    #        accuracy = 100 * float(correct_count) / total_pred[classname]
    #        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')

    RocCurveDisplay.from_predictions(truth, pred, color="darkorange")
    plt.plot([0, 1], [0, 1], "k--", label="chance level (AUC = 0.5)")
    plt.axis("square")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC curve")
    plt.legend()
    plt.show()

    confusion_matrix = metrics.confusion_matrix(truth, pred)

    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=[False, True])

    cm_display.plot()
    plt.show()
