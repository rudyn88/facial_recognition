import os
import numpy as np
from PIL import Image
from scipy.stats import entropy
import shutil
import csv
import matplotlib.pyplot as plt


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

    if len(p_normalized) < len(q_normalized):
        p_normalized = np.concatenate((p_normalized, np.zeros(len(q_normalized) - len(p_normalized))))
    elif len(p_normalized) > len(q_normalized):
        q_normalized = np.concatenate((q_normalized, np.zeros(len(p_normalized) - len(q_normalized))))
    print(entropy(p_normalized, q_normalized))
    return entropy(p_normalized, q_normalized)

def calculate_image_distribution(dataset_path):
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

# Original dataset path
original_dataset_path = 'C:/Users/lucab/Downloads/fairface-img-margin025-trainval/fairface-img-margin025-trainval/train'

# New dataset path
new_dataset_path = 'C:/Users/lucab/Downloads/UTKOutliers/UTKOutliers/UTKOutliers'

# Threshold to determine outliers
threshold = 0.1

# Calculate image distributions for both datasets
original_distribution = calculate_image_distribution(original_dataset_path)
new_distribution = calculate_image_distribution(new_dataset_path)
'''
# Plot the distributions of the original and new datasets
def plot_distribution(distributions, titles):
    plt.figure()
    for distribution in distributions:
        plt.bar(range(len(distribution)), distribution, alpha=0.5)
    plt.title("Dataset Distributions")
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")
    plt.legend(titles)
    plt.show()

plot_distribution([original_distribution, new_distribution], ["Original Dataset", "New Dataset"])
'''
# Calculate the KL divergence between the two datasets
kl_divergence = calculate_kl_divergence(original_distribution, new_distribution)
print("KL Divergence (Original -> New):", kl_divergence)

# Swap the datasets and calculate the KL divergence again
kl_divergence_flipped = calculate_kl_divergence(new_distribution, original_distribution)
print("KL Divergence (New -> Original):", kl_divergence_flipped)

# Sort the images by their KL divergence in descending order

'''
image_scores = []
image_files = os.listdir(new_dataset_path)

for image_file in image_files:
    image_path = os.path.join(new_dataset_path, image_file)
    image = Image.open(image_path)
    pixels = np.array(image)
    pixel_values, counts = np.unique(pixels.flatten(), return_counts=True)
    image_distribution_normalized = counts / (np.sum(counts) * np.sum(new_distribution))
    kl_divergence = calculate_kl_divergence(image_distribution_normalized, original_distribution)
    image_scores.append((image_file, kl_divergence))

image_scores.sort(key=lambda x: x[1], reverse=True)

# Find the largest 20% of outliers
num_outliers = int(len(image_scores) * 0.2)
largest_outliers = image_scores[:num_outliers]

output_folder = 'C:/Users/aashr/OneDrive/Documents/Research Projects/EmoryREU/UTKOutliers'
os.makedirs(output_folder, exist_ok=True)

# Copy identified outlier images to the output folder
for image_file, _ in largest_outliers:
    source_path = os.path.join(new_dataset_path, image_file)
    destination_path = os.path.join(output_folder, image_file)
    shutil.copy(source_path, destination_path)

csv_path = 'C:/Users/aashr/OneDrive/Documents/Research Projects/EmoryREU/outlier_info.csv'
with open(csv_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Image File', 'KL Divergence'])
    for image_file, kl_divergence in largest_outliers:
        writer.writerow([image_file, kl_divergence])

print("Identified outliers have been copied to:", output_folder)
print("Outlier information has been saved to:", csv_path)
'''
