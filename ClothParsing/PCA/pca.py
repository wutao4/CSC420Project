import os
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import pandas as pd
import sys

sys.path.insert(0, '..')
import header


##################################################################
#      Principal Component Analysis to classify clothing         #
##################################################################

datapath = "../datasets/myntradataset/"
savepath = "./results/"
max_num_img = 2000  # maximum number of images to perform PCA for one class
num_eigen = 5  # number of eigen clothing to compute
size = (80, 60, 3)  # shape of images


# Read data in 'styles.csv' file. Neglect lines with error
df = pd.read_csv(os.path.join(datapath, 'styles.csv'), error_bad_lines=False)
df['image'] = df.apply(lambda row: str(row['id']) + ".jpg", axis=1)
df = df.reset_index(drop=True)

# Clean non-existing files and filter the clothing types we need
problems = []
for idx, row in df.iterrows():
    if row.articleType not in header.TYPES \
            or not os.path.exists(os.path.join(datapath, 'images', row['image'])):
        problems.append(idx)
df.drop(df.index[problems], inplace=True)


# Plot the mean and eigens of a clothing type
def plot_results(avg_cloth, eigen_cloth, cloth_type):
    # rescale values in eigen images for plotting
    new_eigen = []
    for eigen in eigen_cloth:
        new = eigen - np.min(eigen)
        new /= np.max(new)
        new = (new * 255).astype(np.uint8)
        new = cv.cvtColor(new, cv.COLOR_BGR2RGB)
        new_eigen.append(new)
    new_avg = cv.cvtColor(avg_cloth, cv.COLOR_BGR2RGB)
    # Plot the results
    plt.subplot(2, 3, 1), plt.imshow(new_avg), plt.axis('off'), plt.title("mean")
    plt.subplot(2, 3, 2), plt.imshow(new_eigen[0]), plt.axis('off'), plt.title("eigen 1")
    plt.subplot(2, 3, 3), plt.imshow(new_eigen[1]), plt.axis('off'), plt.title("eigen 2")
    plt.subplot(2, 3, 4), plt.imshow(new_eigen[2]), plt.axis('off'), plt.title("eigen 3")
    plt.subplot(2, 3, 5), plt.imshow(new_eigen[3]), plt.axis('off'), plt.title("eigen 4")
    plt.subplot(2, 3, 6), plt.imshow(new_eigen[4]), plt.axis('off'), plt.title("eigen 5")
    plt.suptitle(cloth_type)
    plt.show()


# Perform PCA for each clothing type
for cloth_type in header.TYPES:
    print("------------------------------------------")
    print("Processing clothing type: %s" % cloth_type)

    image_names = list(df.loc[df['articleType'] == cloth_type, 'image'])
    num_img = min(len(image_names), max_num_img)
    print("Number of images used: %d" % num_img)

    # Create data matrix for PCA
    data = np.zeros((num_img, size[0] * size[1] * size[2]), dtype=np.float)
    for i in range(num_img):
        img = cv.imread(os.path.join(datapath, 'images', image_names[i])) / 255
        img = cv.resize(img, (60, 80))
        data[i, :] = img.flatten()

    # Compute the eigenvectors from the stack of images
    print("====> Calculating PCA...")
    mean, eigen_vectors = cv.PCACompute(data, mean=None, maxComponents=num_eigen)
    avg_cloth = mean.reshape(size)
    avg_cloth = (avg_cloth * 255).astype(np.uint8)
    eigen_cloth = []
    for eigen in eigen_vectors:
        eigen_img = eigen.reshape(size)
        eigen_cloth.append(eigen_img)

    # Plot the results
    print("====> Plotting results...")
    plot_results(avg_cloth, eigen_cloth, cloth_type)

    # Save the mean images
    print("====> Saving results...")
    cv.imwrite(os.path.join(savepath, cloth_type + ".jpg"), avg_cloth)
