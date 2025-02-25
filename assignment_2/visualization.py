import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import cv2

def visualize_input_and_prediction(input_path, prediction_path):
    input_array = nib.load(input_path).get_fdata()
    input_array = 255 * (input_array - np.min(input_array)) / np.max(input_array)
    prediction_array = nib.load(prediction_path).get_fdata()
    prediction_array = 255 * (prediction_array - np.min(prediction_array)) / np.max(prediction_array)

    # Create two subplots and unpack the output array immediately
    f, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(input_array.squeeze())
    ax2.imshow(prediction_array.squeeze())
    plt.show()

def show_data(dataset,index,n_images_display = 5, message = 'Raw data'):
    fig, ax = plt.subplots(1, n_images_display, figsize=(20, 5))
    for i in range(n_images_display):
        image, label = dataset[index + i]
        if np.shape(image)[0] == 3:
            image = np.einsum('abc->bca',image)
        ax[i].imshow(np.uint8(image))
        # ax[i].imshow(np.uint8(np.transpose(image, (1, 2, 0))));
        ax[i].set_title(f'Cancer : {"Yes" if label else "No"}')
        ax[i].axis('off')
    plt.suptitle(
        f'{message} of Skin Cancer Images [Indices: {index} - {index + n_images_display} - Images Shape: {dataset[0][0].shape}]');
    plt.show()
    return fig


def show_data_Unet(dataset,index,n_images_display = 5):
    # @title Visualizing Images { run: 'auto'}
    fig, ax = plt.subplots(1, n_images_display, figsize=(20, 5))
    for i in range(n_images_display):
        sample = dataset[index + i]
        image, mask = sample['image'], sample['mask']
        image, mask = np.uint8(image), np.uint8(mask)
        if np.shape(image)[0] == 3:
            image = np.einsum('abc->bca', image)
        if np.shape(mask)[0] == 1:
            mask = np.einsum('abc->bca', mask)
        edge = cv2.Canny(mask * 255, 100, 200)
        image[edge > 0] = (0, 255, 0)
        ax[i].imshow(image)
        ax[i].axis('off')
    plt.suptitle(
        f"Skin Cancer Images [Indices: {index} - {index + n_images_display} - Images Shape: {dataset[0]['image'].shape}]");
    plt.show()


def show_data_logger(dataset,index,pred,n_images_display = 5, message = 'Raw data'):
    fig, ax = plt.subplots(1, n_images_display, figsize=(50, 20))
    image, label = dataset
    image = image.to('cpu')
    label = label.to('cpu')
    if np.shape(image)[1] == 3:
        image = np.einsum('dabc->dbca', image)
    for i in range(n_images_display):
        ax[i].imshow(np.uint8(image[i]))
        # ax[i].imshow(np.uint8(np.transpose(image, (1, 2, 0))));
        ax[i].set_title(f'Cancer : {"Yes" if label[i] else "No"}, predict:{pred[i]}')
        ax[i].axis('off')
    return fig

def show_data_logger_Unet(dataset,index,pred,n_images_display = 5,message = 'Raw data'):
    # @title Visualizing Images { run: 'auto'}
    fig, ax = plt.subplots(1, n_images_display, figsize=(50, 20))
    image, mask = dataset['image'].cpu(), dataset['mask'].cpu()
    image, mask = np.uint8(image), np.uint8(mask)
    pred = pred.cpu()
    pred = np.uint8(pred)
    if np.shape(image)[1] == 3:
        image = np.einsum('dabc->dbca', image)
    if np.shape(mask)[1] == 1:
        mask = np.einsum('dabc->dbca', mask)
    if np.shape(pred)[1] == 1:
        pred = np.einsum('dabc->dbca', pred)
    for i in range(n_images_display):
        image2=image[i]
        edge = cv2.Canny(mask[i] * 255, 100, 200)
        image2[edge > 0] = (0, 255, 0)
        edgepred = cv2.Canny(pred[i] * 255, 100, 200)
        image2[edgepred > 0] = (0, 0, 255)
        ax[i].imshow(image2)
        ax[i].axis('off')
    return fig
