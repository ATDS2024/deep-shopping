'''
This python file is responsible for the image processing
'''

import cv2
import numpy as np
import pandas as pd
from hyper_parameters import get_arguments

args = get_arguments()

shuffle = True
localization = args.is_localization
imageNet_mean_pixel = [103.939, 116.799, 123.68]
global_std = 68.76

IMG_ROWS = 64
IMG_COLS = 64


def get_image(path, x1, y1, x2, y2):
    '''
    :param path: image path
    :param x1: the upper left and lower right coordinates to localize the apparels
    :param y1:
    :param x2:
    :param y2:
    :return: a numpy array with dimensions [img_row, img_col, img_depth]
    '''
    img = cv2.imread(path)
    if img is None or img.size == 0:  # More robust check here
        img = np.zeros((IMG_ROWS, IMG_COLS, 3), dtype=np.uint8)
    else:
        img = cv2.resize(img, (IMG_ROWS, IMG_COLS))
    return img.reshape(IMG_ROWS, IMG_COLS, 3)


def load_data_numpy(df):
    '''
    :param df: a pandas dataframe with the image paths and localization coordinates
    :return: the numpy representation of the images and the corresponding labels
    '''

    num_images = len(df)
    image_path_array = df['image_path'].to_numpy()
    label_array = df['category'].to_numpy()
    x1 = df['x1'].to_numpy().reshape(-1, 1)
    y1 = df['y1'].to_numpy().reshape(-1, 1)
    x2 = df['x2'].to_numpy().reshape(-1, 1)
    y2 = df['y2'].to_numpy().reshape(-1, 1)
    bbox_array = np.concatenate((x1, y1, x2, y2), axis=1)

    image_array = np.array([]).reshape(-1, IMG_ROWS, IMG_COLS, 3)
    adjusted_std = 1.0 / np.sqrt(IMG_COLS * IMG_ROWS * 3)

    for i in range(num_images):
        img = get_image(image_path_array[i], x1=x1[i, 0], y1=y1[i, 0], x2=x2[i, 0], y2=y2[i, 0])
        flip_indicator = np.random.randint(low=0, high=2)
        if flip_indicator == 0:
            img = cv2.flip(img, 1)

        image_array = np.concatenate((image_array, [img]))

    image_array = (image_array - imageNet_mean_pixel) / global_std

    return image_array, label_array, bbox_array



def prepare_df(path, usecols, shuffle=shuffle):
    '''
    :param path: the path of a csv file
    :param usecols: which columns to read
    :return: a pandas dataframe
    '''
    df = pd.read_csv(path, usecols=usecols)
    if shuffle:
        df = df.sample(frac=1).reset_index(drop=True)
    return df
