
from sklearn.model_selection import StratifiedGroupKFold
import numpy as np
from tqdm import tqdm
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import pandas as pd


def split_data(data):
    train_inds, test_inds = next(
        StratifiedGroupKFold(shuffle=True, random_state=28)\
        .split(data, y=data['class'], groups=data['video']))

    train_data = data.iloc[train_inds].reset_index()
    test_data = data.iloc[test_inds].reset_index()
    return train_data, test_data


def images_to_array_with_classes(data):
    """
    :param data: a data set of images
    :return:
    """

    SAMPLES = 0
    X = np.empty((data.shape[SAMPLES], 224, 224, 3))
    # for loop to read and store frames
    for i in tqdm(range(data.shape[SAMPLES])):
        # loading the image and keeping the target size as (224,224,3)
        img = image.load_img(data['image'][i], target_size=(224, 224, 3))
        # converting it to array
        img = image.img_to_array(img)
        # normalizing the pixel value
        # img = img/255
        # appending the image to the train_image list
        X[i] = img
    X = preprocess_input(np.array(X))
    y = pd.get_dummies(data['class'])
    y = data['class']

    return X, y
