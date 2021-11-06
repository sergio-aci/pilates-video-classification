from tqdm import tqdm
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import load_model
import numpy as np
from scipy import stats
import pandas as pd
from frame_extraction import get_video_data

ONLY = 0


def test(test_data, y_test):
    y_test = pd.get_dummies(y_test)
    model = load_model('weight.hdf55')
    videos_path = './videos'

    video_data = get_video_data(videos_path)

    # creating two lists to store predicted and actual tags
    predict = []
    actual = []
    for video in tqdm(test_data['video'].unique()):

        prediction_images = []
        for frame in test_data.loc[test_data['video'] == video, 'image']:
            img = image.load_img(frame, target_size=(224, 224, 3))
            img = image.img_to_array(img)
            prediction_images.append(img)

        prediction_images = preprocess_input(np.array(prediction_images))
        prediction = np.argmax(model.predict(prediction_images), axis=-1)
        # appending the mode of predictions in predict list to assign the tag to the video
        predict.append(y_test.columns.values[stats.mode(prediction)[ONLY][ONLY]])
        actual.append(video_data.loc[video_data['name'] == video, 'tag'].iloc[ONLY])
    return predict, actual
