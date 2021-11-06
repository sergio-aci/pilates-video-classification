from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import pandas as pd


from frame_extraction import get_frames_from_folders
from split_preprocces import split_data, images_to_array_with_classes
from testing import test
from training import train, hypertuning


def test_func():
    test_data = pd.read_csv('test.csv')
    y_test = test_data['class']
    predict, actual = test(test_data, y_test)
    print(accuracy_score(predict, actual) * 100)
    return


def train_func():
    # If this is your first time creating frames use this:
    # frame_data = frame_extraction.get_frames_from_videos()
    # frame_data = get_frames_from_videos()

    # if you already have frames and want to load them
    frame_data = get_frames_from_folders()
    train_data, test_data = split_data(frame_data)

    X_train, y_train = images_to_array_with_classes(train_data)
    X_test, y_test = images_to_array_with_classes(test_data)

    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)

    test_data.to_csv('test.csv')
    return X_train, y_train, train_data['video']

if __name__ == '__main__':
    X_train, y_train, groups = train_func()
    best_params = hypertuning(X_train, y_train, groups)
    train(X_train, y_train, best_params)
    test_func()