
import os
from tqdm import tqdm
import cv2
import pandas as pd
import sys

def create_folder(folder_name):
    """
    :param folder_name: the folder name
    Creates that folder if not exist
    """
    if not os.path.isdir(folder_name):
        os.mkdir(folder_name)  # make sure the directory exists


def get_video_data(videos_path):
    """
    :param videos_path: gets a path of videos, a folder that contains all the videos
    :return: A dataframe that contains all the videos' names and their class (class = in what folder they exists)

    For example: name: video_1.mkv tag: criss_cross video_1.mkv (name of the video) is under the sub-folder named
    criss_cross, this folder is under the folder stated in videos_path.
    """

    videos_tags_list = []
    for exercise_folder in os.listdir(path=videos_path):
        if not exercise_folder.startswith('.'):  # skip hidden files
            for video in os.listdir(path=f'{videos_path}/{exercise_folder}'):
                videos_tags_list.append({'name': video, 'tag': exercise_folder})

    video_df = pd.DataFrame(data=videos_tags_list)

    return video_df


def create_folders_and_sub_folders(folder_name, sub_folder_list):
    """
    :param sub_folder_list: sub folders
    :param folder_name: folder name
    :param subfolder_list: list of sub folders to be created
    :return: creates a folder, and after that sub-folders in it
    """

    create_folder(f'{folder_name}')

    for sub_folder in sub_folder_list:
        create_folder(f'{folder_name}/{sub_folder}')


def extract_frames(videos_path, frames_path, df):
    """
    :param videos_path: the video path
    :param frames_path: the frames path
    :param df: dataframe that contains the video data
    :return:
    """

    # indexes 0 and 1, for the list
    NAME = 0
    TAG = 1

    # Frame rate
    FRAMES = 20
    # Max attempts before it will give up?
    MAX_ATTEMPTS = 10

    # storing the frames from training videos
    counter_videos = 0
    frame_videos_list = []
    for video in tqdm(df.itertuples(index=False)):
        counter_frames = 0
        counter_attempts = 0
        video_path = f'{videos_path}/{video[TAG]}/{video[NAME]}'
        cap = cv2.VideoCapture(video_path)  # capturing the video from the given path
        while cap.isOpened():
            ret, frame = cap.read()
            if ret is False:
                if counter_attempts < MAX_ATTEMPTS:
                    counter_attempts += 1
                    continue
                else:
                    break
            if counter_frames < FRAMES:
                # storing the frames in a new folder named train
                filename = f'{frames_path}/{video[TAG]}/video{counter_videos}_frame{counter_frames}.jpg'
                counter_frames += 1
                cv2.imwrite(filename, frame)
                frame_videos_list.append({'video': video[NAME], 'image': filename, 'class': video[TAG]})
        counter_videos += 1
    cap.release()
    print('Done reading the videos and writing the frames')
    return frame_videos_list

def get_frames_from_folders():
    """
    Run this if you are already finished the extraction and want to load exsisting frames
    :return: dataframe with data on the frames
    """

    videos_path = 'videos'
    frame_path = 'frames'
    video_df = get_video_data(videos_path=videos_path)

    videos_list = extract_frames(videos_path=videos_path, frames_path=frame_path, df=video_df)
    frames_data = pd.DataFrame(videos_list)
    return frames_data

def get_frames_from_videos():
    """
    Creates the frames from videos
    :return: a frame dataframe contains data on the extracted frames: video name, frame name, class.
    """
    # Get the videos' names and their classes
    videos_path = './videos'
    frame_path = 'frames'

    video_df = get_video_data(videos_path=videos_path)

    # Set tags as the list of unique classes
    tags = list(video_df['tag'].unique())
    sub_folder_list = ['criss_cross', 'double_leg', 'roll_up']

    create_folders_and_sub_folders(folder_name=frame_path, sub_folder_list=sub_folder_list)

    videos_list = extract_frames(videos_path=videos_path, frames_path=frame_path, df=video_df)
    frames_data = pd.DataFrame(videos_list)
    print('Finished frame extraction')
    return frames_data
