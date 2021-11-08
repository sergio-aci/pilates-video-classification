import csv
import glob
import os
import pandas as pd
import youtube_dl, subprocess
from datetime import datetime
from tqdm import tqdm


def get_duration(start, end):
    start = str(start)[0:5]
    end = str(end)[0:5]
    FMT = '%M:%S'
    print('start:', start)
    print('end:', end)
    duration = datetime.strptime(end, FMT) - datetime.strptime(start, FMT)

    return duration


def read_sheets(file_name):
    xls = pd.ExcelFile(file_name)
    criss_cross = pd.read_excel(xls, 'criss_cross')
    double_leg = pd.read_excel(xls, 'double_leg')
    rollup = pd.read_excel(xls, 'roll_up')
    valid_criss_cross = pd.read_excel(xls, 'valid_criss_cross')
    valid_double_leg = pd.read_excel(xls, 'valid_double_leg')
    valid_rollup = pd.read_excel(xls, 'valid_roll_up')
    return criss_cross, double_leg, rollup, valid_criss_cross, valid_double_leg, valid_rollup


def create_folder(folder_name):
    if not os.path.isdir(folder_name):
        os.mkdir(folder_name)  # make sure the directory exists


def make_directories():
    # Create the videos directories
    create_folder('videos')
    create_folder('videos/train')
    create_folder('videos/train/roll_up')
    create_folder('videos/train/double_leg')
    create_folder('videos/train/criss_cross')
    create_folder('videos/validation')
    create_folder('videos/validation/valid_criss_cross')
    create_folder('videos/validation/valid_double_leg')
    create_folder('videos/validation/valid_roll_up')
    # Create the frames directories
    create_folder('frames/')
    create_folder('frames/validation')
    create_folder('frames/train')
    create_folder('frames/train/roll_up')
    create_folder('frames/train/double_leg')
    create_folder('frames/train/criss_cross')
    create_folder('frames/validation/valid_criss_cross')
    create_folder('frames/validation/valid_double_leg')
    create_folder('frames/validation/valid_roll_up')


count = 0


def download_video(url, start, end, folder_name, dataset_name):
    global count
    try:
        duration = get_duration(start, end)
        with youtube_dl.YoutubeDL({'format': 'best'}) as ydl:
            result = ydl.extract_info(url, download=False)
            video = result['entries'][0] if 'entries' in result else result
        url = video['url']
        # video_output = folder_name + '\\'+ str(video['id']) + '.mkv'
        video_output = 'videos' + '\\' + dataset_name + '\\' + folder_name + '\\' + str(count) + str(
            video['id']) + '.mkv'
        count += 1
        # start = '00:0'+start
        # end = '00:0' + end
        subprocess.call(f'ffmpeg -i "{url}" -ss {start} -t {duration} -c:v copy -c:a copy -an  "{video_output}"',
                        shell=True)
    except:
        pass


def download_videos_from_pandas(df, folder_name, dataset_name):
    for index, row in df.iterrows():
        link = row['link']
        print(link)
        start = row['start']
        end = row['end']
        print('duration:', get_duration(start, end))
        download_video(link, start, end, folder_name, dataset_name)


if __name__ == '__main__':
    criss_cross, double_leg, rollup, valid_criss_cross, valid_double_leg, valid_rollup = read_sheets('data.xlsx')
    make_directories()
    dataset_name = 'train'
    download_videos_from_pandas(criss_cross, 'criss_cross', dataset_name)
    download_videos_from_pandas(double_leg, 'double_leg', dataset_name)
    download_videos_from_pandas(rollup, 'roll_up', dataset_name)
    dataset_name = 'validation'
    download_videos_from_pandas(valid_criss_cross, 'valid_criss_cross', dataset_name)
    download_videos_from_pandas(valid_double_leg, 'valid_double_leg', dataset_name)
    download_videos_from_pandas(valid_rollup, 'valid_roll_up', dataset_name)
