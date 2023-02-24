'''
create Marine work labels data for each frame.
At first, read work record file(.csv) and open video file(.mp4) to analyze below.
get start time and end time from video file name.
get frame rate from video file.
work recode file format is below.
top row is [日付, 開始時間=start time, 終了時間=end time, 作業項目=work name, 作業No.] use as column name.
only use [開始時間, 終了時間, 作業項目] columns.
開始時間 and 終了時間 format is "HH:MM:SS".
file name format is "YYYYMMDDHHMMSS.mp4".
file name format is "*************YYYYMMDDHHMMSS*YYYYMMDDHHMMSS.mp4.
the first 13 characters are not used.
14~27 characters are start time.
29~42 characters are end time.
use only HHMMSS.
convert start time and end time to pandas datetime64 format.
Second, convert work name in work record file corresponding to work list.
Third, calculate total frame number from start time, end time of video file and frame rate.
for example, if start time is 10:00:00, end time is 11:00:00 and frame rate is 30,
total frame number is 1800.
Fourth, below loop process from frame number=0 to frame number=total frame number.
[loop process]
initialize label data which size is [work list length].
convert current frame number to time by using frame rate.
check start time and end time in each work recode rows in work record file.
if start time and end time of work record file is in the loop time,
the value corresponding to work name is 1 in initialized label data.
out put label data to label file(.csv) which name is frame number.
[loop process end]
release video file.
'''

import cv2
import numpy as np
import pandas as pd
import csv
import os
import argparse
import datetime

def main():
    video_file = opt.video_file
    work_record_file = opt.work_record_file
    work_name_list = get_work_name_list(work_record_file)
    name = opt.name
    create_Marine_work_labels(video_file, work_record_file, work_name_list, name)


def replace_with_dict(df, column, replace_dict):
    for key, values in replace_dict.items():
        df.loc[df[column].isin(values), column] = key
    not_found = set(df[column]) - set(replace_dict.keys())
    if not_found:
        raise ValueError(f"The following values in column '{column}' were not found in the replace_dict: {not_found}")
    return df

def get_work_name_list(work_record_file):
    df = pd.read_csv(work_record_file)
    work_name_list = df['work_name'].unique()
    return work_name_list

def get_work_record(work_record_file):
    df = pd.read_csv(work_record_file)
    work_record = df.values.tolist()
    return work_record

def get_work_index(work_name_list, work_name):
    return work_name_list.index(work_name)

def get_work_name(work_name_list, work_name_index):
    return work_name_list[work_name_index]

def create_Marine_work_labels(video_file, work_record_file, work_name_list, name):
    # open video file
    cap = cv2.VideoCapture(video_file)
    # get frame rate
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    # get video file name
    video_file_name = os.path.basename(video_file)
    # release video file
    cap.release()

    # get start time and end time(HHMMSS) from video file name
    video_start_time = video_file_name[21:27]
    video_end_time = video_file_name[36:42]
    # convert start time and end time to pandas datetime64 format
    video_start_time = pd.to_datetime(video_start_time, format='%H%M%S')
    video_end_time = pd.to_datetime(video_end_time, format='%H%M%S')

    # get work record list as pandas dataframe
    df = pd.read_csv(work_record_file)
    # delete unnecessary columns
    df = df.drop(['日付', '作業No.'], axis=1)
    # cpnvert column name
    df = df.rename(columns={'開始時間': 'start_time', '終了時間': 'end_time', '作業項目': 'work_name'})
    # convert work name in work record file corresponding to work list
    df = replace_with_dict(df, 'work_name', {'work_name': work_name_list})

    # check if start time and end time of each work record is in start time and end time of video file
    for index, row in df.iterrows():
        if (row['start_time'] < video_start_time) or (row['end_time'] > video_end_time):
            assert False, 'start time and end time of each work record is not in start time and end time of video file'

    # check if start time of each work record is less than end time of each work record
    for index, row in df.iterrows():
        if (row['start_time'] > row['end_time']):
            assert False, 'start time of each work record is not less than end time of each work record'

    # loop process
    # calculate total frame number
    total_frame_number = int((video_end_time - video_start_time).total_seconds() * frame_rate)

    # loop process from frame number=0 to frame number=total frame number
    for frame_number in range(total_frame_number):
        # loop porecess every frame rate
        if frame_number % frame_rate == 0:
            # initialize label data which size is [work list length]
            label_data = np.zeros(len(work_name_list), dtype=np.int8)
            # convert current frame number to time by using frame rate
            current_time = video_start_time + datetime.timedelta(seconds=frame_number/frame_rate)
            # check start time and end time in each work recode rows in work record file
            for index, row in df.iterrows():
                if (row['start_time'] <= current_time) and (row['end_time'] >= current_time):
                    # the value corresponding to work name is 1 in initialized label data
                    label_data[get_work_index(work_name_list, row['work_name'])] = 1
            # out put label data to label file(.csv) which name is frame number
            # save label file in runs/label directory, if directory is not exist, create it.
            if not os.path.exists('runs/label/' + name):
                os.makedirs('runs/label/' + name)
            with open('runs/label/' + name + '/' + str(frame_number//frame_rate) + '.csv', 'w') as f:
                writer = csv.writer(f)
                writer.writerow(label_data)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_file', type=str, default='source/create_marine_work_labels/2020-01-01_10-00-00_2020-01-01_11-00-00.mp4', help='video file path')
    parser.add_argument('--work_record_file', type=str, default='source/create_marine_work_labels/2020-01-01_10-00-00_2020-01-01_11-00-00.csv', help='work record file path')
    parser.add_argument('--work_name_list', type=str, default='source/create_marine_work_labels/work_name_list.csv', help='work name list file path')
    parser.add_argument('--name', type=str, default='test', help='name of label set')
    opt = parser.parse_args()
    print(opt)

    main()