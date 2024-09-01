import argparse
import sys
import os
import glob
import re

import pandas as pd
import toml
from trc import TRCData
from tqdm import tqdm

import numpy as np

import cv2

import tensorflow_hub as hub
import tensorflow as tf
import tensorflow_io as tfio


parser = argparse.ArgumentParser(description='Metrabs 2D Pose Estimation for iDrink using Tensorflow')
parser.add_argument('--identifier', metavar='id', type=str, help='Trial Identifier')
parser.add_argument('--video_file', metavar='dvi', type=str, help='Path to folder containing videos for pose estimation')
parser.add_argument('--calib_file', metavar='c', type=str, help='Path to calibration file')
parser.add_argument('--dir_out_video', metavar='dvo', type=str, help='Path to folder to save output videos')
parser.add_argument('--dir_out_trc', metavar='dtrc', type=str, help='Path to folder to save output trc files')
parser.add_argument('--skeleton', metavar='skel', type=str, default='coco_19', help='Skeleton to use for pose estimation, Default: coco_19')
parser.add_argument('--model_path', metavar='m', type=str, help='Path to the model to use for pose estimation')
parser.add_argument('--DEBUG', metavar='d', type=bool, default=False, help='Debug Mode, Default: False')


def filter_df(df_unfiltered, fs):
    """
    Use a butterworth-filter on the 3D-keypoints in the DF and return the filtered DF.

    :param df: DataFrame with 3D coordinates
    :param fs: Sampling frequency

    :return: DataFrame with filtered 3D coordinates

    """
    from scipy.signal import butter, sosfiltfilt

    df filtered = pandas.DataFrane(columns=df_unfiltered.columns)

    cutoff = 5
    order = 2  # Desired order 5. Because of filtfilt, half of that needs to be given. --> filtfilt doubles the order

    nyquist = 0.5 * fs

    if cutoff >= nyquist:
        print(f"Warning: Cutoff frequency {cutoff} is higher than Nyquist frequency {nyquist}.")
        print("Filtering with Nyquist frequency.")
        cutoff = int(nyquist - 1)

    normal_cutoff = cutoff / nyquist


    sos = butter(order, normal_cutoff, btype="low", analog=False, output="sos")
    filtered_data = sosfiltfilt(sos, data)


    return df_filtered

def df_to_trc(df, trc_file, identifier, fps, n_frames, n_markers):
    """
    Converts the DataFrame to a .trc file according to nomenclature used by Pose2Sim.

    Writes to trc file to given path.

    :param df: DataFrame with 3D coordinates
    """
    trc = TRCData()

    trc['PathFileType'] = '4'
    trc['DataFormat'] = '(X/Y/Z)'
    trc['FileName'] = f'{identifier}.trc'
    trc['DataRate'] = fps
    trc['CameraRate'] = fps
    trc['NumFrames'] = n_frames
    trc['NumMarkers'] = n_markers
    trc['Units'] = 'm'
    trc['OrigDataRate'] = fps
    trc['OrigDataStartFrame'] = 0
    trc['OrigNumFrames'] = n_frames
    trc['Frame#'] = [i for i in range(n_frames)]
    trc['Time'] = [round(i / fps, 3) for i in range(n_frames)]
    trc['Markers'] = df.columns.tolist()



    for column in df.columns:
        trc[column] = df[column].tolist()

    for i in range(n_frames):
        trc[i] = [trc['Time'][i], df.iloc[i, :].tolist()]

    trc.save(trc_file)

    pass


def get_column_names(joint_names):
    """
    Uses the list of joint names to create a list of column names for the DataFrame

    e.g. neck --> neck_x, neck_y, neck_z

    Input: joint_names: List of joint names
    Output: List of column names for x, y, and z axis
    """
    columns = []
    """
    # Old version might be used if x, y, z coordinates need to be split up in DataFrame
    for joint in joint_names:
        for axis in ['x', 'y', 'z']:
            columns.append(f"{joint}_{axis}")"""

    for joint in joint_names:
        columns.append(joint)

    return columns

def add_to_dataframe(df, pose_result_3d):
    """
    Add 3D  Keypoints to DataFrame

    Returns DataFrame with added 3D keypoints
    """

    temp = []
    for i in range(pose_result_3d.shape[1]):
        temp.append([x/1000 for x in pose_result_3d[0][i].tolist()])

    df.loc[len(df)] = temp

    return df

def metrabs_pose_estimation_3d(video_file, calib_file, dir_out_video, dir_out_trc, model_path, identifier, skeleton='coco_19', DEBUG=False):
    """
    3D Pose estimaiton using Metrabs

    The coordinates are saved into .trc files according to the Pose2Sim nomenclature.

    This script uses the tensorflow version of Metrabs.


    :param in_video:
    :param out_video:
    :param out_json:
    :param skeleton:
    :param writevideofiles:
    :param filter_2d:
    :param DEBUG:
    :return:
    """

    try:
        print("loading HPE model")
        model = hub.load(model_path)
    except:
        tmp = os.path.join(os.getcwd(), 'metrabs_models')
        #tmp = input("Loading model failed. The model will be donwloaded. Please give a path to save the model.") ##If we want to give the choice of the path to the user
        if not os.path.exists(tmp):
            os.makedirs(tmp)

        # Add path to the environment variable --> This is necessary to save the model in the given path
        os.environ['TFHUB_CACHE_DIR'] = tmp
        model = hub.load('https://bit.ly/metrabs_l')  # To load the model from the internet and save it in a given tmp folder

    # Check if the directory exists, if not create it
    if not os.path.exists(dir_out_video):
        os.makedirs(dir_out_video)

    calib = toml.load(calib_file)

    ##################################################
    #############  OPENING THE VIDEO  ################
    # For a video file
    cap = cv2.VideoCapture(video_file)
    # Check if file is opened correctly
    if not cap.isOpened():
        print("Could not open file")
        exit()
    # get intrinsics from calib file
    cam = re.search(r"cam\d*", os.path.basename(video_file)).group()
    intrinsic_matrix = None
    distortions = None
    for key in calib.keys():
        if calib.get(key).get("name") == cam:
            intrinsic_matrix = tf.constant(calib.get(key).get("matrix"), dtype=tf.float32)
            distortions = tf.constant(calib.get(key).get("distortions"), dtype=tf.float32)
    joint_names = model.per_skeleton_joint_names[skeleton].numpy().astype(str)
    joint_edges = model.per_skeleton_joint_edges[skeleton].numpy()
    # Initializing variables for the loop
    frame_idx = 0
    #Prepare DataFrame
    df = pd.DataFrame(columns=get_column_names(joint_names))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_markers = len(joint_names)
    progress = tqdm(total=n_frames, desc=f"Processing {os.path.basename(video_file)}", position=0, leave=True)
    while True:
        # Read frame from the webcam
        ret, frame = cap.read()
        # If frame is read correctly ret is True
        if not ret:
            break
        # Stop setting for development
        if DEBUG:
            if frame_idx == 30:
                break
        #convert Image t0 jpeg
        _, frame = cv2.imencode('.jpg', frame)
        frame = frame.tobytes()
        #covnert jpeg to tensor and run prediction
        frame = tf.image.decode_jpeg(frame, channels=3)
        ##############################################
        ################## DETECTION #################
        # Perform inference on the frame
        pred = model.detect_poses(frame, intrinsic_matrix=intrinsic_matrix, skeleton=skeleton)
        # Save detection's parameters
        bboxes = pred['boxes']
        pose_result_3d = pred['poses3d'].numpy()
        ################## Add to DataFrame #################
        # Add coordinates to Dataframe
        df = add_to_dataframe(df, pose_result_3d)
        frame_idx += 1
        progress.update(1)
    # Release the VideoCapture object and close progressbar
    cap.release()
    progress.close()

    filter_df(df, fps)

    if not os.path.exists(dir_out_trc):
        os.makedirs(dir_out_trc)
    trc_file = os.path.join(dir_out_trc, f"{os.path.basename(os.path.basename(video_file)).split('.mp4')[0]}_0-{frame_idx}.trc")
    df_to_trc(df, trc_file, identifier, fps, n_frames, n_markers)

if __name__ == '__main__':
    args = parser.parse_args()

    if sys.gettrace() is not None or args.DEBUG:
        print("Debug Mode is activated\n"
              "Starting debugging script.")

        if os.name == 'posix': # if running on WSL
            args.identifier = "S20240501-115510_P01_T01"
            args.model_path = hub.load("/mnt/c/iDrink/metrabs_models/tensorflow/metrabs_eff2l_y4_384px_800k_28ds/d8503163f1198d9d4ee97bfd9c7f316ad23f3d90")
            args.video_file = "/mnt/c/iDrink/Session Data/S20240501-115510/S20240501-115510_P07/S20240501-115510_P07_T44/videos/recordings/cam1_trial_44_R_affected.mp4"
            args.dir_out_video = "/mnt/c/iDrink/Session Data/S20240501-115510/S20240501-115510_P07/S20240501-115510_P07_T44/videos/pose-3d"
            args.dir_out_trc = "/mnt/c/iDrink/Session Data/S20240501-115510/S20240501-115510_P07/S20240501-115510_P07_T44/pose"
            args.calib_file = "/mnt/c/iDrink/Session Data/S20240501-115510/S20240501-115510_Calibration/Calib_S20240501-115510.toml"
            args.skeleton = 'coco_19'
        else:
            args.identifier = "S20240501-115510_P01_T01"
            args.model_path= hub.load(r"C:\iDrink\metrabs_models\tensorflow\metrabs_eff2l_y4_384px_800k_28ds\d8503163f1198d9d4ee97bfd9c7f316ad23f3d90")
            args.video_file = r"C:\iDrink\Session Data\S20240501-115510\S20240501-115510_P07\S20240501-115510_P07_T44\videos\recordings\cam1_trial_44_R_affected.mp4"
            args.dir_out_video = r"C:\iDrink\Session Data\S20240501-115510\S20240501-115510_P07\S20240501-115510_P07_T44\videos\pose-3d"
            args.dir_out_trc = r"C:\iDrink\Session Data\S20240501-115510\S20240501-115510_P07\S20240501-115510_P07_T44\pose"
            args.calib_file = r"C:\iDrink\Session Data\S20240501-115510\S20240501-115510_Calibration\Calib_S20240501-115510.toml"
            args.skeleton = 'coco_19'

    metrabs_pose_estimation_3d(args.video_file, args.calib_file, args.dir_out_video, args.dir_out_trc, args.model_path,
                               args.identifier, args.skeleton, args.DEBUG)