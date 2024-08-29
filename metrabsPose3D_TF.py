import argparse
import sys
import os
import glob
import re

import pandas as pd
import toml
import time
import json

import numpy as np

import cv2

import tensorflow_hub as hub
import tensorflow as tf
import tensorflow_io as tfio


parser = argparse.ArgumentParser(description='Metrabs 2D Pose Estimation for iDrink using Tensorflow')
parser.add_argument('--dir_video', metavar='dvi', type=str, help='Path to folder containing videos for pose estimation')
parser.add_argument('--calib_file', metavar='c', type=str, help='Path to calibration file')
parser.add_argument('--dir_out_video', metavar='dvo', type=str, help='Path to folder to save output videos')
parser.add_argument('--dir_out_trc', metavar='dtrc', type=str, help='Path to folder to save output trc files')
parser.add_argument('--skeleton', metavar='skel', type=str, default='coco_19', help='Skeleton to use for pose estimation, Default: coco_19')
parser.add_argument('--model_path', metavar='m', type=str, help='Path to the model to use for pose estimation')
parser.add_argument('--DEBUG', metavar='d', type=bool, default=False, help='Debug Mode, Default: False')

def csv_to_trc(df):
    """
    Converts the DataFrame to a .trc file according to nomenclature used by Pose2Sim
    :param df:
    :return:
    """


def get_column_names(joint_names):
    """
    Uses the list of joint names to create a list of column names for the DataFrame

    e.g. neck --> neck_x, neck_y, neck_z
    :return:
    """

    pass

def get_column(df, pose_result_3d):

def metrabs_pose_estimation_2d(dir_video, calib_file, dir_out_video, dir_out_trc, model_path, skeleton='coco_19', DEBUG=False):
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

    # Path to the first image/video file
    video_files = [filename for filename in os.listdir(dir_video) if
                   filename.endswith('.mp4') or filename.endswith('.mov') or filename.endswith('.avi')]

    for video_name in video_files:
        filepath = os.path.realpath(os.path.join(dir_video, video_name))

        ##################################################
        #############  OPENING THE VIDEO  ################
        # For a video file
        cap = cv2.VideoCapture(filepath)

        # Check if file is opened correctly
        if not cap.isOpened():
            print("Could not open file")
            exit()





        # get intrinsics from calib file
        cam = re.search(r"cam\d*", video_name).group()
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

        df = pd.DataFrame(columns=get_colums(joint_names))

        tot_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        progress = tqdm(total=tot_frames, desc=f"Processing {video_name}", position=0, leave=True)

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
            pose_result_3d = pred['poses3d']

            ################## Add to DataFrame #################
            # Add coordinates to Dataframe
            get_column(df, pose_result_3d)

            frame_idx += 1
            progress.update(1)

        # Release the VideoCapture object and close progressbar
        cap.release()
        progress.close()

        if not os.path.exists(dir_out_trc):
            os.makedirs(dir_out_trc)

        trc_file = os.path.join(dir_out_trc, f"{os.path.basename(video_name).split('.mp4')[0]}_0-{frame_idx}.trc")

if __name__ == '__main__':
    args = parser.parse_args()

    if sys.gettrace() is not None or args.DEBUG:
        print("Debug Mode is activated\n"
              "Starting debugging script.")

        if os.name == 'posix': # if running on WSL
            args.model_path = hub.load("/mnt/c/iDrink/metrabs_models/tensorflow/metrabs_eff2l_y4_384px_800k_28ds/d8503163f1198d9d4ee97bfd9c7f316ad23f3d90")
            args.dir_video = "/mnt/c/iDrink/Session Data/S20240501-115510/S20240501-115510_P07/S20240501-115510_P07_T44/videos/recordings"
            args.dir_out_video = "/mnt/c/iDrink/Session Data/S20240501-115510/S20240501-115510_P07/S20240501-115510_P07_T44/videos/pose"
            args.dir_out_trc = "/mnt/c/iDrink/Session Data/S20240501-115510/S20240501-115510_P07/S20240501-115510_P07_T44/pose"
            args.calib_file = "/mnt/c/iDrink/Session Data/S20240501-115510/S20240501-115510_Calibration/Calib_S20240501-115510.toml"
            args.skeleton = 'coco_19'

        else:
            args.model_path= hub.load(r"C:\iDrink\metrabs_models\tensorflow\metrabs_eff2l_y4_384px_800k_28ds\d8503163f1198d9d4ee97bfd9c7f316ad23f3d90")
            args.dir_video = r"C:\iDrink\Session Data\S20240501-115510\S20240501-115510_P07\S20240501-115510_P07_T44\videos\recordings"
            args.dir_out_video = r"C:\iDrink\Session Data\S20240501-115510\S20240501-115510_P07\S20240501-115510_P07_T44\videos\pose"
            args.dir_out_trc = r"C:\iDrink\Session Data\S20240501-115510\S20240501-115510_P07\S20240501-115510_P07_T44\pose"
            args.calib_file = r"C:\iDrink\Session Data\S20240501-115510\S20240501-115510_Calibration\Calib_S20240501-115510.toml"
            args.skeleton = 'coco_19'

    metrabs_pose_estimation_3D(args.dir_video, args.calib_file, args.dir_out_video, args.dir_out_trc, args.model_path,
                               args.skeleton, args.DEBUG)
    pass