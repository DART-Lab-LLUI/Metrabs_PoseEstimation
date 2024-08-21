import arparse
import sys
import os
import glob
import re
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
parser.add_argument('--dir_out_json', metavar='djo', type=str, help='Path to folder to save output json files')
parser.add_argument('--skeleton', metavar='skel', type=str, default='coco_19', help='Skeleton to use for pose estimation, Default: coco_19')
parser.add_argument('--model_path', metavar='m', type=str, help='Path to the model to use for pose estimation')
parser.add_argument('--filter_2d', metavar='f2d', type=bool, default=False, help='Filter 2D Pose Data, Default: False')
parser.add_argument('--DEBUG', metavar='d', type=bool, default=False, help='Debug Mode, Default: False')

def pose_data_to_json(pose_data_samples):
    """
    Write 2D Keypoints to Json File

    Args:
        pose_data_samples: List of PoseData Objects
        data_source: 'mmpose' or 'metrabs'

    Thanks to Loïc Kreienbühl
    """
    json_data = {}
    json_data["people"] = []


    json_data = {}
    json_data["people"] = []
    person_id = -1
    cat_id = 1
    score = 0.95  # Assume good certainty for all keypoints
    for pose_data in pose_data_samples:
        keypoints = pose_data.numpy()
        keypoints_with_score = []
        for i in range(keypoints.shape[0]):
            keypoints_with_score.extend([float(keypoints[i, 0]), float(keypoints[i, 1]), score])
            json_data["people"].append({
                'person_id': person_id,
                'pose_keypoints_2d': keypoints_with_score,
            })
            person_id += 1

    return json_data

def json_out(pred, id, json_dir, video):

    json_name = os.path.join(json_dir, f"{os.path.basename(video).split('.mp4')[0]}_{id:06d}.json")
    json_file = open(json_name, "w")
    json.dump(pose_data_to_json(pred), json_file, indent=6)
    id += 1

def metrabs_pose_estimation_2d(dir_video, calib_file, dir_out_video, dir_out_json, skeleton='coco_19', model_path, filter_2d=False, DEBUG=False):
    """
    This sscript uses metrabs for 2D Pose estimation and writes json files.

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


    os.getcwd()


    try:
        print("loading HPE model")
        model = hub.load(model_path)
    except:
        tmp = os.path.join(os.cwd(), 'metrabs_models')
        #tmp = input("Loading model failed. The model will be donwloaded. Please give a path to save the model.") ##If we want to give the choice of the path to the user
        if not os.path.exists(tmp):
            os.makedirs(tmp)

        os.environ['TFHUB_CACHE_DIR'] = tmp
        model = hub.load('https://bit.ly/metrabs_l')  # To load the model from the internet and save it in a given tmp folder



    # Check if the directory exists, if not create it
    if not os.path.exists(out_video):
        os.makedirs(out_video)

    calib = toml.load(glob.glob(os.path.join(curr_trial.dir_calib_files, "*.toml"))[0])

    # Path to the first image/video file
    path = in_video
    video_files = [filename for filename in os.listdir(path) if
                   filename.endswith('.mp4') or filename.endswith('.mov') or filename.endswith('.avi')]

    print(f"DEBUG: {video_files}")
    for video_name in video_files:
        filepath = os.path.realpath(os.path.join(path, video_name))

        ##################################################
        #############  OPENING THE VIDEO  ################
        # For a video file
        cap = cv2.VideoCapture(filepath)

        # Check if file is opened correctly
        if not cap.isOpened():
            print("Could not open file")
            exit()

        # Prepare Jsonwriterprocess
        json_dir = os.path.join(out_json, f"{os.path.basename(video_name).split('.mp4')[0]}_json")

        if not os.path.exists(json_dir):
            os.makedirs(json_dir)

        # get intrinsics from calib file
        cam = re.search(r"cam\d*", video_name).group()
        intrinsic_matrix = None
        distortions = None

        for key in calib.keys():
            if calib.get(key).get("name") == cam:
                print(calib.get(key).get("name"))
                intrinsic_matrix = tf.constant(calib.get(key).get("matrix"), dtype=tf.float32)
                distortions = tf.constant(calib.get(key).get("distortions"), dtype=tf.float32)


        print(f"Current Video: {video_name}")
        joint_names = model.per_skeleton_joint_names[skeleton].numpy().astype(str)
        joint_edges = model.per_skeleton_joint_edges[skeleton].numpy()

        # Initializing variables for the loop
        frame_idx = 0

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
            # Show progress
            print(f" ------- {video_name}  Frame Nr. {frame_idx} ------- ")

            #convert Image t0 jpeg
            _, frame = cv2.imencode('.jpg', frame)
            frame = frame.tobytes()

            #covnert jpeg to tensor and run prediction
            frame = tf.image.decode_jpeg(frame, channels=3)

            ##############################################
            ################## DETECTION #################
            # Perform inference on the frame
            pred = model.detect_poses(frame, intrinsic_matrix=intrinsic_matrix, skeleton='coco_19')

            # Save detection's parameters
            bboxes = pred['boxes']
            pose_result_2d = pred['poses2d']

            ################## JSON Output #################
            # Add track id (useful for multiperson tracking)
            json_out(pred, frame_idx, json_dir, video_name)

            frame_idx += 1
            # Break the loop when 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release the VideoCapture object and close display window
        cap.release()
        print("DEBUG: Finished Video Processing.")

        if filter_2d:
            filter_2d_pose_data(curr_trial, json_dir)
        print("DEBUG: Go to next video")

    pass

if __name__ == '__main__':
    args = parser.parse_args()

    if sys.gettrace() is not None or args.DEBUG:
        print("Debug Mode is activated\n"
              "Starting debugging script.")

        if os.name == 'posix': # if running on WSL
            args.model_path = hub.load("/mnt/c/Users/johan/OneDrive/Dokumente/Studium/LLUI_Praktikum/05_iDrink/iDrink_GUI/src/metrabs_models/tensorflow/metrabs_eff2l_y4_384px_800k_28ds/d8503163f1198d9d4ee97bfd9c7f316ad23f3d90")
        else:
            args.model_path= hub.load(r"C:\Users\johan\OneDrive\Dokumente\Studium\LLUI_Praktikum\05_iDrink\iDrink_GUI\src\metrabs_models\tensorflow\metrabs_eff2l_y4_384px_800k_28ds\d8503163f1198d9d4ee97bfd9c7f316ad23f3d90")

        in_video = curr_trial.dir_recordings
        out_video = os.path.realpath(os.path.join(curr_trial.dir_trial, "videos", "pose"))
        out_json = os.path.realpath(os.path.join(curr_trial.dir_trial, "pose"))

    metrabs_pose_estimation_2d(args.dir_video, args.calib_file, args.dir_out_video, args.dir_out_json, args.skeleton,
                               args.model_path, args.filter_2d, args.DEBUG)
    pass