import argparse
import sys
import os
import glob
import re
import toml
import time
import json

import numpy as np

import cv2
from tqdm import tqdm

import toml
import glob
import re
import cameralib
import numpy as np
import posepile.joint_info
import poseviz
import simplepyutils as spu
import torch
import torchvision.io
import poseviz
import cameralib

import metrabs_pytorch.backbones.efficientnet as effnet_pt
import metrabs_pytorch.models.metrabs as metrabs_pt
from metrabs_pytorch.multiperson import multiperson_model
from metrabs_pytorch.util import get_config





parser = argparse.ArgumentParser(description='Metrabs 2D Pose Estimation for iDrink using Pytorch')
parser.add_argument('--dir_video', metavar='dvi', type=str,
                    help='Path to folder containing videos for pose estimation')
parser.add_argument('--calib_file', metavar='c', type=str,
                    help='Path to calibration file')
parser.add_argument('--dir_out_video', metavar='dvo', type=str,
                    help='Path to folder to save output videos')
parser.add_argument('--dir_out_json', metavar='djo', type=str,
                    help='Path to folder to save output json files')
parser.add_argument('--skeleton', metavar='skel', type=str, default='coco_19',
                    help='Skeleton to use for pose estimation, Default: coco_19')
parser.add_argument('--model_path', metavar='m', type=str,
                    default=os.path.join(os.getcwd(), 'metrabs_models'),
                    help=f'Path to the model to use for pose estimation. \n'
                         f'Default: {os.path.join(os.getcwd(), "metrabs_models")}')
parser.add_argument('--DEBUG', metavar='d', type=bool, default=False, help='Debug Mode, Default: False')


def load_multiperson_model(model_path):
    model_pytorch = load_crop_model(model_path)
    skeleton_infos = spu.load_pickle(f'{model_path}/skeleton_infos.pkl')
    joint_transform_matrix = np.load(f'{model_path}/joint_transform_matrix.npy')

    with torch.device('cuda'):
        return multiperson_model.Pose3dEstimator(
            model_pytorch.cuda(), skeleton_infos, joint_transform_matrix)


def load_crop_model(model_path):
    cfg = get_config()
    ji_np = np.load(f'{model_path}/joint_info.npz')
    ji = posepile.joint_info.JointInfo(ji_np['joint_names'], ji_np['joint_edges'])
    backbone_raw = getattr(effnet_pt, f'efficientnet_v2_{cfg.efficientnet_size}')()
    preproc_layer = effnet_pt.PreprocLayer()
    backbone = torch.nn.Sequential(preproc_layer, backbone_raw.features)
    model = metrabs_pt.Metrabs(backbone, ji)
    model.eval()

    inp = torch.zeros((1, 3, cfg.proc_side, cfg.proc_side), dtype=torch.float32)
    intr = torch.eye(3, dtype=torch.float32)[np.newaxis]

    model((inp, intr))
    model.load_state_dict(torch.load(f'{model_path}/ckpt.pt'))
    return model

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
    score = 0.8  # Assume good certainty for all keypoints
    for pose_data in pose_data_samples:
        keypoints = pose_data
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

def metrabs_pose_estimation_2d(dir_video, calib_file, dir_out_video, dir_out_json, model_path, skeleton='coco_19', DEBUG=False):

    get_config(f'{model_path}/config.yaml')

    multiperson_model_pt = load_multiperson_model(model_path).cuda()
    joint_names = multiperson_model_pt.per_skeleton_joint_names[skeleton]
    joint_edges = multiperson_model_pt.per_skeleton_joint_edges[skeleton].cpu().numpy()

    # Check if the directory exists, if not create it
    if not os.path.exists(dir_out_video):
        os.makedirs(dir_out_video)

    calib = toml.load(calib_file)

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

        # Prepare Jsonwriterprocess
        json_dir = os.path.join(dir_out_json, f"{os.path.basename(video_name).split('.mp4')[0]}_json")

        if not os.path.exists(json_dir):
            os.makedirs(json_dir)

        # get intrinsics from calib file
        cam = re.search(r"cam\d*", video_name).group()
        intrinsic_matrix = None
        distortions = None

        for key in calib.keys():
            if calib.get(key).get("name") == cam:
                intrinsic_matrix = calib.get(key).get("matrix")
                distortions = calib.get(key).get("distortions")

        print(f"Current Video: {video_name}")

        # Initializing variables for the loop
        frame_idx = 0
        buffer = []
        BUFFER_SIZE = 27

        tot_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        progress = tqdm(total=tot_frames, desc=f"Processing {video_name}", position=0, leave=True)

        with torch.inference_mode(), torch.device('cuda'):
            frames_in, _, _ = torchvision.io.read_video(filepath, output_format='TCHW')

            for frame_idx, frame in enumerate(frames_in):

                pred = multiperson_model_pt.detect_poses(frame, skeleton=skeleton,
                                                         intrinsic_matrix=torch.FloatTensor(intrinsic_matrix),
                                                         distortion_coeffs=torch.FloatTensor(distortions))

                # Save detection's parameters
                bboxes = pred['boxes'].cpu().numpy()
                pose_result_2d = pred['poses3d'].cpu().numpy()

                ################## JSON Output #################
                # Add track id (useful for multiperson tracking)
                json_out(pose_result_2d, frame_idx, json_dir, video_name)

                frame_idx += 1
                progress.update(1)

            # Release the VideoCapture object and close progressbar
            cap.release()
            progress.close()

if __name__ == '__main__':
    args = parser.parse_args()

    if sys.gettrace() is not None or args.DEBUG:
        print("Debug Mode is activated\n"
              "Starting debugging script.")

        if os.name == 'posix': # if running on WSL
            args.model_path = "/mnt/c/iDrink/metrabs_models/pytorch/metrabs_eff2l_384px_800k_28ds_pytorch"
            args.dir_video = "/mnt/c/iDrink/Session Data/S20240501-115510/S20240501-115510_P07/S20240501-115510_P07_T44/videos/recordings"
            args.dir_out_video = "/mnt/c/iDrink/Session Data/S20240501-115510/S20240501-115510_P07/S20240501-115510_P07_T44/videos/pose"
            args.dir_out_json = "/mnt/c/iDrink/Session Data/S20240501-115510/S20240501-115510_P07/S20240501-115510_P07_T44/pose"
            args.calib_file = "/mnt/c/iDrink/Session Data/S20240501-115510/S20240501-115510_Calibration/Calib_S20240501-115510.toml"
            args.skeleton = 'coco_19'
            args.filter_2d = False

        else:
            args.model_path= r"C:\iDrink\metrabs_models\pytorch\metrabs_eff2l_384px_800k_28ds_pytorch"
            args.dir_video = r"C:\iDrink\Session Data\S20240501-115510\S20240501-115510_P07\S20240501-115510_P07_T44\videos\recordings"
            args.dir_out_video = r"C:\iDrink\Session Data\S20240501-115510\S20240501-115510_P07\S20240501-115510_P07_T44\videos\pose"
            args.dir_out_json = r"C:\iDrink\Session Data\S20240501-115510\S20240501-115510_P07\S20240501-115510_P07_T44\pose"
            args.calib_file = r"C:\iDrink\Session Data\S20240501-115510\S20240501-115510_Calibration\Calib_S20240501-115510.toml"
            args.skeleton = 'coco_19'
            args.filter_2d = False

    metrabs_pose_estimation_2d(args.dir_video, args.calib_file, args.dir_out_video, args.dir_out_json, args.model_path,
                               args.skeleton, args.DEBUG)
    pass