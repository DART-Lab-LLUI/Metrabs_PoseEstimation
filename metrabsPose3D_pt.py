import argparse
import os
import re
import sys

import cv2
import numpy as np
import pandas as pd
import simplepyutils as spu
import toml
import torch
import torchvision.io
from tqdm import tqdm
from trc import TRCData

import metrabs_pytorch.backbones.efficientnet as effnet_pt
import metrabs_pytorch.models.metrabs as metrabs_pt
import posepile.joint_info
from metrabs_pytorch.multiperson import multiperson_model
from metrabs_pytorch.util import get_config

parser = argparse.ArgumentParser(description='Metrabs 2D Pose Estimation for iDrink using Pytorch')
parser.add_argument('--video_file', metavar='dvi', type=str,
                    help='Path to video-file for pose estimation')
parser.add_argument('--calib_file', metavar='c', type=str,
                    help='Path to calibration file')
parser.add_argument('--dir_out_video', metavar='dvo', type=str,
                    help='Path to folder to save output videos')
parser.add_argument('--dir_out_trc', metavar='dtrc', type=str,
                    help='Path to folder to save output trc files')
parser.add_argument('--skeleton', metavar='skel', type=str, default='coco_19',
                    help='Skeleton to use for pose estimation, Default: coco_19')
parser.add_argument('--model_path', metavar='m', type=str,
                    default=os.path.join(os.getcwd(), 'metrabs_models'),
                    help=f'Path to the model to use for pose estimation. \n'
                         f'Default: {os.path.join(os.getcwd(), "metrabs_models")}')
parser.add_argument('--verbose', metavar='v', type=int, default=1, help='Verbose Mode (0, 1, 2), Default: 1')
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


def filter_df(df_unfiltered, fs, verbose, normcutoff=False):
    """
    Use a butterworth-filter on the 3D-keypoints in the DF and return the filtered DF.

    :param df: DataFrame with 3D coordinates
    :param fs: Sampling frequency

    :return: DataFrame with filtered 3D coordinates

    """
    from scipy.signal import butter, sosfiltfilt

    df_filtered = pd.DataFrame(columns=df_unfiltered.columns)

    cutoff = 5
    order = 2  # Desired order 5. Because of filtfilt, half of that needs to be given. --> filtfilt doubles the order

    nyquist = 0.5 * fs

    if cutoff >= nyquist:
        if verbose >= 1:
            print(f"Warning: Cutoff frequency {cutoff} is higher than Nyquist frequency {nyquist}.")
            print("Filtering with Nyquist frequency.")
        cutoff = int(nyquist - 1)

    if normcutoff:
        cutoff = cutoff / nyquist
    else:
        sos = butter(order, cutoff, btype="low", analog=False, output="sos", fs=fs)

    if verbose >= 2:
        print(f"Filtering 3D keypoints:\n"
              f"Filter: Butterworth\n"
              f"Order: {order}\n"
              f"Sampling frequency: {fs} Hz\n"
              f"cutoff frequency of {cutoff} Hz")

    if verbose >= 1:
        progress = tqdm(total=len(df_unfiltered.columns), desc="Filtering 3D keypoints", position=0, leave=True)

    for column in df_unfiltered.columns:
        data = np.array(df_unfiltered[column].tolist())
        df_filtered[column] = sosfiltfilt(sos, data, axis=0).tolist()

        if verbose >= 1:
            progress.update(1)

    if verbose >= 1:
        progress.close()

    return df_filtered


def df_to_trc(df, trc_file, identifier, fps, n_frames, n_markers, verbose=1):
    """
    Converts the DataFrame to a .trc file according to nomenclature used by Pose2Sim.

    Writes to trc file to given path.

    :param df: DataFrame with 3D coordinates
    """
    trc = TRCData()

    if verbose >= 1:
        print(f"Writing .trc file {os.path.basename(trc_file)}")

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

    if verbose >= 2:
        print(f'columns added to {os.path.basename(trc_file)}')

    for column in tqdm(df.columns, desc="Writing .trc file", position=0, leave=True):
        trc[column] = df[column].tolist()

    for i in range(n_frames):
        trc[i] = [trc['Time'][i], df.iloc[i, :].tolist()]

    if verbose >= 2:
        print(f'Timestamps added to {os.path.basename(trc_file)}')

    trc.save(trc_file)
    if verbose >= 1:
        print(f"Saved .trc file {os.path.basename(trc_file)}")


def get_column_names(joint_names, verbose=1):
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

    if verbose >= 2:
        print(f'Columns created: {columns}')

    return columns


def add_to_dataframe(df, pose_result_3d):
    """
    Add 3D  Keypoints to DataFrame

    Returns DataFrame with added 3D keypoints
    """

    temp = []
    for i in range(pose_result_3d.shape[1]):
        temp.append([x / 1000 for x in pose_result_3d[0][i].tolist()])

    df.loc[len(df)] = temp

    return df


def metrabs_pose_estimation_3d(video_file, calib_file, dir_out_video, dir_out_trc, model_path, identifier,
                               skeleton='coco_19', verbose=1, DEBUG=False):

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
            intrinsic_matrix = calib.get(key).get("matrix")
            distortions = calib.get(key).get("distortions")
    print(f"Current Video: {os.path.basename(video_file)}")
    # Initializing variables for the loop
    frame_idx = 0


    # Prepare DataFrame
    df = pd.DataFrame(columns=get_column_names(joint_names))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_markers = len(joint_names)

    if verbose >= 2:
        print(f"Number of frames: {n_frames}")
        print(f"FPS: {fps}")
        print(f"Number of markers: {n_markers}")
    tot_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if verbose >=1:
        progress = tqdm(total=tot_frames, desc=f"Processing {os.path.basename(video_file)}", position=0, leave=True)
    with torch.inference_mode(), torch.device('cuda'):
        frames_in, _, _ = torchvision.io.read_video(video_file, output_format='TCHW')
        for frame_idx, frame in enumerate(frames_in):
            pred = multiperson_model_pt.detect_poses(frame, skeleton=skeleton, detector_threshold=0.01,
                                                     suppress_implausible_poses=False, max_detections=1,
                                                     intrinsic_matrix=torch.FloatTensor(intrinsic_matrix),
                                                     distortion_coeffs=torch.FloatTensor(distortions), num_aug=2)
            # Save detection's parameters
            bboxes = pred['boxes'].cpu().numpy()
            pose_result_3d = pred['poses3d'].cpu().numpy()

            df = add_to_dataframe(df, pose_result_3d)

            frame_idx += 1
            if verbose >= 1:
                progress.update(1)
        # Release the VideoCapture object and close progressbar

        if verbose >= 1:
            progress.close()

    if verbose >= 2:
        print(f'Call filtering function')
    df_filt = filter_df(df, fps, verbose)

    if not os.path.exists(dir_out_trc):
        os.makedirs(dir_out_trc)

    trc_file_filt = os.path.join(dir_out_trc,
                                 f"{os.path.basename(video_file).split('.mp4')[0]}_0-{frame_idx}_filt_iDrinkbutter.trc")
    trc_file_unfilt = os.path.join(dir_out_trc,
                                   f"{os.path.basename(video_file).split('.mp4')[0]}_0-{frame_idx}_unfilt_iDrink.trc")

    df_to_trc(df_filt, trc_file_filt, identifier, fps, n_frames, n_markers)
    df_to_trc(df, trc_file_unfilt, identifier, fps, n_frames, n_markers)

    if verbose >= 2:
        print(f'3D Pose Estimation done and .trc files saved to {dir_out_trc}')


if __name__ == '__main__':
    args = parser.parse_args()

    if sys.gettrace() is not None or args.DEBUG:
        print("Debug Mode is activated\n"
              "Starting debugging script.")

        if os.name == 'posix':  # if running on WSL
            args.identifier = "S20240501-115510_P01_T01"
            args.model_path = "/mnt/c/iDrink/metrabs_models/pytorch/metrabs_eff2l_384px_800k_28ds_pytorch"
            args.video_file = "/mnt/c/iDrink/Session Data/S20240501-115510/S20240501-115510_P07/S20240501-115510_P07_T44/videos/recordings/cam1_trial_44_R_affected.mp4"
            args.dir_out_video = "/mnt/c/iDrink/Session Data/S20240501-115510/S20240501-115510_P07/S20240501-115510_P07_T44/videos/pose"
            args.dir_out_trc = "/mnt/c/iDrink/Session Data/S20240501-115510/S20240501-115510_P07/S20240501-115510_P07_T44/pose-3d"
            args.calib_file = "/mnt/c/iDrink/Session Data/S20240501-115510/S20240501-115510_Calibration/Calib_S20240501-115510.toml"
            args.skeleton = 'coco_19'
            args.filter_2d = False
            args.verbose = 1

        else:
            args.identifier = "S20240501-115510_P01_T01"
            args.model_path = r"C:\iDrink\metrabs_models\pytorch\metrabs_eff2l_384px_800k_28ds_pytorch"
            args.video_file = r"C:\iDrink\Session Data\S20240501-115510\S20240501-115510_P07\S20240501-115510_P07_T44\videos\recordings\cam1_trial_44_R_affected.mp4"
            args.dir_out_video = r"C:\iDrink\Session Data\S20240501-115510\S20240501-115510_P07\S20240501-115510_P07_T44\videos\pose"
            args.dir_out_trc = r"C:\iDrink\Session Data\S20240501-115510\S20240501-115510_P07\S20240501-115510_P07_T44\pose-3d"
            args.calib_file = r"C:\iDrink\Session Data\S20240501-115510\S20240501-115510_Calibration\Calib_S20240501-115510.toml"
            args.skeleton = 'coco_19'
            args.filter_2d = False
            args.verbose = 1

    metrabs_pose_estimation_3d(args.video_file, args.calib_file, args.dir_out_video, args.dir_out_trc, args.model_path,
                               args.identifier, args.skeleton, args.verbose, args.DEBUG)
    pass
