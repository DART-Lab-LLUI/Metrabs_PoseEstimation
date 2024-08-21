import functools
import glob
import importlib
import itertools
import os.path as osp
import zlib

import barecat
import boxlib
import more_itertools
import msgpack_numpy
import numpy as np
import simplepyutils as spu
from simplepyutils import logger

import posepile.joint_info
from posepile import joint_filtering, util
from posepile.joint_info import JointInfo
from posepile.util import TEST, TRAIN, VALID, geom3d, improc


class Pose3DDataset:
    def __init__(
            self, joint_info, train_examples=None, valid_examples=None, test_examples=None,
            compute_bone_lengths=True):
        self.joint_info = joint_info
        self.examples = {
            TRAIN: train_examples or [],
            VALID: valid_examples or [],
            TEST: test_examples or []}

        if compute_bone_lengths:
            self.update_bones()

    def update_bones(self):
        logger.info('Updating bone lengths')
        trainval_examples = [*self.examples[TRAIN], *self.examples[VALID]]
        if trainval_examples:
            self.trainval_bones = compute_mean_bones(trainval_examples, self.joint_info)
        if self.examples[TRAIN]:
            if not self.examples[VALID]:
                self.train_bones = self.trainval_bones
            else:
                self.train_bones = compute_mean_bones(self.examples[TRAIN], self.joint_info)

    def iter_examples(self):
        return itertools.chain(self.examples[TRAIN], self.examples[VALID], self.examples[TEST])


class Pose3DExample:
    def __init__(
            self, image_path, world_coords, bbox, camera, *,
            activity_name='unknown', scene_name='unknown', mask=None, univ_coords=None,
            instance_mask=None, image=None):
        self.image_path = image_path
        self.world_coords = world_coords
        self.univ_coords = univ_coords if univ_coords is not None else None
        self.bbox = np.asarray(bbox)
        self.camera = camera
        self.activity_name = activity_name
        self.scene_name = scene_name
        self.mask = mask
        self.instance_mask = instance_mask
        self.custom = None
        self.image = image

    def get_world_coords(self):
        if isinstance(self.world_coords, SparseCoords):
            return self.world_coords.to_array()
        else:
            return self.world_coords


class Pose3DDatasetBareCat:
    def __init__(self, annotations_path, images_path):
        self.bc_annotations = get_bc_annotations(annotations_path)
        self.bc_images = get_bc_images(images_path)
        metadata = self.bc_annotations['metadata.msgpack']
        self.joint_info = JointInfo(metadata['joint_names'], metadata['joint_edges'])
        # for i, j in self.joint_info.stick_figure_edges:
        #    print(f'{self.joint_info.names[i]}-{self.joint_info.names[j]}')
        exs = spu.groupby(spu.progressbar(self.bc_annotations), lambda p: p.split('/')[0])
        self.examples = {
            label: [Pose3DExampleBareCat(
                self.joint_info, annotations_path, images_path, p)
                for p in spu.progressbar(exs[name])]
            for label, name in zip([TRAIN, VALID, TEST], ['train', 'val', 'test'])
        }
        self.train_bones = metadata['train_bone_lengths']
        self.trainval_bones = metadata['trainval_bone_lengths']


@functools.lru_cache()
def get_bc_annotations(path):
    return barecat.Reader(path, decoder=msgpack_numpy.unpackb)

@functools.lru_cache()
def get_bc_images(path):
    return barecat.Reader(path, decoder=improc.decode_jpeg)


class Pose3DExampleBareCat:
    def __init__(self, joint_info, bc_annotation_path, bc_image_path, path_in_file):
        self.bc_annotation_path = bc_annotation_path
        self.bc_image_path = bc_image_path
        self.path = path_in_file
        self.image_path = self.path.partition('/')[2]
        self.n_joints = joint_info.n_joints

    def load(self):
        import cameralib
        import rlemasklib
        import cv2
        d = get_bc_annotations(self.bc_annotation_path)[self.path]
        cam = cameralib.Camera(
            rot_world_to_cam=cv2.Rodrigues(d['cam']['rotvec_w2c'])[0],
            optical_center=np.array(d['cam']['loc']),
            intrinsic_matrix=np.concatenate([d['cam']['intr'], np.array([[0, 0, 1]])]),
            distortion_coeffs=d['cam'].get('distcoef', None),
            world_up=d['cam'].get('up', (0, 0, 1))
        )

        world_coords = np.full(
            shape=[self.n_joints, 3], dtype=np.float32, fill_value=np.nan)
        world_coords[d['joints3d']['i_rows']] = d['joints3d']['rows']
        image_path = d['impath']
        image = get_bc_images(self.bc_image_path)[image_path]

        return Pose3DExample(
            image_path=image_path,
            image=image,
            bbox=d['bbox'].astype(np.float32),
            camera=cam,
            world_coords=world_coords,
            mask=rlemasklib.decompress(d['mask']) if 'mask' in d else None
        )


def compute_mean_bones(examples, joint_info):
    n_bones = len(joint_info.stick_figure_edges)
    joints2bones = posepile.joint_info.get_joint2bone_mat(joint_info)
    joints2bones_abs = np.abs(joints2bones)

    mean_bones = np.zeros(shape=[n_bones], dtype=np.float32)
    n_valids = np.zeros(shape=[n_bones], dtype=np.float32)

    for ex_chunk in spu.progressbar(more_itertools.chunked(examples, 256),
                                    total=len(examples) // 256):
        coords = np.stack([ex.get_world_coords() for ex in ex_chunk])
        is_joint_valid = geom3d.are_joints_valid(coords).astype(np.float32)
        is_bone_valid = (joints2bones_abs @ is_joint_valid[..., np.newaxis] == 2)[..., 0]
        n_valids_now = np.count_nonzero(is_bone_valid, axis=0)
        n_valids += n_valids_now
        is_any_valid = np.any(is_bone_valid, axis=0)

        bones = joints2bones @ np.nan_to_num(coords)
        bones[~is_bone_valid] = np.nan
        mean_bones_now = np.nanmean(np.linalg.norm(bones, axis=-1), axis=0)
        fac = n_valids_now[is_any_valid] / n_valids[is_any_valid]
        mean_bones[is_any_valid] = (
                mean_bones_now[is_any_valid] * fac + mean_bones[is_any_valid] * (1 - fac))
    return mean_bones


def add_masks(ds, mask_dir, n_components=None, relative_root=None):
    mask_paths = glob.glob(f'{mask_dir}/*.pkl')
    mask_dict = {}
    for path in mask_paths:
        mask_dict.update(spu.load_pickle(path))

    for ex in itertools.chain(ds.examples[0], ds.examples[1]):
        # for ex in ds.examples[0]:
        if relative_root is None:
            relpath = spu.last_path_components(ex.image_path, n_components)
        else:
            relpath = osp.relpath(
                util.ensure_absolute_path(ex.image_path), util.ensure_absolute_path(relative_root))
        ex.mask = mask_dict[relpath]
    return ds


class SparseCoords:
    def __init__(self, coords):
        self.shape = coords.shape
        self.i_valid_joints = np.where(geom3d.are_joints_valid(coords))[0].astype(np.uint16)
        self.valid_coords = np.ascontiguousarray(coords[self.i_valid_joints])

    def to_array(self):
        result = np.full(shape=self.shape, dtype=self.valid_coords.dtype, fill_value=np.nan)
        result[self.i_valid_joints] = self.valid_coords
        return result

    def invalidate_coords(self, i_bad_coords):
        i_good_joints_among_valid = [
            i_among_valids
            for i_among_valids, i_joint in enumerate(self.i_valid_joints)
            if i_joint not in i_bad_coords]

        self.i_valid_joints = self.i_valid_joints[i_good_joints_among_valid]
        self.valid_coords = np.ascontiguousarray(self.valid_coords[i_good_joints_among_valid])


def compress_if_possible(arr, dtype, thresh):
    compressed = arr.astype(dtype)
    diff = np.nanmax(np.abs(arr - compressed))
    return compressed if diff < thresh else arr


def compress_example(ex):
    ex.world_coords = SparseCoords(compress_if_possible(ex.world_coords, np.float16, 1))
    ex.bbox = compress_if_possible(ex.bbox, np.float16, 3)
    ex.univ_coords = None
    if hasattr(ex, 'mask') and ex.mask is not None and 'counts' in ex.mask:
        ex.mask['zcounts'] = zlib.compress(ex.mask['counts'])
        del ex.mask['counts']


def compress_dataset(ds):
    n_examples = sum(len(x) for x in ds.examples.values())
    for ex in spu.progressbar(ds.iter_examples(), total=n_examples, desc='Compress'):
        compress_example(ex)


def make_surreal_vertices(*args, **kwargs):
    import posepile.ds.surreal.main
    return posepile.ds.surreal.main.make_dataset(store_vertices=True, adaptive_threshold=500)


def make_surreal_vertices_plausible(*args, **kwargs):
    import posepile.ds.surreal.main
    ds = posepile.ds.surreal.main.make_dataset(store_vertices=True, adaptive_threshold=500)
    filter_dataset_by_plausibility(
        ds, min_joints_in_box=int(ds.joint_info.n_joints * 0.75), surreal_always_plausible=False)
    return ds


def make_h36m_incorrect_S9(*args, **kwargs):
    import posepile.ds.h36m.main
    return posepile.ds.h36m.main.make_dataset(*args, **kwargs, correct_S9=False)


def make_h36m_alljoints(*args, **kwargs):
    import posepile.ds.h36m.main
    return posepile.ds.h36m.main.make_h36m_alljoints(*args, **kwargs)


def make_h36m_partial(*args, **kwargs):
    import posepile.ds.h36m.main
    return posepile.ds.h36m.main.make_dataset(*args, **kwargs, partial_visibility=True)


@spu.picklecache(f'3dhp_full_with_mupots.pkl', min_time="2020-06-29T21:16:09")
def make_3dhp_full():
    import posepile.ds.tdhp.full
    ds = posepile.ds.tdhp.full.make_dataset()
    import posepile.ds.mupots.main
    mupots = posepile.ds.mupots.main.make_mupots_yolo()
    mupots = posepile.joint_filtering.convert_dataset(mupots, ds.joint_info, update_bones=False)
    ds.examples[VALID] = mupots.examples[VALID]
    ds.examples[TEST] = mupots.examples[TEST]
    return ds


def make_mpi_inf_3dhp_correctedTS6():
    import posepile.ds.tdhp.main
    return posepile.ds.tdhp.main.make_dataset(ts6_corr=True)


def make_mupots_yolo():
    import posepile.ds.mupots.main
    return posepile.ds.mupots.main.make_mupots_yolo()


@spu.picklecache('muco_17_150k.pkl', min_time="2021-09-10T00:00:04")
def make_muco_17_150k():
    ds = get_dataset('muco')

    # Take only the first 150,000 composite images
    def get_image_id(ex):
        return int(osp.basename(ex.image_path).split('_')[0])

    ds.examples[TRAIN] = [e for e in ds.examples[0] if get_image_id(e) < 150000]

    # Take only the 17 MuPoTS joints
    mupots = get_dataset('mupots_yolo')
    posepile.joint_filtering.convert_dataset(ds, mupots.joint_info)
    ds.examples[VALID] = mupots.examples[VALID]
    ds.examples[TEST] = mupots.examples[TEST]
    ds.update_bones()
    return ds


@spu.picklecache('muco_full.pkl', min_time="2021-09-10T00:00:04")
def make_muco_full():
    ds = get_dataset('muco')
    mupots = get_dataset('mupots_yolo')
    mupots = joint_filtering.convert_dataset(mupots, ds.joint_info, update_bones=False)
    ds.examples[VALID] = mupots.examples[VALID]
    ds.examples[TEST] = mupots.examples[TEST]
    return ds


def make_huge8():
    import posepile.merging.merged_dataset3d
    return posepile.merging.merged_dataset3d.make_huge8()


def make_huge8_dummy():
    import posepile.merging.merged_dataset3d
    return posepile.merging.merged_dataset3d.make_huge8_dummy()


def make_small5():
    import posepile.merging.merged_dataset3d
    return posepile.merging.merged_dataset3d.make_small5()


def make_medium3():
    import posepile.merging.merged_dataset3d
    return posepile.merging.merged_dataset3d.make_medium3()


def make_muco_3dhp_200k():
    muco_3dhp = get_dataset('muco')

    def get_image_id(ex):
        return int(osp.basename(ex.image_path).split('_')[0])

    muco_3dhp.examples[0] = [e for e in muco_3dhp.examples[0] if get_image_id(e) < 200000]
    return muco_3dhp


######
def filter_dataset_by_plausibility(
        dataset, relsmall_thresh=0.1, relbig_thresh=3, absbig_thresh=150, min_joints_in_box=4,
        set_to_nan_instead_of_removal=False, piano1=False, surreal_always_plausible=True):
    joints2bones = posepile.joint_info.get_joint2bone_mat(dataset.joint_info)
    joints2bones_abs = np.abs(joints2bones)

    def get_bone_lengths(world_coords):
        is_joint_valid = geom3d.are_joints_valid(world_coords).astype(np.float32)
        is_bone_valid = joints2bones_abs @ is_joint_valid[:, np.newaxis] == 2
        is_bone_valid = np.squeeze(is_bone_valid, -1)
        bones = joints2bones @ np.nan_to_num(world_coords)
        bone_lengths = np.linalg.norm(bones, axis=-1)
        bone_lengths[~is_bone_valid] = np.nan
        return bone_lengths

    dataset.update_bones()
    train_bones = np.asarray(dataset.train_bones)
    denominator = 1 / (train_bones + 1e-8)

    def is_plausible(ex):
        # I thought all surreal poses should be considered valid, since it's synthetic
        # but actually there are some cases when the image doesn't contain the person
        # because the person walks outside the frame. These should have been removed.
        # Training with surreal_always_plausible=False would be preferable when doing it again.
        if surreal_always_plausible and 'surreal' in ex.image_path:
            return True
        if not piano1 and '161029_piano1' in ex.image_path:
            return False

        world_coords = ex.get_world_coords()
        imcoords = ex.camera.world_to_image(world_coords)
        n_joints_within_bbox = np.count_nonzero(boxlib.contains(ex.bbox, imcoords))
        if n_joints_within_bbox < min_joints_in_box:
            return False

        bone_lengths = get_bone_lengths(world_coords)
        bone_length_relative = bone_lengths * denominator
        bone_length_diff = np.abs(bone_lengths - train_bones)

        with np.errstate(invalid='ignore'):
            relsmall = bone_length_relative < relsmall_thresh
            relbig = bone_length_relative > relbig_thresh
            absdiffbig = bone_length_diff > absbig_thresh

        is_bone_implausible = np.logical_and(np.logical_or(relbig, relsmall), absdiffbig)
        any_bone_implausible = np.any(is_bone_implausible)
        if any_bone_implausible:
            print(ex.image_path)

        if set_to_nan_instead_of_removal:
            bad_joints = [
                j for implausible, edge in zip(
                    is_bone_implausible, dataset.joint_info.stick_figure_edges)
                if implausible
                for j in edge]

            if isinstance(ex.world_coords, SparseCoords):
                ex.world_coords.invalidate_coords(bad_joints)
            else:
                ex.world_coords[bad_joints] = np.nan

            imcoords = ex.camera.world_to_image(world_coords)
            n_joints_within_bbox = np.count_nonzero(boxlib.contains(ex.bbox, imcoords))
            return n_joints_within_bbox >= min_joints_in_box

        return not any_bone_implausible

    keep_if(dataset.examples[0], is_plausible)
    dataset.update_bones()


def keep_if(list_, predicate):
    i = 0
    n_before = len(list_)
    while i < len(list_):
        if not predicate(list_[i]):
            del list_[i]
        else:
            i += 1
    n_after = len(list_)
    removed_ratio = (n_before - n_after) / n_before
    logger.info(f'Removed {removed_ratio:.1%}, {n_before - n_after}')
    return list_


def get_dataset(dataset_name, *args, **kwargs):
    from simplepyutils import FLAGS

    # Datasets can be loaded from pickle files
    if dataset_name.endswith('.pkl'):
        return spu.load_pickle(util.ensure_absolute_path(dataset_name))

    # Datasets can be loaded from barecat files
    if dataset_name.endswith('.barecat'):
        return Pose3DDatasetBareCat(dataset_name, images_path=FLAGS.image_barecat_path)

    logger.debug(f'Making dataset {dataset_name}...')

    # Training and validation subjects can be specified as command line arguments
    # for Human3.6M
    kwargs = {}

    def string_to_intlist(string):
        return tuple(int(s) for s in string.split(','))

    for subj_key in ['train_subjects', 'valid_subjects', 'test_subjects']:
        if hasattr(FLAGS, subj_key) and getattr(FLAGS, subj_key):
            kwargs[subj_key] = string_to_intlist(getattr(FLAGS, subj_key))

    try:
        # Some datasets (or their variants) are defined in this file
        make_fn = globals()[f'make_{dataset_name}']
    except KeyError:
        try:
            # Others are defined in their own modules
            dataset_module = importlib.import_module(f'posepile.ds.{dataset_name}.main')
        except ImportError:
            raise ValueError(f'Dataset {dataset_name} not found.')

        # Each dataset module should has a make_dataset function
        make_fn = dataset_module.make_dataset

    return make_fn(*args, **kwargs)


@functools.lru_cache()
def get_dataset_cached(dataset_name):
    return get_dataset(dataset_name)


def get_compressed_dataset(dataset_name):
    dataset = get_dataset(dataset_name)
    compress_dataset(dataset)
    return dataset


@functools.lru_cache()
@spu.picklecache('joint_info')
def get_joint_info(dataset_name):
    return get_dataset(dataset_name).joint_info
