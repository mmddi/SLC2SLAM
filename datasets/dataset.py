import glob
import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from .utils import get_camera_rays, as_intrinsics_matrix
from model import image_utils
import csv
import zipfile

def get_dataset(config):
    '''
    Get the dataset class from the config file.
    '''
    if config['dataset'] == 'replica':
        dataset = ReplicaDataset

    elif config['dataset'] == 'scannet':
        dataset = ScannetDataset


    elif config['dataset'] == 'azure':
        dataset =  Azure


    return dataset(config,
                   config['data']['datadir'],
                   trainskip=config['data']['trainskip'],
                   downsample_factor=config['data']['downsample'],
                   sc_factor=config['data']['sc_factor'])

class BaseDataset(Dataset):
    def __init__(self, cfg):
        self.png_depth_scale = cfg['cam']['png_depth_scale']
        self.H, self.W = cfg['cam']['H']//cfg['data']['downsample'],\
            cfg['cam']['W']//cfg['data']['downsample']

        self.fx, self.fy =  cfg['cam']['fx']//cfg['data']['downsample'],\
             cfg['cam']['fy']//cfg['data']['downsample']
        self.cx, self.cy = cfg['cam']['cx']//cfg['data']['downsample'],\
             cfg['cam']['cy']//cfg['data']['downsample']





        self.distortion = np.array(
            cfg['cam']['distortion']) if 'distortion' in cfg['cam'] else None
        self.crop_size = cfg['cam']['crop_edge'] if 'crop_edge' in cfg['cam'] else 0
        self.ignore_w = cfg['tracking']['ignore_edge_W']
        self.ignore_h = cfg['tracking']['ignore_edge_H']

        self.total_pixels = (self.H - self.crop_size*2) * (self.W - self.crop_size*2)
        self.num_rays_to_save = int(self.total_pixels * cfg['mapping']['n_pixels'])


    def __len__(self):
        raise NotImplementedError()

    def __getitem__(self, index):
        raise NotImplementedError()


class Azure(BaseDataset):
    def __init__(self, cfg, basedir, trainskip=1, downsample_factor=1, translation=0.0, sc_factor=1., crop=0,
                 mode="nyu34"):

        super(Azure, self).__init__(cfg)
        self.datadir = basedir
        self.config = cfg
        self.basedir = basedir
        self.trainskip = trainskip
        self.downsample_factor = downsample_factor
        self.translation = translation
        self.sc_factor = sc_factor
        self.crop = crop
        self.img_files = sorted(
            glob.glob(os.path.join(self.datadir, 'color', '*.jpg')))
        self.color_paths = sorted(
            glob.glob(os.path.join(self.datadir, 'color', '*.jpg')))
        self.depth_paths = sorted(
            glob.glob(os.path.join(self.datadir, 'depth', '*.png')))
        self.num_frames = len(self.color_paths)
        self.load_poses(os.path.join(
            self.datadir, 'scene', 'trajectory.log'))
        self.K = None
        self.rays_d = None
        self.tracking_mask = None
        self.frame_ids = range(0, len(self.img_files))
        self.image_paths = sorted(
            glob.glob(os.path.join(self.datadir, 'color', '*.jpg')))
        self.colour_map_np = None

    def __len__(self):
        return self.num_frames

    def __getitem__(self, index):

        color_path = self.img_files[index]
        depth_path = self.depth_paths[index]

        color_data = cv2.imread(color_path)
        if '.png' in depth_path:
            depth_data = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        elif '.exr' in depth_path:
            raise NotImplementedError()
        if self.distortion is not None:
            raise NotImplementedError()


        color_data = cv2.cvtColor(color_data, cv2.COLOR_BGR2RGB)
        color_data = color_data / 255.

        # 深度图转换为浮点数
        depth_data = depth_data.astype(np.float32) / self.png_depth_scale * self.sc_factor

        H, W = depth_data.shape


        color_data = cv2.resize(color_data, (W, H))

        if self.downsample_factor > 1:  # 将图像或数据在空间上减小尺寸
            H = H // self.downsample_factor
            W = W // self.downsample_factor
            color_data = cv2.resize(color_data, (W, H), interpolation=cv2.INTER_AREA)
            depth_data = cv2.resize(depth_data, (W, H), interpolation=cv2.INTER_NEAREST)

        if self.rays_d is None:
            self.rays_d = get_camera_rays(self.H, self.W, self.fx, self.fy, self.cx, self.cy)

        color_data = torch.from_numpy(color_data.astype(np.float32))
        depth_data = torch.from_numpy(depth_data.astype(np.float32))
        # semantic_data = torch.from_numpy(semantic_data.astype(np.float32))

        fx = self.fx
        fy = self.fy
        cx = self.cx
        cy = self.cy

        # 构建相机内参数矩阵 K
        self.K = np.array([[fx, 0, cx],
                           [0, fy, cy],
                           [0, 0, 1]])

        ret = {
            "frame_id": self.frame_ids[index],
            "c2w": self.poses[index],
            "rgb": color_data,
            "depth": depth_data,
            "direction": self.rays_d,
            "intrinsic": torch.from_numpy(self.K),
            "path": self.image_paths,
            # "semantic": semantic_data,
            "num_semantic": 40
        }

        return ret

    def load_poses(self, path):
        self.poses = []
        if os.path.exists(path):
            with open(path) as f:
                content = f.readlines()

                # Load .log file.
                for i in range(0, len(content), 5):
                    # format %d (src) %d (tgt) %f (fitness)
                    data = list(map(float, content[i].strip().split(' ')))
                    ids = (int(data[0]), int(data[1]))
                    fitness = data[2]

                    # format %f x 16
                    c2w = np.array(
                        list(map(float, (''.join(
                            content[i + 1:i + 5])).strip().split()))).reshape((4, 4))

                    c2w[:3, 1] *= -1
                    c2w[:3, 2] *= -1
                    c2w = torch.from_numpy(c2w).float()
                    self.poses.append(c2w)
        else:
            for i in range(self.num_frames):
                c2w = np.eye(4)
                c2w = torch.from_numpy(c2w).float()
                self.poses.append(c2w)


class ReplicaDataset(BaseDataset):
    def __init__(self, cfg, basedir, trainskip=1,
                 downsample_factor=1, translation=0.0,
                 sc_factor=1., crop=0, mode="nyu34"):
        super(ReplicaDataset, self).__init__(cfg)

        self.config = cfg
        self.basedir = basedir
        self.trainskip = trainskip
        self.downsample_factor = downsample_factor
        self.translation = translation
        self.sc_factor = sc_factor
        self.crop = crop
        self.img_files = sorted(glob.glob(f'{self.basedir}/frames/frame*.jpg'))
        self.depth_paths = sorted(
            glob.glob(f'{self.basedir}/frames/depth*.png'))
        self.load_poses(os.path.join(self.basedir, 'traj.txt'))
        self.path = os.path.join(self.basedir, 'frames')

        self.K = None
        self.rays_d = None
        self.tracking_mask = None
        self.frame_ids = range(0, len(self.img_files))
        self.num_frames = len(self.frame_ids)
        self.image_paths = sorted(glob.glob(os.path.join(self.path, 'frame*.jpg')))
        self.colour_map_np = None


    def __len__(self):
        return self.num_frames

    def __getitem__(self, index):
        # 读取颜色图和深度图
        color_path = self.img_files[index]
        depth_path = self.depth_paths[index]

        color_data = cv2.imread(color_path)
        if '.png' in depth_path:
            depth_data = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        elif '.exr' in depth_path:
            raise NotImplementedError()
        if self.distortion is not None:
            raise NotImplementedError()


        color_data = cv2.cvtColor(color_data, cv2.COLOR_BGR2RGB)
        color_data = color_data / 255.


        depth_data = depth_data.astype(np.float32) / self.png_depth_scale * self.sc_factor

        H, W = depth_data.shape


        color_data = cv2.resize(color_data, (W, H))

        if self.downsample_factor > 1:  # 将图像或数据在空间上减小尺寸
            H = H // self.downsample_factor
            W = W // self.downsample_factor
            color_data = cv2.resize(color_data, (W, H), interpolation=cv2.INTER_AREA)
            depth_data = cv2.resize(depth_data, (W, H), interpolation=cv2.INTER_NEAREST)

        if self.rays_d is None:
            self.rays_d = get_camera_rays(self.H, self.W, self.fx, self.fy, self.cx, self.cy)

        color_data = torch.from_numpy(color_data.astype(np.float32))
        depth_data = torch.from_numpy(depth_data.astype(np.float32))

        fx = self.fx
        fy = self.fy
        cx = self.cx
        cy = self.cy


        self.K = np.array([[fx, 0, cx],
                           [0, fy, cy],
                           [0, 0, 1]])

        ret = {
            "frame_id": self.frame_ids[index],
            "c2w": self.poses[index],
            "rgb": color_data,
            "depth": depth_data,
            "direction": self.rays_d,
            "intrinsic": torch.from_numpy(self.K),
            "path": self.image_paths,
        }

        return ret

    def get_frame_data(self, index):
        color_path = self.image_paths[index]
        depth_path = self.depth_paths[index]
        color_data = cv2.imread(color_path)
        if '.png' in depth_path:
            depth_data = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        elif '.exr' in depth_path:
            raise NotImplementedError()

        depth_data = depth_data.astype(np.float32) / self.png_depth_scale * self.sc_factor

        return color_data, depth_data

    def get_depth_image(self, index):
        depth_path = self.depth_paths[index]
        if '.png' in depth_path:
            depth_data = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        elif '.exr' in depth_path:
            raise NotImplementedError()
        depth_data = depth_data.astype(np.float32) / self.png_depth_scale * self.sc_factor

        if self.config['cam']['crop_edge'] > 0:
            edge = self.config['cam']['crop_edge']
            depth_data = depth_data[edge:-edge, edge:-edge]

        return depth_data

    def load_poses(self, path):
        self.poses = []
        with open(path, "r") as f:
            lines = f.readlines()
        for i in range(len(self.img_files)):
            line = lines[i]
            c2w = np.array(list(map(float, line.split()))).reshape(4, 4)
            c2w[:3, 1] *= -1
            c2w[:3, 2] *= -1
            c2w[:3, 3] *= self.sc_factor
            c2w = torch.from_numpy(c2w).float()
            self.poses.append(c2w)



class ScannetDataset(BaseDataset):
    def __init__(self, cfg, basedir, trainskip=1, downsample_factor=1, translation=0.0,
                 sc_factor=1., crop=0, mode="nyu40"):

        super(ScannetDataset, self).__init__(cfg)

        self.config = cfg
        self.basedir = basedir
        self.trainskip = trainskip
        self.downsample_factor = downsample_factor
        self.translation = translation
        self.sc_factor = sc_factor
        self.crop = crop
        self.img_files = sorted(glob.glob(os.path.join(self.basedir, 'color', '*.jpg')),
                                key=lambda x: int(os.path.basename(x)[:-4]))

        self.depth_paths = sorted(glob.glob(os.path.join(self.basedir, 'depth', '*.png')),
                                  key=lambda x: int(os.path.basename(x)[:-4]))

        self.load_poses(os.path.join(self.basedir, 'pose'))
        self.label_zip_pattern = "scene*_00_2d-label-filt.zip"

        self.extract_dir = os.path.join(self.basedir, "label-filt-unzipped")

        self.label_zip_path = self.find_zip_file()


        self.Unzip()


        self.label_file_path = sorted(
            glob.glob(os.path.join(self.extract_dir,"label-filt", "*.png")),
            key=lambda x: int(os.path.basename(x)[:-4])
        )


        self.path = os.path.join(self.basedir, 'color')
        self.image_paths = sorted(glob.glob(os.path.join(self.path, '*.jpg')))
        self.rays_d = None
        self.frame_ids = range(0, len(self.img_files))
        self.num_frames = len(self.frame_ids)
        self.frames_file_list = []
        if mode == "nyu40":
            self.label_mapping_nyu = self.load_scannet_nyu40_mapping(self.basedir)  # 加载 ScanNet 到 NYU40 标签定义的映射。
            self.colour_map_np = image_utils.nyu40_colour_code
            assert self.colour_map_np.shape[0] == 41
        elif mode == "nyu13":
            self.label_mapping_nyu = self.load_scannet_nyu13_mapping(self.basedir)
            self.colour_map_np = image_utils.nyu13_colour_code
            assert self.colour_map_np.shape[0] == 14
        else:
            assert False
        self.near = 0
        self.far = 8
        if self.config['cam']['crop_edge'] > 0:
            self.H -= self.config['cam']['crop_edge'] * 2
            self.W -= self.config['cam']['crop_edge'] * 2
            self.cx -= self.config['cam']['crop_edge']
            self.cy -= self.config['cam']['crop_edge']

    def __len__(self):
        return self.num_frames

    def __getitem__(self, index):
        self.near = self.config['cam']['near']
        self.far = self.config['cam']['far']
        color_path = self.img_files[index]
        depth_path = self.depth_paths[index]

        label_data = cv2.imread(self.label_file_path[index], cv2.IMREAD_UNCHANGED)

        color_data = cv2.imread(color_path)
        if '.png' in depth_path:
            depth_data = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        elif '.exr' in depth_path:
            raise NotImplementedError()
        if self.distortion is not None:
            raise NotImplementedError()
        color_data = cv2.cvtColor(color_data, cv2.COLOR_BGR2RGB)
        color_data = color_data / 255.
        depth_data = depth_data.astype(np.float32) / self.png_depth_scale * self.sc_factor
        H, W = depth_data.shape
        color_data = cv2.resize(color_data, (W, H))

        semantic_data = cv2.resize(label_data, (W, H), interpolation=cv2.INTER_NEAREST)
        semantic_raw = semantic_data
        semantic_nyu = semantic_raw.copy()

        for scan_id, nyu_id in self.label_mapping_nyu.items():
            semantic_nyu[semantic_raw == scan_id] = nyu_id

        semantic_data = semantic_nyu

        if self.downsample_factor > 1:
            H = H // self.downsample_factor
            W = W // self.downsample_factor
            self.fx = self.fx // self.downsample_factor
            self.fy = self.fy // self.downsample_factor
            color_data = cv2.resize(color_data, (W, H), interpolation=cv2.INTER_AREA)
            depth_data = cv2.resize(depth_data, (W, H), interpolation=cv2.INTER_NEAREST)
            semantic_data = cv2.resize(semantic_data, (W, H), interpolation=cv2.INTER_NEAREST)
            # instance_data = cv2.resize(instance, (W, H), interpolation=cv2.INTER_NEAREST)
        edge = self.config['cam']['crop_edge']
        if edge > 0:
            # crop image edge, there are invalid value on the edge of the color image
            color_data = color_data[edge:-edge, edge:-edge]
            depth_data = depth_data[edge:-edge, edge:-edge]
            semantic_data = semantic_data[edge:-edge, edge:-edge]
            # instance_data = instance_data[edge:-edge, edge:-edge]

        if self.rays_d is None:
            self.rays_d = get_camera_rays(self.H, self.W, self.fx, self.fy, self.cx, self.cy)
        color_data = torch.from_numpy(color_data.astype(np.float32))
        # color_data = torch.from_numpy(color_data)
        depth_data = torch.from_numpy(depth_data.astype(np.float32))
        semantic_data = torch.from_numpy(semantic_data.astype(np.float32))
        # instance_data = torch.from_numpy(instance_data.astype(np.float32))
        fx = self.fx
        fy = self.fy
        cx = self.cx
        cy = self.cy
        # 构建相机内参数矩阵 K
        K = np.array([[fx, 0, cx],
                      [0, fy, cy],
                      [0, 0, 1]])

        ret = {
            "frame_id": self.frame_ids[index],
            "c2w": self.poses[index],
            "rgb": color_data,
            "depth": depth_data,
            "direction": self.rays_d,
            "intrinsic": torch.from_numpy(K),
            "semantic": semantic_data,
            "num_semantic": 41
        }

        return ret

    def find_zip_file(self):
        """
        查找符合条件的压缩包文件。如果找到多个，抛出错误。
        """
        zip_files = glob.glob(os.path.join(self.basedir, self.label_zip_pattern))
        if not zip_files:
            raise FileNotFoundError(f"No zip file matching pattern {self.label_zip_pattern} found in {self.basedir}")
        if len(zip_files) > 1:
            raise ValueError(f"Multiple zip files found: {zip_files}. Please specify the correct one.")
        return zip_files[0]

    def Unzip(self):
        """
        Extract the zip file
        """
        if not os.path.exists(self.extract_dir):
            os.makedirs(self.extract_dir, exist_ok=True)
            with zipfile.ZipFile(self.label_zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.extract_dir)
        else:
            print(f"Files already extracted to {self.extract_dir}")
    def remap_semantic_data(self, semantic_data, label_mapping):
        # 遍历语义数据中的每个像素
        for scan_id, nyu_id in label_mapping.items():
            # 将 ScanNet 类别映射到 NYU40 类别
            semantic_data[semantic_data == scan_id] = nyu_id
        return semantic_data


    def load_scannet_nyu40_mapping(self, path):
        """ Returns a dict mapping scannet Ids to NYU40 Ids

        Args:
            path: Path to the original scannet data.
                This is used to get scannetv2-labels.combined.tsv

        Returns:
            mapping: A dict from ints to ints
                example:
                    {1: 1,
                     2: 5,
                     22: 23}

        """

        mapping = {}
        with open(os.path.join(path, 'scannetv2-labels.combined.tsv')) as tsvfile:
            tsvreader = csv.reader(tsvfile, delimiter='\t')
            for i, line in enumerate(tsvreader):
                if i == 0:
                    continue
                scannet_id, nyu40id = int(line[0]), int(line[4])
                mapping[scannet_id] = nyu40id

        return mapping

    def load_scannet_nyu13_mapping(self, path):
        """ Returns a dict mapping scannet Ids to NYU13 Ids

        Args:
            path: Path to the original scannet data.
                This is used to get scannetv2-labels.combined.tsv

        Returns:
            mapping: A dict from ints to ints
                example:
                    {1: 1,
                     2: 5,
                     22: 23}

        """

        mapping = {}
        with open(os.path.join(path, 'scannetv2-labels.combined.tsv')) as tsvfile:
            tsvreader = csv.reader(tsvfile, delimiter='\t')
            for i, line in enumerate(tsvreader):
                if i == 0:
                    continue
                scannet_id, nyu13id = int(line[0]), int(line[5])
                mapping[scannet_id] = nyu13id

        return mapping

    def get_frame_data(self, index):

        color_path = self.image_paths[index]
        color_data = cv2.imread(color_path)

        depth_path = self.depth_paths[index]
        if '.png' in depth_path:
            depth_data = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        elif '.exr' in depth_path:
            raise NotImplementedError()
        depth_data = depth_data.astype(np.float32) / self.png_depth_scale * self.sc_factor

        color_data = cv2.cvtColor(color_data, cv2.COLOR_BGR2RGB)
        color_data = color_data / 255.
        depth_data = depth_data.astype(np.float32) / self.png_depth_scale * self.sc_factor

        H, W = depth_data.shape
        color_data = cv2.resize(color_data, (W, H))

        if self.downsample_factor > 1:
            H = H // self.downsample_factor
            W = W // self.downsample_factor
            self.fx = self.fx // self.downsample_factor
            self.fy = self.fy // self.downsample_factor
            color_data = cv2.resize(color_data, (W, H), interpolation=cv2.INTER_AREA)
            depth_data = cv2.resize(depth_data, (W, H), interpolation=cv2.INTER_NEAREST)

        edge = self.config['cam']['crop_edge']
        if edge > 0:
            # crop image edge, there are invalid value on the edge of the color image
            color_data = color_data[edge:-edge, edge:-edge]
            depth_data = depth_data[edge:-edge, edge:-edge]

        return color_data, depth_data

    def get_depth_image(self, index):
        depth_path = self.depth_paths[index]
        if '.png' in depth_path:
            depth_data = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        elif '.exr' in depth_path:
            raise NotImplementedError()
        depth_data = depth_data.astype(np.float32) / self.png_depth_scale * self.sc_factor

        if self.config['cam']['crop_edge'] > 0:
            edge = self.config['cam']['crop_edge']
            depth_data = depth_data[edge:-edge, edge:-edge]

        return depth_data

    def load_poses(self, path):
        self.poses = []
        pose_paths = sorted(glob.glob(os.path.join(path, '*.txt')),
                            key=lambda x: int(os.path.basename(x)[:-4]))
        for pose_path in pose_paths:
            with open(pose_path, "r") as f:
                lines = f.readlines()
            ls = []
            for line in lines:
                l = list(map(float, line.split(' ')))
                ls.append(l)
            c2w = np.array(ls).reshape(4, 4)
            c2w[:3, 1] *= -1
            c2w[:3, 2] *= -1
            c2w = torch.from_numpy(c2w).float()
            self.poses.append(c2w)


