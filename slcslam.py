# -*- coding: utf-8 -*-

import os
#os.environ['TCNN_CUDA_ARCHITECTURES'] = '86'


# Package imports
import model.Descriptors
import torch
import torch.optim as optim
import numpy as np
import random
import torch.nn.functional as F
import argparse
import shutil
import json
import cv2
import matplotlib
matplotlib.use('Agg')
from torch.utils.data import DataLoader
from tqdm import tqdm
import config
from model.scene_rep import JointEncoding
from model.keyframe import KeyFrameDatabase
from datasets.dataset import get_dataset
from utils import coordinates, extract_mesh, colormap_image
from tools.eval_ate import pose_evaluation
from optimization.utils import at_to_transform_matrix, qt_to_transform_matrix, matrix_to_axis_angle, matrix_to_quaternion
from model.loop_closure import PlaceRecogDatabase
from model.Descriptors import *
from model.loop_opt import PoseGraph,plot_and_save,pypose
import argparse
from scipy.spatial.transform import Rotation as R
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
from model.point_cloud_icp import generate_point_cloud, icp,visualize_semantic_segmentation,plot_semantic_map,apply_nyu40_color_map

from src.common import get_rays, sample_pdf, normalize_3d_coordinate
class SLC2SLAM():
    def __init__(self, config,size):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dataset = get_dataset(config)
        self.tree = None
        self.create_bounds()
        self.create_pose_data()
        self.get_pose_representation()
        self.keyframeDatabase = self.create_kf_database(config)
        self.model = JointEncoding(config, self.bounding_box).to(self.device)
        self.placeDatabase = PlaceRecogDatabase(config, self.device, aggregate='vlad')
        self.con_frameid = []
        self.rel_id = [0] * size
        self.enable_semantic = False
        self.sem_logits = []
        self.indice = []
        self.sem_id = []
        self.con_data = {}
        self.t = 1.8
        self.num_samples = 8192
        self.train = True

    def seed_everything(self, seed):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)


    #获取姿态表示形式
    def get_pose_representation(self):
        '''
        Get the pose representation axis-angle or quaternion
        '''

        #轴角
        if self.config['training']['rot_rep'] == 'axis_angle':
            self.matrix_to_tensor = matrix_to_axis_angle
            self.matrix_from_tensor = at_to_transform_matrix
            print('Using axis-angle as rotation representation, identity init would cause inf')
        #四元数
        elif self.config['training']['rot_rep'] == "quat":
            print("Using quaternion as rotation representation")
            self.matrix_to_tensor = matrix_to_quaternion
            self.matrix_from_tensor = qt_to_transform_matrix
        else:
            raise NotImplementedError
    #创建姿态数据
    def create_pose_data(self):
        '''
        Create the pose data
        '''
        self.est_c2w_data = {}
        self.est_c2w_data_rel = {}
        self.load_gt_pose()
        self.est_c2w_data_keyrel = {}

    #获取场景的预定义边界信息
    def create_bounds(self):
        '''
        Get the pre-defined bounds for the scene
        '''
        self.bounding_box = torch.from_numpy(np.array(self.config['mapping']['bound'])).to(self.device)
        self.marching_cube_bound = torch.from_numpy(np.array(self.config['mapping']['marching_cubes_bound'])).to(self.device)

    def create_kf_database(self, config):  
        '''
        Create the keyframe database
        '''
        num_kf = int(self.dataset.num_frames // self.config['mapping']['keyframe_every'] + 1)  
        print('#kf:', num_kf)
        print('#Pixels to save:', self.dataset.num_rays_to_save)
        return KeyFrameDatabase(config, 
                                self.dataset.H, 
                                self.dataset.W, 
                                num_kf, 
                                self.dataset.num_rays_to_save, 
                                self.device)

    def load_gt_pose(self):
        '''
        Load the ground truth pose
        '''
        self.pose_gt = {}
        for i, pose in enumerate(self.dataset.poses):

            self.pose_gt[i] = pose

    def save_state_dict(self, save_path):
        torch.save(self.model.state_dict(), save_path)
    
    def load(self, load_path):
        self.model.load_state_dict(torch.load(load_path))
    
    def save_ckpt(self, save_path):
        '''
        Save the model parameters and the estimated pose
        '''
        save_dict = {'pose': self.est_c2w_data,
                     'pose_rel': self.est_c2w_data_rel,
                     'model': self.model.state_dict()}
        torch.save(save_dict, save_path)
        print('Save the checkpoint')

    def load_ckpt(self, load_path):
        '''
        Load the model parameters and the estimated pose
        '''
        dict = torch.load(load_path)
        self.model.load_state_dict(dict['model'])
        self.est_c2w_data = dict['pose']
        self.est_c2w_data_rel = dict['pose_rel']

    def select_samples(self, H, W, samples):
        '''
        randomly select samples from the image
        '''
        #indice = torch.randint(H*W, (samples,))
        indice = random.sample(range(H * W), int(samples))
        indice = torch.tensor(indice)
        return indice

    def get_loss_from_ret(self, ret,enable_semantic=False, rgb=True, sdf=True, depth=True, fs=True, smooth=False):
        '''
        Get the training loss
        '''
        loss = 0
        if enable_semantic:
            if rgb:
                loss += self.config['training']['rgb_weight']/10 * ret['rgb_loss']
            if depth:
                loss += self.config['training']['depth_weight']/10 * ret['depth_loss']
            if sdf:
                loss += self.config['training']['sdf_weight']/10 * ret["sdf_loss"]
            if fs:
                loss +=  self.config['training']['fs_weight']/10 * ret["fs_loss"]


            loss += self.config['training']['sem_weight'] * ret["sem_loss"]

            if smooth and self.config['training']['smooth_weight']>0:
                loss += self.config['training']['smooth_weight'] * self.smoothness(self.config['training']['smooth_pts'],
                                                                                      self.config['training']['smooth_vox'],
                                                                                      margin=self.config['training']['smooth_margin'])
        else:
            if rgb:
                loss += self.config['training']['rgb_weight'] * ret['rgb_loss']
            if depth:
                loss += self.config['training']['depth_weight'] * ret['depth_loss']
            if sdf:
                loss += self.config['training']['sdf_weight'] * ret["sdf_loss"]
            if fs:
                loss += self.config['training']['fs_weight'] * ret["fs_loss"]

            if smooth and self.config['training']['smooth_weight'] > 0:
                loss += self.config['training']['smooth_weight'] * self.smoothness(
                    self.config['training']['smooth_pts'],
                    self.config['training']['smooth_vox'],
                    margin=self.config['training']['smooth_margin'])

        return loss

    def first_frame_mapping(self, batch, n_iters=100):
        '''
        First frame mapping
        Params:
            batch['c2w']: [1, 4, 4]
            batch['rgb']: [1, H, W, 3]
            batch['depth']: [1, H, W, 1]
            batch['direction']: [1, H, W, 3]
        Returns:
            ret: dict
            loss: float

        '''
        print('First frame mapping...')
        c2w = batch['c2w'][0].to(self.device)
        self.est_c2w_data[0] = c2w
        self.est_c2w_data_rel[0] = c2w
        self.est_c2w_data_keyrel[0] = c2w
        self.model.train()

        # Training
        for i in range(n_iters):
            self.map_optimizer.zero_grad()
            indice = self.select_samples(self.dataset.H, self.dataset.W, self.config['mapping']['sample'])

            indice_h, indice_w = indice % (self.dataset.H), indice // (self.dataset.H)
            rays_d_cam = batch['direction'].squeeze(0)[indice_h, indice_w, :].to(self.device)
            target_s = batch['rgb'].squeeze(0)[indice_h, indice_w, :].to(self.device)
            target_d = batch['depth'].squeeze(0)[indice_h, indice_w].to(self.device).unsqueeze(-1)





            rays_o = c2w[None, :3, -1].repeat(self.config['mapping']['sample'], 1)
            rays_d = torch.sum(rays_d_cam[..., None, :] * c2w[:3, :3], -1)
            if self.train:
                target_sem = batch['semantic'].squeeze(0)[indice_h, indice_w].to(self.device).unsqueeze(-1)
            else:

                target_sem = None

            # Forward
            # ret = self.model.forward(rays_o, rays_d, target_s, target_d)
            ret,features = self.model.forward(batch, rays_o, rays_d, target_s, target_d, target_sem, self.enable_semantic)
            loss = self.get_loss_from_ret(ret)
            loss.backward()
            self.map_optimizer.step()


        # First frame will always be a keyframe
        self.keyframeDatabase.add_keyframe(batch,self.train, filter_depth=self.config['mapping']['filter_depth'])
        # if self.config['mapping']['first_mesh']:
        #     self.save_mesh(0)

        print('First frame mapping done')
        return ret, loss

    def current_frame_mapping(self, batch, cur_frame_id):
        '''
        Current frame mapping
        Params:
            batch['c2w']: [1, 4, 4]
            batch['rgb']: [1, H, W, 3]
            batch['depth']: [1, H, W, 1]
            batch['direction']: [1, H, W, 3]
        Returns:
            ret: dict
            loss: float

        '''
        if self.config['mapping']['cur_frame_iters'] <= 0:
            return
        print('Current frame mapping...')

        c2w = self.est_c2w_data[cur_frame_id].to(self.device)

        self.model.train()

        # Training
        for i in range(self.config['mapping']['cur_frame_iters']):
            self.cur_map_optimizer.zero_grad()
            indice = self.select_samples(self.dataset.H, self.dataset.W, self.config['mapping']['sample'])

            indice_h, indice_w = indice % (self.dataset.H), indice // (self.dataset.H)
            rays_d_cam = batch['direction'].squeeze(0)[indice_h, indice_w, :].to(self.device)
            target_s = batch['rgb'].squeeze(0)[indice_h, indice_w, :].to(self.device)
            target_d = batch['depth'].squeeze(0)[indice_h, indice_w].to(self.device).unsqueeze(-1)

            rays_o = c2w[None, :3, -1].repeat(self.config['mapping']['sample'], 1)
            rays_d = torch.sum(rays_d_cam[..., None, :] * c2w[:3, :3], -1)
            if self.train:
                target_sem = batch['semantic'].squeeze(0)[indice_h, indice_w].to(self.device).unsqueeze(-1)
            else:

                target_sem = None
            # Forward
            ret,features = self.model.forward(batch, rays_o, rays_d, target_s, target_d, target_sem, self.enable_semantic)
            loss = self.get_loss_from_ret(ret)
            loss.backward()
            self.cur_map_optimizer.step()


        return ret, loss

    def smoothness(self, sample_points=256, voxel_size=0.1, margin=0.05, color=False):
        '''
        Smoothness loss of feature grid
        '''
        volume = self.bounding_box[:, 1] - self.bounding_box[:, 0]

        grid_size = (sample_points-1) * voxel_size
        offset_max = self.bounding_box[:, 1]-self.bounding_box[:, 0] - grid_size - 2 * margin

        offset = torch.rand(3).to(offset_max) * offset_max + margin
        coords = coordinates(sample_points - 1, 'cpu', flatten=False).float().to(volume)
        pts = (coords + torch.rand((1,1,1,3)).to(volume)) * voxel_size + self.bounding_box[:, 0] + offset

        if self.config['grid']['tcnn_encoding']:
            pts_tcnn = (pts - self.bounding_box[:, 0]) / (self.bounding_box[:, 1] - self.bounding_box[:, 0])


        sdf = self.model.query_sdf(pts_tcnn, embed=True)
        tv_x = torch.pow(sdf[1:,...]-sdf[:-1,...], 2).sum()
        tv_y = torch.pow(sdf[:,1:,...]-sdf[:,:-1,...], 2).sum()
        tv_z = torch.pow(sdf[:,:,1:,...]-sdf[:,:,:-1,...], 2).sum()

        loss = (tv_x + tv_y + tv_z)/ (sample_points**3)

        return loss

    def get_pose_param_optim(self, poses, mapping=True):
        task = 'mapping' if mapping else 'tracking'
        cur_trans = torch.nn.parameter.Parameter(poses[:, :3, 3])
        cur_rot = torch.nn.parameter.Parameter(self.matrix_to_tensor(poses[:, :3, :3]))
        pose_optimizer = torch.optim.Adam([{"params": cur_rot, "lr": self.config[task]['lr_rot']},
                                               {"params": cur_trans, "lr": self.config[task]['lr_trans']}])

        return cur_rot, cur_trans, pose_optimizer

    def global_BA(self, batch, cur_frame_id):
        '''
        Global bundle adjustment that includes all the keyframes and the current frame
        Params:
            batch['c2w']: ground truth camera pose [1, 4, 4]
            batch['rgb']: rgb image [1, H, W, 3]
            batch['depth']: depth image [1, H, W, 1]
            batch['direction']: view direction [1, H, W, 3]
            cur_frame_id: current frame id
        '''
        pose_optimizer = None

        # all the KF poses: 0, 5, 10, ...
        poses = torch.stack(
            [self.est_c2w_data[i] for i in range(0, cur_frame_id, self.config['mapping']['keyframe_every'])])

        # frame ids for all KFs, used for update poses after optimization
        frame_ids_all = torch.tensor(list(range(0, cur_frame_id, self.config['mapping']['keyframe_every'])))

        if len(self.keyframeDatabase.frame_ids) < 2:
            poses_fixed = torch.nn.parameter.Parameter(poses).to(self.device)
            current_pose = self.est_c2w_data[cur_frame_id][None, ...]
            poses_all = torch.cat([poses_fixed, current_pose], dim=0)

        else:
            poses_fixed = torch.nn.parameter.Parameter(poses[:1]).to(self.device)
            current_pose = self.est_c2w_data[cur_frame_id][None, ...]

            if self.config['mapping']['optim_cur']:
                cur_rot, cur_trans, pose_optimizer, = self.get_pose_param_optim(torch.cat([poses[1:], current_pose]))
                pose_optim = self.matrix_from_tensor(cur_rot, cur_trans).to(self.device)
                poses_all = torch.cat([poses_fixed, pose_optim], dim=0)

            else:
                cur_rot, cur_trans, pose_optimizer, = self.get_pose_param_optim(poses[1:])
                pose_optim = self.matrix_from_tensor(cur_rot, cur_trans).to(self.device)
                poses_all = torch.cat([poses_fixed, pose_optim, current_pose], dim=0)

        # Set up optimizer
        self.map_optimizer.zero_grad()
        if pose_optimizer is not None:
            pose_optimizer.zero_grad()

        # current_rays = torch.cat([batch['direction'], batch['rgb'], batch['depth'][..., None]], dim=-1)
        if self.train:
            current_rays = torch.cat( [batch['direction'], batch['rgb'], batch['depth'][..., None], batch['semantic'][..., None]], dim=-1)
            # current_rays = torch.cat(
            #     [batch['direction'], batch['rgb'], batch['depth'][..., None], batch['semantic']], dim=-1)
        else:
            H, W = batch['rgb'].shape[1], batch['rgb'].shape[2]
            print("direction is ",batch['direction'].shape)
            current_rays = torch.cat(
                [batch['direction'], batch['rgb'], batch['depth'][..., None]], dim=-1)

        current_rays = current_rays.reshape(-1, current_rays.shape[-1])



        for i in range(self.config['mapping']['iters']):

            # Sample rays with real frame ids
            # rays [bs, 7]
            # frame_ids [bs]
            rays, ids = self.keyframeDatabase.sample_global_rays(self.config['mapping']['sample'])

            # TODO: Checkpoint...
            idx_cur = random.sample(range(0, self.dataset.H * self.dataset.W),
                                    max(self.config['mapping']['sample'] // len(self.keyframeDatabase.frame_ids),
                                        self.config['mapping']['min_pixels_cur']))
            current_rays_batch = current_rays[idx_cur, :]

            rays = torch.cat([rays, current_rays_batch], dim=0)  # N, 7
            ids_all = torch.cat([ids // self.config['mapping']['keyframe_every'], -torch.ones((len(idx_cur)))]).to(
                torch.int64)

            rays_d_cam = rays[..., :3].to(self.device)
            target_s = rays[..., 3:6].to(self.device)
            target_d = rays[..., 6:7].to(self.device)
            target_sem = rays[..., -1].to(self.device)
            # [N, Bs, 1, 3] * [N, 1, 3, 3] = (N, Bs, 3)
            rays_d = torch.sum(rays_d_cam[..., None, None, :] * poses_all[ids_all, None, :3, :3], -1)
            rays_o = poses_all[ids_all, None, :3, -1].repeat(1, rays_d.shape[1], 1).reshape(-1, 3)
            rays_d = rays_d.reshape(-1, 3)

            ret,features = self.model.forward(batch, rays_o, rays_d, target_s, target_d, target_sem, self.enable_semantic)

            loss = self.get_loss_from_ret(ret, smooth=True)

            loss.backward(retain_graph=True)

            if (i + 1) % cfg["mapping"]["map_accum_step"] == 0:

                if (i + 1) > cfg["mapping"]["map_wait_step"]:
                    self.map_optimizer.step()
                else:
                    print('Wait update')
                self.map_optimizer.zero_grad()

            if pose_optimizer is not None and (i + 1) % cfg["mapping"]["pose_accum_step"] == 0:
                pose_optimizer.step()
                # get SE3 poses to do forward pass
                pose_optim = self.matrix_from_tensor(cur_rot, cur_trans)
                pose_optim = pose_optim.to(self.device)
                # So current pose is always unchanged
                if self.config['mapping']['optim_cur']:
                    poses_all = torch.cat([poses_fixed, pose_optim], dim=0)

                else:
                    current_pose = self.est_c2w_data[cur_frame_id][None, ...]
                    # SE3 poses

                    poses_all = torch.cat([poses_fixed, pose_optim, current_pose], dim=0)

                # zero_grad here
                pose_optimizer.zero_grad()

        print("BA ID is ",cur_frame_id)
        if pose_optimizer is not None and len(frame_ids_all) > 1:
            for i in range(len(frame_ids_all[1:])):
                self.est_c2w_data[int(frame_ids_all[i + 1].item())] = \
                self.matrix_from_tensor(cur_rot[i:i + 1], cur_trans[i:i + 1]).detach().clone()[0]

            if self.config['mapping']['optim_cur']:
                print('Update current pose')
                self.est_c2w_data[cur_frame_id] = \
                self.matrix_from_tensor(cur_rot[-1:], cur_trans[-1:]).detach().clone()[0]

    def predict_current_pose(self, frame_id, constant_speed=True):
        '''
        Predict current pose from previous pose using camera motion model
        '''
        if frame_id == 1 or (not constant_speed):
            c2w_est_prev = self.est_c2w_data[frame_id-1].to(self.device)
            self.est_c2w_data[frame_id] = c2w_est_prev

        else:
            c2w_est_prev_prev = self.est_c2w_data[frame_id-2].to(self.device)
            c2w_est_prev = self.est_c2w_data[frame_id-1].to(self.device)
            delta = c2w_est_prev.float() @ c2w_est_prev_prev.float().inverse()

            self.est_c2w_data[frame_id] = delta@c2w_est_prev

        return self.est_c2w_data[frame_id]

    def tracking_pc(self, batch, frame_id):
        '''
        Tracking camera pose of current frame using point cloud loss
        (Not used in the paper, but might be useful for some cases)
        '''

        c2w_gt = batch['c2w'][0].to(self.device)

        cur_c2w = self.predict_current_pose(frame_id, self.config['tracking']['const_speed'])

        cur_trans = torch.nn.parameter.Parameter(cur_c2w[..., :3, 3].unsqueeze(0))
        cur_rot = torch.nn.parameter.Parameter(self.matrix_to_tensor(cur_c2w[..., :3, :3]).unsqueeze(0))
        pose_optimizer = torch.optim.Adam([{"params": cur_rot, "lr": self.config['tracking']['lr_rot']},
                                               {"params": cur_trans, "lr": self.config['tracking']['lr_trans']}])
        best_sdf_loss = None

        iW = self.config['tracking']['ignore_edge_W']
        iH = self.config['tracking']['ignore_edge_H']

        thresh=0

        if self.config['tracking']['iter_point'] > 0:
            indice_pc = self.select_samples(self.dataset.H-iH*2, self.dataset.W-iW*2, self.config['tracking']['pc_samples'])
            rays_d_cam = batch['direction'][:, iH:-iH, iW:-iW].reshape(-1, 3)[indice_pc].to(self.device)
            target_s = batch['rgb'][:, iH:-iH, iW:-iW].reshape(-1, 3)[indice_pc].to(self.device)
            target_d = batch['depth'][:, iH:-iH, iW:-iW].reshape(-1, 1)[indice_pc].to(self.device)

            valid_depth_mask = ((target_d > 0.) * (target_d < 5.))[:,0]

            rays_d_cam = rays_d_cam[valid_depth_mask]
            target_s = target_s[valid_depth_mask]
            target_d = target_d[valid_depth_mask]

            for i in range(self.config['tracking']['iter_point']):
                pose_optimizer.zero_grad()
                c2w_est = self.matrix_from_tensor(cur_rot, cur_trans)


                rays_o = c2w_est[...,:3, -1].repeat(len(rays_d_cam), 1)
                rays_d = torch.sum(rays_d_cam[..., None, :] * c2w_est[:, :3, :3], -1)
                pts = rays_o + target_d * rays_d

                pts_flat = (pts - self.bounding_box[:, 0]) / (self.bounding_box[:, 1] - self.bounding_box[:, 0])

                out = self.model.query_color_sdf(pts_flat)

                sdf = out[:, -1]
                rgb = torch.sigmoid(out[:,:3])

                # TODO: Change this
                loss = 5 * torch.mean(torch.square(rgb-target_s)) + 1000 * torch.mean(torch.square(sdf))

                if best_sdf_loss is None:
                    best_sdf_loss = loss.cpu().item()
                    best_c2w_est = c2w_est.detach()

                with torch.no_grad():
                    c2w_est = self.matrix_from_tensor(cur_rot, cur_trans)

                    if loss.cpu().item() < best_sdf_loss:
                        best_sdf_loss = loss.cpu().item()
                        best_c2w_est = c2w_est.detach()
                        thresh = 0
                    else:
                        thresh +=1
                if thresh >self.config['tracking']['wait_iters']:
                    break

                loss.backward()
                pose_optimizer.step()


        if self.config['tracking']['best']:
            self.est_c2w_data[frame_id] = best_c2w_est.detach().clone()[0]
        else:
            self.est_c2w_data[frame_id] = c2w_est.detach().clone()[0]


        if frame_id % self.config['mapping']['keyframe_every'] != 0:
            # Not a keyframe, need relative pose
            kf_id = frame_id // self.config['mapping']['keyframe_every']
            kf_frame_id = kf_id * self.config['mapping']['keyframe_every']
            c2w_key = self.est_c2w_data[kf_frame_id]
            delta = self.est_c2w_data[frame_id] @ c2w_key.float().inverse()
            self.est_c2w_data_rel[frame_id] = delta
        print('Best loss: {}, Camera loss{}'.format(F.l1_loss(best_c2w_est.to(self.device)[0,:3], c2w_gt[:3]).cpu().item(), F.l1_loss(c2w_est[0,:3], c2w_gt[:3]).cpu().item()))

    def poses_rel(self,i,j):



        c2w_prev_keyframe =self.est_c2w_data[i].to(self.device)
        c2w_current_keyframe = self.est_c2w_data[j].to(self.device)



        relative_transformation = c2w_current_keyframe @ c2w_prev_keyframe.float().inverse()
        # relative_transformation = c2w_prev_keyframe.float().inverse() @ c2w_current_keyframe


        return relative_transformation


    def tracking_render(self, batch, frame_id):
        '''
        Tracking camera pose using of the current frame
        Params:
            batch['c2w']: Ground truth camera pose [B, 4, 4]
            batch['rgb']: RGB image [B, H, W, 3]
            batch['depth']: Depth image [B, H, W, 1]
            batch['direction']: Ray direction [B, H, W, 3]
            frame_id: Current frame id (int)
        '''

        c2w_gt = batch['c2w'][0].to(self.device)

        # Initialize current pose
        if self.config['tracking']['iter_point'] > 0:
            cur_c2w = self.est_c2w_data[frame_id]
        else:
            cur_c2w = self.predict_current_pose(frame_id, self.config['tracking']['const_speed'])

        indice = None
        best_sdf_loss = None
        thresh=0

        iW = self.config['tracking']['ignore_edge_W']
        iH = self.config['tracking']['ignore_edge_H']

        cur_rot, cur_trans, pose_optimizer = self.get_pose_param_optim(cur_c2w[None,...], mapping=False)

        # Start tracking
        for i in range(self.config['tracking']['iter']):
            pose_optimizer.zero_grad()
            c2w_est = self.matrix_from_tensor(cur_rot, cur_trans)

            # Note here we fix the sampled points for optimisation
            if indice is None:
                indice = self.select_samples(self.dataset.H-iH*2, self.dataset.W-iW*2, self.config['tracking']['sample'])

                # Slicing
                indice_h, indice_w = indice % (self.dataset.H - iH * 2), indice // (self.dataset.H - iH * 2)
                rays_d_cam = batch['direction'].squeeze(0)[iH:-iH, iW:-iW, :][indice_h, indice_w, :].to(self.device)
            target_s = batch['rgb'].squeeze(0)[iH:-iH, iW:-iW, :][indice_h, indice_w, :].to(self.device)
            target_d = batch['depth'].squeeze(0)[iH:-iH, iW:-iW][indice_h, indice_w].to(self.device).unsqueeze(-1)


            rays_o = c2w_est[...,:3, -1].repeat(self.config['tracking']['sample'], 1)
            rays_d = torch.sum(rays_d_cam[..., None, :] * c2w_est[:, :3, :3], -1)
            if self.train:
                target_sem = batch['semantic'].squeeze(0)[iH:-iH, iW:-iW][indice_h, indice_w].to(self.device).unsqueeze(-1)
            else:

                    target_sem = None

            # ret = self.model.forward(rays_o, rays_d, target_s, target_d)
            ret,features = self.model.forward(batch, rays_o, rays_d, target_s, target_d, target_sem)
            loss = self.get_loss_from_ret(ret)

            if best_sdf_loss is None:
                best_sdf_loss = loss.cpu().item()
                best_c2w_est = c2w_est.detach()

            with torch.no_grad():
                c2w_est = self.matrix_from_tensor(cur_rot, cur_trans)

                if loss.cpu().item() < best_sdf_loss:
                    best_sdf_loss = loss.cpu().item()
                    best_c2w_est = c2w_est.detach()
                    thresh = 0
                else:
                    thresh +=1

            if thresh >self.config['tracking']['wait_iters']:
                break

            loss.backward()
            pose_optimizer.step()

        if self.config['tracking']['best']:
            # Use the pose with smallest loss
            self.est_c2w_data[frame_id] = best_c2w_est.detach().clone()[0]
        else:
            # Use the pose after the last iteration
            self.est_c2w_data[frame_id] = c2w_est.detach().clone()[0]

       # Save relative pose of non-keyframes
        if frame_id % self.config['mapping']['keyframe_every'] != 0:
            kf_id = frame_id // self.config['mapping']['keyframe_every']
            kf_frame_id = kf_id * self.config['mapping']['keyframe_every']
            c2w_key = self.est_c2w_data[kf_frame_id]
            delta = self.est_c2w_data[frame_id] @ c2w_key.float().inverse()
            self.est_c2w_data_rel[frame_id] = delta

        print('Best loss: {}, Last loss{}'.format(F.l1_loss(best_c2w_est.to(self.device)[0,:3], c2w_gt[:3]).cpu().item(), F.l1_loss(c2w_est[0,:3], c2w_gt[:3]).cpu().item()))

        loss = F.l1_loss(best_c2w_est.to(self.device)[0, :3], c2w_gt[:3]).cpu().item()


    def convert_relative_pose(self):
        poses = {}
        for i in range(len(self.est_c2w_data)):
            if i % self.config['mapping']['keyframe_every'] == 0:
                poses[i] = self.est_c2w_data[i]
            else:
                kf_id = i // self.config['mapping']['keyframe_every']
                kf_frame_id = kf_id * self.config['mapping']['keyframe_every']
                c2w_key = self.est_c2w_data[kf_frame_id]
                delta = self.est_c2w_data_rel[i]
                poses[i] =delta @ c2w_key

        return poses

    def convert_relative_loop_pose(self,pre,current):

        for i in range(pre,current):
            if i in self.con_frameid :
                pass
            else :
                if i % self.config['mapping']['keyframe_every'] == 0:

                    con_frame_id = self.rel_id[i]

                    c2w_key = self.est_c2w_data[con_frame_id]
                    delta = self.est_c2w_data_rel[i]  # 当前@索引
                    self.est_c2w_data[i] = delta.to(self.device).float() @ c2w_key.to(self.device).float()

                else:
                    kf_id = i // self.config['mapping']['keyframe_every']
                    kf_frame_id = kf_id * self.config['mapping']['keyframe_every']
                    c2w_key = self.est_c2w_data[kf_frame_id]
                    delta = self.est_c2w_data_rel[i]
                    self.est_c2w_data[i] = delta.to(self.device).float() @ c2w_key.to(self.device).float()

    def handel_path(self, index):

        color_path = self.dataset.img_files[index]
        depth_path = self.dataset.depth_paths[index]
        color_data = cv2.imread(color_path)
        if '.png' in depth_path:
            depth_data = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        elif '.exr' in depth_path:
            raise NotImplementedError()

        color_data = cv2.cvtColor(color_data, cv2.COLOR_BGR2RGB)
        color_data = color_data / 255.

        depth_data = depth_data.astype(np.float32) / self.dataset.png_depth_scale * self.dataset.sc_factor

        H, W = depth_data.shape
        color_data = cv2.resize(color_data, (W, H))
        if self.dataset.downsample_factor > 1:  # 将图像或数据在空间上减小尺寸
            H = H // self.dataset.downsample_factor
            W = W // self.dataset.downsample_factor
            color_data = cv2.resize(color_data, (W, H), interpolation=cv2.INTER_AREA)
            depth_data = cv2.resize(depth_data, (W, H), interpolation=cv2.INTER_NEAREST)
        edge = self.config['cam']['crop_edge']
        if edge > 0:
            color_data = color_data[edge:-edge, edge:-edge]
            depth_data = depth_data[edge:-edge, edge:-edge]

        color_data = torch.from_numpy(color_data.astype(np.float32))
        depth_data = torch.from_numpy(depth_data.astype(np.float32))
        return color_data, depth_data

    def create_optimizer(self):
        '''
        Create optimizer for mapping
        '''
        # Optimizer for BA
        trainable_parameters = [{'params': self.model.decoder.sdf_net.parameters(), 'weight_decay': 1e-6, 'lr': self.config['mapping']['lr_decoder']},
                                {'params': self.model.decoder.color_net.parameters(), 'weight_decay': 1e-6, 'lr': self.config['mapping']['lr_decoder']},
                                {'params': self.model.embed_fn.parameters(), 'eps': 1e-15, 'lr': self.config['mapping']['lr_embed']}]

        if not self.config['grid']['oneGrid']:
            trainable_parameters.append({'params': self.model.embed_fn_color.parameters(), 'eps': 1e-15, 'lr': self.config['mapping']['lr_embed_color']})

        self.map_optimizer = optim.Adam(trainable_parameters, betas=(0.9, 0.99))


        sem_params = [{'params': self.model.decoder.semantic_net.parameters(), 'eps': 1e-15, 'weight_decay': 1e-6, 'lr': self.config['mapping']['lr_sem']}]
        self.sem_optimizer = optim.Adam(sem_params, betas=(0.9, 0.99))


        # Optimizer for current frame mapping
        if self.config['mapping']['cur_frame_iters'] > 0:
            params_cur_mapping = [{'params': self.model.embed_fn.parameters(), 'eps': 1e-15, 'lr': self.config['mapping']['lr_embed']}]
            if not self.config['grid']['oneGrid']:
                params_cur_mapping.append({'params': self.model.embed_fn_color.parameters(), 'eps': 1e-15, 'lr': self.config['mapping']['lr_embed_color']})

            self.cur_map_optimizer = optim.Adam(params_cur_mapping, betas=(0.9, 0.99))

    def save_mesh(self, i, voxel_size=0.05):
        mesh_savepath = os.path.join(self.config['data']['output'], self.config['data']['exp_name'], f'{i:04d}_mesh.ply')


        if self.config['mesh']['render_color']:
            color_func = self.model.render_surface_color
        else:
            color_func = self.model.query_color

        extract_mesh(self.model.query_sdf,

                        self.config,
                        self.bounding_box,
                        color_func=color_func,
                        marching_cube_bound=self.marching_cube_bound,
                        voxel_size=voxel_size,
                        mesh_savepath=mesh_savepath)

    def se3_to_matrix(self, se3_tensor_list):
        transformation_matrices = []
        for data in se3_tensor_list:
            translation_vector = data[:3].cpu().detach().numpy()
            quaternion = data[3:].cpu().detach().numpy()
            rotation_matrix = R.from_quat(quaternion).as_matrix()
            transformation_matrix = np.eye(4)
            transformation_matrix[:3, :3] = rotation_matrix
            transformation_matrix[:3, 3] = translation_vector
            transformation_matrices.append(torch.from_numpy(transformation_matrix).to(self.device))

        transformation_matrices_list = [tensor.cpu().detach().numpy() for tensor in transformation_matrices]

        return transformation_matrices_list

    def sem(self, batch, cur_c2w):
        iW = self.config['tracking']['ignore_edge_W']
        iH = self.config['tracking']['ignore_edge_H']
        cur_c2w = cur_c2w.unsqueeze(0)
        self.model.eval()
        H = self.dataset.H - iH * 2
        W = self.dataset.W - iW * 2


        indice = self.select_samples(self.dataset.H - iH * 2, self.dataset.W - iW * 2, self.num_samples)
        indice_h, indice_w = indice % (self.dataset.H - iH * 2), indice // (self.dataset.H - iH * 2)

        rays_d_cam = batch['direction'].squeeze(0)[iH:-iH, iW:-iW, :][indice_h, indice_w, :].to(self.device)
        rays_d_world = torch.sum(rays_d_cam[..., None, :] * cur_c2w[:, :3, :3], -1)
        target_s = batch['rgb'].squeeze(0)[iH:-iH, iW:-iW, :][indice_h, indice_w, :].to(self.device)
        target_d = batch['depth'].squeeze(0)[iH:-iH, iW:-iW][indice_h, indice_w].to(self.device).unsqueeze(-1)
        rays_o = cur_c2w[..., :3, -1].repeat(self.num_samples, 1)
        rays_d = torch.sum(rays_d_cam[..., None, :] * cur_c2w[:, :3, :3], -1)
        gt_depth = target_d.reshape(-1)
        device = self.device
        sem_logits = self.render_batch_ray(rays_d, rays_o, device, truncation=0.08, gt_depth=gt_depth)
        return sem_logits, indice

    def train_sem(self, batch, cur_c2w, iters=20):

        # Initialize current pose

        iW = self.config['tracking']['ignore_edge_W']
        iH = self.config['tracking']['ignore_edge_H']

        num_rays = 1024

        cur_c2w = cur_c2w.unsqueeze(0)
        cur_rot, cur_trans, pose_optimizer = self.get_pose_param_optim(cur_c2w, mapping=False)

        self.model.train()

        for i in range(iters):
            self.sem_optimizer.zero_grad()
            c2w_est = self.matrix_from_tensor(cur_rot, cur_trans).to(self.device)
            # Note here we fix the sampled points for optimisation

            indice = random.sample(range((self.dataset.H - iH * 2) * (self.dataset.W - iW * 2)), num_rays)
            indice = torch.tensor(indice)

            indice_h, indice_w = indice % (self.dataset.H - iH * 2), indice // (self.dataset.H - iH * 2)
            rays_d_cam = batch['direction'].squeeze(0)[iH:-iH, iW:-iW, :][indice_h, indice_w, :].to(self.device)
            rays_d_world = torch.sum(rays_d_cam[..., None, :] * cur_c2w[:, :3, :3], -1)


            target_s = batch['rgb'].squeeze(0)[iH:-iH, iW:-iW, :][indice_h, indice_w, :].to(self.device)
            target_d = batch['depth'].squeeze(0)[iH:-iH, iW:-iW][indice_h, indice_w].to(self.device).unsqueeze(-1)
            rays_o = c2w_est[..., :3, -1].repeat(num_rays, 1)
            rays_d = torch.sum(rays_d_cam[..., None, :] * c2w_est[:, :3, :3], -1)

            target_sem = batch['semantic'].squeeze(0)[indice_h, indice_w].to(self.device).unsqueeze(-1)

            ret, features = self.model.forward(batch, rays_o, rays_d, target_s, target_d, target_sem,
                                               enable_semantic=True)
            if i == iters-1:
                print("mean_IU is",ret["mean_IU"])
                print("mean_IU_no_back is", ret["mean_IU_no_back"])
                print("mean_pixel_acc is", ret["mean_pixel_acc"])
            loss = self.get_loss_from_ret(ret, enable_semantic=True, smooth=True)



            est_c2w_list = list(self.est_c2w_data.values())
            for pose in est_c2w_list:
                if not isinstance(pose, torch.Tensor) or pose.shape != (4, 4):
                    print("Invalid pose shape:", pose.shape)

            loss.backward(retain_graph=True)
            self.sem_optimizer.step()
            self.sem_optimizer.zero_grad()


        return

    def render_batch_ray(self, rays_d, rays_o, device, truncation, gt_depth=None):
        n_stratified = 32
        n_importance = 8
        n_rays = rays_o.shape[0]
        z_vals = torch.empty([n_rays, n_stratified + n_importance], device=device)  # [10000, 40]
        near = 0.0
        t_vals_uni = torch.linspace(0., 1., steps=n_stratified, device=device)
        t_vals_surface = torch.linspace(0., 1., steps=n_importance, device=device)
        gt_depth = gt_depth.reshape(-1, 1)
        gt_mask = (gt_depth > 0).squeeze()
        gt_nonezero = gt_depth[gt_mask]
        gt_depth_surface = gt_nonezero.expand(-1, n_importance)
        z_vals_surface = gt_depth_surface - (1.5 * truncation) + (3 * truncation * t_vals_surface)

        gt_depth_free = gt_nonezero.expand(-1, n_stratified)
        z_vals_free = near + 1.2 * gt_depth_free * t_vals_uni

        z_vals_nonzero, _ = torch.sort(torch.cat([z_vals_free, z_vals_surface], dim=-1), dim=-1)

        z_vals_nonzero = self.perturbation(z_vals_nonzero)
        z_vals[gt_mask] = z_vals_nonzero

        if not gt_mask.all():
            with torch.no_grad():
                rays_o_uni = rays_o[~gt_mask].detach()
                rays_d_uni = rays_d[~gt_mask].detach()
                det_rays_o = rays_o_uni.unsqueeze(-1)  # (N, 3, 1)
                det_rays_d = rays_d_uni.unsqueeze(-1)  # (N, 3, 1)
                t = (self.bounding_box.unsqueeze(0) - det_rays_o) / det_rays_d  # (N, 3, 2)
                far_bb, _ = torch.min(torch.max(t, dim=2)[0], dim=1)
                far_bb = far_bb.unsqueeze(-1)
                far_bb += 0.01

                z_vals_uni = near * (1. - t_vals_uni) + far_bb * t_vals_uni
                z_vals_uni = self.perturbation(z_vals_uni).float()  # 转换为 Float
                pts_uni = rays_o_uni.unsqueeze(1) + rays_d_uni.unsqueeze(1) * z_vals_uni.unsqueeze(
                    -1)  # [n_rays, n_stratified, 3]
                pts_uni_nor = normalize_3d_coordinate(pts_uni.clone(), self.bounding_box)
                sdf_uni = self.model.query_sdf(pts_uni_nor)
                sdf_uni = sdf_uni.reshape(*pts_uni.shape[0:2])
                alpha_uni = self.sdf2alpha(sdf_uni)
                weights_uni = alpha_uni * torch.cumprod(
                    torch.cat([torch.ones((alpha_uni.shape[0], 1), device=device), (1. - alpha_uni + 1e-10)], -1), -1)[
                                          :, :-1]

                z_vals_uni_mid = .5 * (z_vals_uni[..., 1:] + z_vals_uni[..., :-1])
                z_samples_uni = sample_pdf(z_vals_uni_mid, weights_uni[..., 1:-1], n_importance, det=False,
                                           device=device)
                z_vals_uni, ind = torch.sort(torch.cat([z_vals_uni, z_samples_uni], -1), -1)
                z_vals[~gt_mask] = z_vals_uni

        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :,
                                                            None]  # [n_rays, n_stratified+n_importance, 3]

        raw = self.model.run_network(pts, enable_semantic=True)
        semantic_alpha = self.sdf2alpha(raw[..., 3])
        semantic_weights = semantic_alpha * torch.cumprod(
            torch.cat([torch.ones((semantic_alpha.shape[0], 1), device=device), (1. - semantic_alpha + 1e-10)], -1),
            -1)[:, :-1]
        rendered_semantic = torch.sum(semantic_weights[..., None] * raw[..., 4:], -2)
        rendered_semantic = torch.argmax(rendered_semantic, dim=-1).cpu()

        return rendered_semantic

    def sdf2alpha(self, sdf, beta=10):
        return 1. - torch.exp(-beta * torch.sigmoid(-sdf * beta))

    def perturbation(self, z_vals):
        mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]], -1)
        lower = torch.cat([z_vals[..., :1], mids], -1)
        t_rand = torch.rand(z_vals.shape, device=z_vals.device)
        return lower + (upper - lower) * t_rand

    def calculate_overlapping_percentage(self,set1, set2):
        set1 = set(set1.cpu().numpy().flatten())
        set2 = set(set2.cpu().numpy().flatten())
        intersection = set1.intersection(set2)
        union = set1.union(set2)
        overlapping_percentage = (len(intersection) / len(union))  if set1 else 0
        print("Overlapping percentage is ",overlapping_percentage)

        return overlapping_percentage
    def con_cloud_data(self,batch):
        current_id = batch['frame_id'].item()
        if current_id == 0:
            rdgdata,depthdata = self.handel_path(current_id)
            self.con_data[current_id] = {'rgb': rdgdata, 'depth': depthdata}
        else:
            if current_id % self.config['mapping']['keyframe_every'] == 0:
                former_id = current_id - 1
                rdgdatacur, depthdatacur = self.handel_path(current_id)
                rdgdata, depthdata = self.handel_path(former_id)
                self.con_data[former_id] = {'rgb': rdgdata, 'depth':depthdata}
                self.con_data[current_id] = {'rgb': rdgdatacur, 'depth': depthdatacur}
            else:
                rdgdata, depthdata = self.handel_path(current_id)
                self.con_data[current_id] = {'rgb': rdgdata, 'depth': depthdata}
    def getfixsem(self,batch,c2w,index):
        iW = self.config['tracking']['ignore_edge_W']
        iH = self.config['tracking']['ignore_edge_H']
        cur_c2w = c2w.unsqueeze(0)
        self.model.eval()


        indice = index
        indice_h, indice_w = indice % (self.dataset.H - iH * 2), indice // (self.dataset.H - iH * 2)

        rays_d_cam = batch['direction'].squeeze(0)[iH:-iH, iW:-iW, :][indice_h, indice_w, :].to(self.device)

        target_d = batch['depth'].squeeze(0)[iH:-iH, iW:-iW][indice_h, indice_w].to(self.device).unsqueeze(-1)
        rays_o = cur_c2w[..., :3, -1].repeat(self.num_samples, 1)
        rays_d = torch.sum(rays_d_cam[..., None, :] * cur_c2w[:, :3, :3], -1)
        gt_depth = target_d.reshape(-1)
        device = self.device
        sem_logits = self.render_batch_ray(rays_d, rays_o, device, truncation=0.08, gt_depth=gt_depth)
        self.model.train()
        return sem_logits
    def get_sem(self,batch,c2w):

        if self.train:
            sem_logits, indice = self.sem(batch, c2w)
            self.model.train()

            return sem_logits, indice
        else:
            indice = self.select_samples(self.dataset.H, self.dataset.W, self.config['mapping']['sample'])

            indice_h, indice_w = indice % (self.dataset.H), indice // (self.dataset.H)
            rays_d_cam = batch['direction'].squeeze(0)[indice_h, indice_w, :].to(self.device)
            target_d = batch['depth'].squeeze(0)[indice_h, indice_w].to(self.device).unsqueeze(-1)

            rays_o = c2w[None, :3, -1].repeat(self.config['mapping']['sample'], 1)
            rays_d = torch.sum(rays_d_cam[..., None, :] * c2w[:3, :3], -1)
            n_rays = rays_o.shape[0]
            z_samples = torch.linspace(-self.config['training']['range_d'], self.config['training']['range_d'],
                                       steps=self.config['training']['n_range_d']).to(target_d)
            z_samples = z_samples[None, :].repeat(n_rays, 1) + target_d
            z_samples[target_d.squeeze() <= 0] = torch.linspace(self.config['cam']['near'], self.config['cam']['far'],
                                                                steps=self.config['training']['n_range_d']).to(target_d)
            pts = rays_o[..., None, :] + rays_d[..., None, :] * z_samples[..., :, None]  # [N_rays, N_samples, 3]
            generated_semantic_output = self.model.query_sem_feature(pts)
            return generated_semantic_output, indice

    def handle_semantic(self,batch,est_c2w_data,iters = 40):
        id = batch["frame_id"].item()
        self.train_sem(batch, est_c2w_data[id], iters)
        sem_logits, indice = self.get_sem(batch, est_c2w_data[id])
        self.sem_logits.append(sem_logits)
        self.indice.append(indice)
        self.sem_id.append(id)

    def diff_level_frame(self,batch):
        id = batch["frame_id"].item()
        isplace = self.placeDatabase.add_placeframe(batch)
        if isplace or id == 0:
            self.placeDatabase.add_place_data(batch, self.sem_logits[-1], self.indice[-1], self.est_c2w_data[0])
            self.placeDatabase.update_placeframes_features(self.model.query_local_features, en=True)
        ifcoview = self.placeDatabase.covisible_frame(batch, self.sem_logits[-1], self.indice[-1])
        if ifcoview or id == 0:
            self.con_cloud_data(batch)
        return ifcoview

    def loop_map(self,batch, ifcoview, ind_ids, i, edge, score_max=0.6):
        if ind_ids is not None or ind_ids != -1:
            can_ind = []
            cansem = []
            # filter with semantic
            for ind_id in range(len(ind_ids)):
                ind = ind_ids[ind_id]
                c2r = torch.matmul(torch.linalg.inv(self.est_c2w_data[ind].to(self.device)),
                                   self.est_c2w_data[i].to(self.device))
                score2 = self.placeDatabase.check_projection(batch['depth'][0, ...], batch['direction'][0, ...],
                                                             batch['intrinsic'][0, ...], c2r).item()


                if score2 > score_max and (i - ind) > 30 and score2 < 1:
                    if self.train:
                        # sem_logits_current, indice = self.get_sem(batch, self.est_c2w_data[i])
                        index = self.indice[self.sem_id.index(ind)]
                        sem_logits_current = self.getfixsem(batch, self.est_c2w_data[i],index)



                        sem_score = self.calculate_overlapping_percentage(self.sem_logits[self.sem_id.index(ind)],
                                                                          sem_logits_current)

                        if sem_score > 0.6:
                            cansem.append(sem_score)
                            can_ind.append(ind)
            if can_ind:
                ind_max = min(can_ind, key=lambda x: abs(x - i))
                self.con_frameid.append(i)

                self.est_c2w_data = self.convert_relative_pose()
                selected_edges = []
                index = self.con_frameid.index(ind_max)


                if ind_max == 0:

                    for k1 in range(ind_max, ind_max + 2):
                        if k1 < ind_max + 1:
                            k2 = k1 + 1
                            edge = self.poses_rel(k1, k2)
                            edge = edge.to(self.device)
                            line_data = [k1, k2] + edge.tolist()  # 检查矩阵变换，把edge直接存入
                            selected_edges.append(line_data)
                        else:
                            k2 = self.con_frameid[index + 1]
                            edge = self.poses_rel(k1, k2)
                            edge = edge.to(self.device)
                            line_data = [k1, k2] + edge.tolist()  # 检查矩阵变换，把edge直接存入
                            selected_edges.append(line_data)
                else:
                    for k1 in range(ind_max - 1, ind_max + 2):
                        if k1 < ind_max + 1:
                            k2 = k1 + 1
                            edge = self.poses_rel(k1, k2)
                            edge = edge.to(self.device)
                            line_data = [k1, k2] + edge.tolist()  # 检查矩阵变换，把edge直接存入
                            selected_edges.append(line_data)
                        else:
                            k2 = self.con_frameid[index + 1]
                            edge = self.poses_rel(k1, k2)
                            edge = edge.to(self.device)
                            line_data = [k1, k2] + edge.tolist()  # 检查矩阵变换，把edge直接存入
                            selected_edges.append(line_data)

                # the adge of optimation map/common
                for k in range(index, len(self.con_frameid) - 1):
                    k1 = self.con_frameid[k]
                    k2 = self.con_frameid[k + 1]
                    edge = self.poses_rel(k1, k2)
                    edge = edge.to(self.device)
                    line_data = [k1, k2] + edge.tolist()
                    selected_edges.append(line_data)

                frameid = []
                seen = set()
                for row in selected_edges:
                    first_col = row[0]
                    if not isinstance(first_col, int):
                        first_col = int(first_col)
                    if first_col not in seen:
                        frameid.append(first_col)
                        seen.add(first_col)
                frameid.append(i)
                frameid.sort()

                edge_loop = []
                target_point = generate_point_cloud(batch['rgb'], batch['depth'], self.est_c2w_data[i], self.dataset.fx,
                                                    self.dataset.fy, self.dataset.cx, self.dataset.cy)

                # the adge of optimation map/loop
                if ind_max == 0:
                    for source_id in range(ind_max, ind_max + 2):

                        source_pc = generate_point_cloud(self.con_data[source_id]['rgb'],
                                                         self.con_data[source_id]['depth'],
                                                         self.est_c2w_data[source_id],
                                                         self.dataset.fx, self.dataset.fy, self.dataset.cx,
                                                         self.dataset.cy)
                        init_est = self.poses_rel(source_id, i)
                        edge, result_icp = icp(source_pc, target_point, init_est)
                        print("result_icp.inlier_rmse is", result_icp.inlier_rmse)
                        if result_icp.inlier_rmse > 0.0028:
                            edge = None
                        if edge is not None:
                            edge = edge.to(self.device)
                            line_data = [source_id, i] + edge.tolist()  # 检查矩阵变换，把edge直接存入
                            edge_loop.append(line_data)
                else:
                    for source_id in range(ind_max - 1, ind_max + 2):

                        source_pc = generate_point_cloud(self.con_data[source_id]['rgb'],
                                                         self.con_data[source_id]['depth'],
                                                         self.est_c2w_data[source_id],
                                                         self.dataset.fx, self.dataset.fy, self.dataset.cx,
                                                         self.dataset.cy)
                        init_est = self.poses_rel(source_id, i)
                        edge, result_icp = icp(source_pc, target_point, init_est)
                        print("result_icp.inlier_rmse is", result_icp.inlier_rmse)
                        if result_icp.inlier_rmse > 0.0028:
                            edge = None
                        if edge is not None:
                            edge = edge.to(self.device)
                            line_data = [source_id, i] + edge.tolist()  # 检查矩阵变换，把edge直接存入
                            edge_loop.append(line_data)
                if edge is not None:
                    self.opt_loop_map(selected_edges,frameid ,ind_max,edge_loop,i)

            elif ifcoview:
                self.con_frameid.append(i)
            else:
                last_element = self.con_frameid[-1]
                self.rel_id[i] = last_element

    def opt_loop_map(self, selected_edges,frameid,ind_max, edge_loop,i):
        edges = selected_edges + edge_loop
        for j in range(ind_max, i - 1):
            if j in self.con_frameid:
                pass
            else:
                if j % self.config['mapping']['keyframe_every'] == 0:
                    last_element = self.rel_id[j]
                    c2w_key = self.est_c2w_data[last_element]
                    delta = self.est_c2w_data[j] @ c2w_key.float().inverse()
                    self.est_c2w_data_rel[j] = delta
        end = pypose(config, edges, frameid, self.est_c2w_data, self.pose_gt,
                     self.device, self.config['data']['exp_name'], self.config['data']['output'])

        end_poses = self.se3_to_matrix(end)
        end_poses = [torch.tensor(pose).float().to(self.device) for pose in end_poses]
        k = 0
        for num in frameid:
            self.est_c2w_data[num] = end_poses[k]
            k = k + 1
        self.convert_relative_loop_pose(ind_max, i)


    def run(self):
        self.create_optimizer()
        data_loader = DataLoader(self.dataset, num_workers=self.config['data']['num_workers'])



        for i, batch in tqdm(enumerate(data_loader)):
            # Visualisation
            if self.config['mesh']['visualisation']:
                rgb = cv2.cvtColor(batch["rgb"].squeeze().cpu().numpy(), cv2.COLOR_BGR2RGB)
                raw_depth = batch["depth"]
                mask = (raw_depth >= self.config["cam"]["depth_trunc"]).squeeze(0)
                depth_colormap = colormap_image(batch["depth"])
                depth_colormap[:, mask] = 255.
                depth_colormap = depth_colormap.permute(1, 2, 0).cpu().numpy()
                image = np.hstack((rgb, depth_colormap))
                cv2.namedWindow('RGB-D'.format(i), cv2.WINDOW_AUTOSIZE)
                cv2.imshow('RGB-D'.format(i), image)
                key = cv2.waitKey(1)

            # First frame mapping
            if i == 0:

                self.first_frame_mapping(batch, self.config['mapping']['first_iters'])
                self.handle_semantic(batch,self.est_c2w_data)
                self.diff_level_frame(batch)
                self.con_frameid.append(i)


            # Tracking + Mapping
            else:
                if self.config['tracking']['iter_point'] > 0:
                    self.tracking_pc(batch, i)
                self.tracking_render(batch, i)


                if i % self.config['mapping']['keyframe_every'] == 0:

                    self.keyframeDatabase.add_keyframe(batch,self.train, filter_depth=self.config['mapping']['filter_depth'])

                    self.handle_semantic(batch, self.est_c2w_data)


                    ifcoview =  self.diff_level_frame(batch)

                    ind_ids = []
                    edge = None
                    placeframe_id, loop_score = self.placeDatabase.search_loops(self.model.query_local_features, i,
                                                                                batch, self.config['mapping'][ 'keyframe_every'])

                    if placeframe_id is not None:
                        if 30 < i - placeframe_id[0] <= 50:
                            self.t = loop_score[0] * 0.2 + self.t * 0.8

                        for ix in range(0, len(placeframe_id)):
                            if loop_score[ix] < self.t:  # 0.15
                                ind_ids.append(placeframe_id[ix])


                    if ifcoview or ind_ids is not None:

                        if ind_ids is not None or ind_ids != -1:
                            self.loop_map(batch, ifcoview, ind_ids, i, edge)

                        else:
                            self.con_frameid.append(i)


                if i-1 in self.con_frameid:
                    self.con_cloud_data(batch)

                if i % self.config['mapping']['keyframe_every'] == 0:
                    self.current_frame_mapping(batch, i)
                    self.global_BA(batch, i)


                if i % self.config['mesh']['vis'] ==0:
                    self.est_c2w_data = self.convert_relative_pose()
                    pose_evaluation(self.con_frameid, self.pose_gt, self.est_c2w_data, 1,
                                    os.path.join(self.config['data']['output'], self.config['data']['exp_name']), i,
                                    point=False, img='mesh', name='output_after_rel_relative.txt')

        model_savepath = os.path.join(self.config['data']['output'], self.config['data']['exp_name'],
                                      'checkpoint{}.pt'.format(i))
        #mesh图
        self.save_ckpt(model_savepath)
        self.save_mesh(i, voxel_size=self.config['mesh']['voxel_final'])


        pose_relative = self.convert_relative_pose()
        pose_evaluation(self.con_frameid,self.pose_gt, self.est_c2w_data, 1, os.path.join(self.config['data']['output'], self.config['data']['exp_name']), i,point= False,img='pose_after', name='output_after_relative.txt')
        pose_evaluation(self.con_frameid,self.pose_gt, pose_relative, 1, os.path.join(self.config['data']['output'], self.config['data']['exp_name']), i, point= False,img='pose_relative_after', name='output_after_relative.txt')


        #TODO: Evaluation of reconstruction


if __name__ == '__main__':
            
    print('Start running...')
    parser = argparse.ArgumentParser(
        description='Arguments for running the NICE-SLAM/iMAP*.'
    )
    parser.add_argument('--config', type=str, help='Path to config file.')
    parser.add_argument('--input_folder', type=str,
                        help='input folder, this have higher priority, can overwrite the one in config file')
    parser.add_argument('--output', type=str,
                        help='output folder, this have higher priority, can overwrite the one in config file')
    
    args = parser.parse_args()



    cfg = config.load_config(args.config)
    if args.input_folder is not None:
        cfg['data']['datadir'] = args.input_folder
    if args.output is not None:
        cfg['data']['output'] = args.output

    print("Saving config and script...")
    save_path = os.path.join(cfg["data"]["output"], cfg['data']['exp_name'])
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    shutil.copy("slcslam.py", os.path.join(save_path, 'slacslam.py'))

    with open(os.path.join(save_path, 'config.json'),"w", encoding='utf-8') as f:
        f.write(json.dumps(cfg, indent=4))

    slam = SLC2SLAM(cfg,size=10000)
    slam.run()