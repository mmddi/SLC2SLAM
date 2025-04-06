import random
import typing
import torch
import numpy as np
from .place_recog import VLAD, BoW
from collections import Counter
PT_NUM=1024

class PlaceRecogDatabase(object):
    def __init__(self, config, device, aggregate='vlad') :
        self.config = config
        self.device = device
        self.place_frames = {}   # frame_id: (rays_cam_d, target_d, c2w)    地点帧所有信息
        self.con_frames = {}
        self.con_id = []
        self.frm2features = {}
        self.frame_ids = None   #仅有地点帧索引
        self.change = None
        if aggregate.lower() == 'vlad':
            self.model = VLAD(config,num_clusters=64)
            self.x = 1
        elif aggregate.lower() == 'bow':
            self.model = BoW(num_clusters=64)
            self.x = 2
        else:
            raise NotImplementedError(f'Aggregate method {aggregate} not implemented')

    def __len__(self):
        if self.frame_ids == None:
            return 0
        else:
            return len(self.frame_ids)

    def get_length(self):
        return self.__len__()

    def attach_ids(self, frame_ids):
        '''
        Attach the frame ids to list
        '''

        if self.frame_ids is None:
            self.frame_ids = frame_ids
        else:
            self.frame_ids = torch.cat([self.frame_ids, frame_ids], dim=0)

    def check_projection(self,
                         cur_depth: torch.Tensor,
                         cur_dirs: torch.Tensor,
                         K: torch.Tensor,
                         c2r: torch.Tensor) -> float:
        H, W = cur_depth.shape
        device = self.device  # Assuming `self.device` is set to the desired device (e.g., 'cuda:0' or 'cpu')

        # Move tensors to the correct device
        cur_depth = cur_depth.to(device)
        cur_dirs = cur_dirs.to(device)
        K = K.to(device)
        c2r = c2r.to(device)

        cur_deps = torch.cat(3 * [cur_depth[..., None]], -1).to(device)  # Depth map extended to 3D vector
        pcd_ref = (cur_deps * cur_dirs).to(device)
        pcd_ref = pcd_ref.reshape((-1, 3)).to(device)  # W*H, 3
        pcd_ref = torch.t(pcd_ref).to(device)  # 3, H*W (point cloud for each pixel)

        # Apply transformation
        R = c2r[:3, :3]
        t = c2r[:3, 3]
        pcd_cur = torch.matmul(R, pcd_ref)  # Rotate point cloud according to camera transformation matrix
        pcd_cur = pcd_cur + t[..., None]  # Add translation

        # Compute projection coordinates
        pcd_u = K[0][0] * (pcd_cur[0, :] / pcd_cur[2, :]) + K[0][2]
        pcd_v = K[1][1] * (pcd_cur[1, :] / pcd_cur[2, :]) + K[1][2]

        # Check range of points
        u_in_range = ((pcd_u > 0).int() + (pcd_u < W).int()) > 1
        v_in_range = ((pcd_v > 0).int() + (pcd_v < H).int()) > 1
        cnt = torch.sum((u_in_range.int() + v_in_range.int()) > 1)

        return cnt / float(H * W)



    def covisible_frame(self, batch, sem_logits,indice,c2w_est=None):
        '''
        Params:
            batch['frame_id']: frame id of current frame
            batch['c2w']: Ground truth camera pose [B, 4, 4]
            batch['rgb']: RGB image [B, H, W, 3]
            batch['depth']: Depth image [B, H, W, 1]
            batch['direction']: Ray direction [B, H, W, 3]
            batch['intrinsic']: intrinsic matrix [B, 3, 3]
            c2w_est: estimated or optmized camera pose [4, 4]
        '''
        # use gt or est pose
        ### !!! check where poses are updated and how to determine which Kf pose is optimized !!! ###
        if c2w_est is None:
            print('Using ground truth pose', batch['frame_id'])
            c2w = batch['c2w'][0]    # [4, 4]
        else:
            c2w = c2w_est

        # edge
        iW = self.config['tracking']['ignore_edge_W']
        iH = self.config['tracking']['ignore_edge_H']

        # get direction rays
        rays_d_cam = batch['direction'].squeeze(0)[iH:-iH, iW:-iW, :].reshape(-1, 3)#.to(self.device)
        rays_d_cam_avg = torch.mean(rays_d_cam, dim=0).unsqueeze(0)
        rays_d_avg = torch.t(torch.sum(rays_d_cam_avg[..., None, :] * c2w[:3, :3], -1))        # [N_rays, 3]

        if batch['frame_id'].cpu().numpy()[0] == 0:
            pass
        else:
            # check for overlaps
            # get the last frme in existing place_frames
            last_frame_id = self.con_id[-1]
            (_, _, r2w, r_rays_d_avg) = list(self.con_frames[last_frame_id])
            c2r = torch.matmul(torch.linalg.inv(r2w), c2w)
            overlap = self.check_projection(batch['depth'][0, ...], batch['direction'][0, ...], batch['intrinsic'][0, ...], c2r)
            dir_changes = torch.matmul(rays_d_avg.t(), r_rays_d_avg)
            if overlap > 0.45 and dir_changes > 0. and last_frame_id != batch['frame_id'].cpu().numpy()[0] :
                print(
                    f"{batch['frame_id'].cpu().numpy()[0]}-{last_frame_id}, overlap={overlap}, d_c*d_r={dir_changes}, skipped.")
                return False
            if abs(batch['frame_id'].cpu().numpy()[0] - last_frame_id)  < 50 and last_frame_id != batch['frame_id'].cpu().numpy()[0] :
                print(f"{batch['frame_id'].cpu().numpy()[0]}-{last_frame_id}, < 50. ")
                return False
            if last_frame_id == batch['frame_id'].cpu().numpy()[0]:
                return True
        # rgb-d
        target_d = batch['depth'].squeeze(0)[iH:-iH, iW:-iW].reshape(-1, 1)  # .to(self.device) # [N_rays, 1]
        idxs = self.sampleidx_select(sem_logits, indice, rays_d_cam.shape[0])
        # idxs = random.sample(range(0, rays_d_cam.shape[0]), PT_NUM)
        rays_d_cam = rays_d_cam[idxs, :]
        target_d = target_d[idxs, :]
        self.con_frames[batch['frame_id'].cpu().numpy()[0]] = (rays_d_cam, target_d, c2w, rays_d_avg)
        if batch['frame_id'].cpu().numpy()[0] != 0:
            self.con_id.append(batch['frame_id'].cpu().numpy()[0])
        return True

    def add_placeframe(self, batch, c2w_est=None):
        '''
        Params:
            batch['frame_id']: frame id of current frame
            batch['c2w']: Ground truth camera pose [B, 4, 4]
            batch['rgb']: RGB image [B, H, W, 3]
            batch['depth']: Depth image [B, H, W, 1]
            batch['direction']: Ray direction [B, H, W, 3]
            batch['intrinsic']: intrinsic matrix [B, 3, 3]
            c2w_est: estimated or optmized camera pose [4, 4]
        '''
        # use gt or est pose
        ### !!! check where poses are updated and how to determine which Kf pose is optimized !!! ###
        if c2w_est is None:
            print('Using ground truth pose', batch['frame_id'])
            c2w = batch['c2w'][0]  # [4, 4]
        else:
            c2w = c2w_est

        # edge
        iW = self.config['tracking']['ignore_edge_W']
        iH = self.config['tracking']['ignore_edge_H']

        # get direction rays
        rays_d_cam = batch['direction'].squeeze(0)[iH:-iH, iW:-iW, :].reshape(-1, 3)  # .to(self.device)
        rays_d_cam_avg = torch.mean(rays_d_cam, dim=0).unsqueeze(0)
        rays_d_avg = torch.t(torch.sum(rays_d_cam_avg[..., None, :] * c2w[:3, :3], -1))  # [N_rays, 3]

        # check for overlaps
        for id, (_, _, r2w, r_rays_d_avg) in self.place_frames.items():
            c2r = torch.matmul(torch.linalg.inv(r2w), c2w)

            overlap = self.check_projection(batch['depth'][0, ...], batch['direction'][0, ...],
                                            batch['intrinsic'][0, ...], c2r)

            dir_changes = torch.matmul(rays_d_avg.t(), r_rays_d_avg)
            if overlap > 0.3 and dir_changes > 0.:
                print(f"{batch['frame_id'].cpu().numpy()[0]}-{id}, overlap={overlap}, d_c*d_r={dir_changes}, skipped.")
                return False
            if abs(batch['frame_id'].cpu().numpy()[0] - self.frame_ids[-1])  < 100 :
                print(f"{batch['frame_id'].cpu().numpy()[0]}-{self.frame_ids[-1]}, < 100. ")
                return False
        return True


    def sampleidx_select(self,sem_label , indice,picture_size):

        sem_labels = []
        for i in range(len(sem_label)):
            sem_labels.append(int(sem_label[i].item()))
        label_counts = Counter(sem_labels)

        # total_rays = sem_labels.shape[0]
        total_rays = len(sem_labels)
        label_proportions = {label: count / total_rays for label, count in label_counts.items()}
        num_samples = PT_NUM
        label_samples = {label: int(num_samples * prop) for label, prop in label_proportions.items()}
        sem_labels = np.array(sem_labels)
        sampled_indices = []
        for label, count in label_samples.items():
            label_indices = np.nonzero(sem_labels == label)[0]
            if len(label_indices) > 0:
                sampled_label_indices = np.random.choice(label_indices, size=min(count, len(label_indices)),
                                                         replace=False)
                sampled_indices.extend(sampled_label_indices)

        valid_indices_set = set(range(picture_size))
        for idx in indice:
            valid_indices_set.discard(idx)
        idxs = random.sample(list(valid_indices_set), num_samples - len(sampled_indices))
        sampled_indices.extend(idxs)
        if len(sampled_indices) < num_samples:
            additional_indices = np.random.choice(np.arange(total_rays), size=num_samples - len(sampled_indices),
                                                  replace=False)
            sampled_indices.extend(additional_indices)
        return sampled_indices


    def check_for_nan(self,tensor, name):
        if torch.isnan(tensor).any():
            print(f"NaN values found in {name}")
        else:
            print(f"No NaN values in {name}")

    def check_for_inf(self,tensor, name):
        if torch.isinf(tensor).any():
            print(f"Inf values found in {name}")
        else:
            print(f"No Inf values in {name}")

    def add_place_data(self,batch,sem_logits,indice, c2w_est):
        print(f"---- Adding {batch['frame_id'].cpu().numpy()[0]}.")
        selected_semantic_true = []


        c2w = c2w_est.cpu()
        semantic_flatten = batch['semantic'].squeeze(0).view(-1)
        # semantic_labels = semantic_flatten[indices_cpu]
        for idx in indice:
            selected_semantic_true.append(semantic_flatten[idx])

        # edge
        iW = self.config['tracking']['ignore_edge_W']
        iH = self.config['tracking']['ignore_edge_H']

            # get direction rays
        rays_d_cam = batch['direction'].squeeze(0)[iH:-iH, iW:-iW, :].reshape(-1, 3)  # .to(self.device)
        rays_d_cam_avg = torch.mean(rays_d_cam, dim=0).unsqueeze(0)

        rays_d_avg = torch.t(torch.sum(rays_d_cam_avg[..., None, :] * c2w[:3, :3], -1))  # [N_rays, 3]

        target_d = batch['depth'].squeeze(0)[iH:-iH, iW:-iW].reshape(-1, 1)  # .to(self.device) # [N_rays, 1]

        idxs = self.sampleidx_select(sem_logits, indice,rays_d_cam.shape[0])



        rays_d_cam = rays_d_cam[idxs, :]
        target_d = target_d[idxs, :]


        # # check c2w
        # self.check_for_nan(c2w, "c2w")
        # self.check_for_inf(c2w, "c2w")
        #
        # # check rays_d_avg
        # self.check_for_nan(rays_d_avg, "rays_d_avg")
        # self.check_for_inf(rays_d_avg, "rays_d_avg")


        if torch.isnan(rays_d_cam).any() or torch.isinf(rays_d_cam).any():

            bad_indices = (torch.isnan(rays_d_cam) | torch.isinf(rays_d_cam)).nonzero(as_tuple=False)
            print(f"Bad indices in rays_d_cam: {bad_indices}")


        # store the place frame

        self.attach_ids(batch['frame_id'])
        self.place_frames[batch['frame_id'].cpu().numpy()[0]] = (rays_d_cam, target_d, c2w, rays_d_avg)
        self.con_frames[batch['frame_id'].cpu().numpy()[0]] = (rays_d_cam, target_d, c2w, rays_d_avg)
        self.con_id.append(batch['frame_id'].cpu().numpy()[0])


    def extract_pts(self, batch,sem_logits,indice, c2w,feature_fn):

        iW = self.config['tracking']['ignore_edge_W']
        iH = self.config['tracking']['ignore_edge_H']
        device = self.device
        # 获取射线方向和原点
        rays_d_cam = batch['direction'].squeeze(0)[iH:-iH, iW:-iW, :].reshape(-1, 3)
        rays_d_cam = rays_d_cam.cpu()  # 确保 rays_d_cam 在正确的设备上
        c2w = c2w.cpu()
        idxs = self.sampleidx_select(sem_logits, indice,rays_d_cam.shape[0])
        rays_d_cam = rays_d_cam[idxs, :]
        rays_o = c2w[:3, -1].repeat(rays_d_cam.shape[0], 1)
        rays_d = torch.sum(rays_d_cam[..., None, :] * c2w[:3, :3], -1)
        target_d = batch['depth'].squeeze(0)[iH:-iH, iW:-iW].reshape(-1, 1)
        target_d = target_d[idxs, :]

        n_rays = rays_o.shape[0]
        z_samples = torch.linspace(-self.config['training']['range_d'], self.config['training']['range_d'],
                                   steps=self.config['training']['n_range_d']).to(target_d)



        z_samples = z_samples[None, :].repeat(n_rays, 1) + target_d

        # replace invalid depth pts with sample from range near to far
        z_samples[target_d.squeeze() <= 0] = torch.linspace(self.config['cam']['near'], self.config['cam']['far'],
                                                            steps=self.config['training']['n_range_d']).to(target_d)
        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_samples[..., :, None]  # [N_rays, N_samples, 3]
        with torch.no_grad():
            voxel_code, geom_feature = feature_fn(pts.to(device))
        return voxel_code

    def update_placeframes_features(self, feature_fn,en):
        '''
        feature_fn: feature extractor function from scene_rep.py
        '''
        print(f'-- Updating {len(self.place_frames)} place frames, existing place frame ids: {self.place_frames.keys()}')

        # local feature holder
        all_local_features = None

        for id, (rays_d_cam, target_d, c2w, _) in self.place_frames.items():
            rays_o = c2w[:3, -1].repeat(rays_d_cam.shape[0], 1)    # [N_rays, 3]
            rays_d = torch.sum(rays_d_cam[..., None, :] * c2w[:3, :3], -1)        # [N_rays, 3]
            n_rays = rays_o.shape[0]
            z_samples = torch.linspace(-self.config['training']['range_d'], self.config['training']['range_d'], steps=self.config['training']['n_range_d']).to(target_d)
            z_samples = z_samples[None, :].repeat(n_rays, 1) + target_d
            # replace invalid depth pts with sample from range near to far
            z_samples[target_d.squeeze()<=0] = torch.linspace(self.config['cam']['near'], self.config['cam']['far'], steps=self.config['training']['n_range_d']).to(target_d)

            pts = rays_o[...,None,:] + rays_d[...,None,:] * z_samples[...,:,None] # [N_rays, N_samples, 3]

            with torch.no_grad():
                voxel_code, geom_feature = feature_fn(pts.to(self.device))
                ### !!! decide which to use: voxel_code OR geom_feature OR cat[voxel_code, geom_feature] !!! ###
                local_feature = voxel_code.cpu().numpy()
                local_feature = local_feature.reshape(-1, local_feature.shape[-1])
                # frame id to local feature
                self.frm2features[id] = local_feature

                # store all DB local features
                if all_local_features is None:
                    all_local_features = local_feature # [N_rays*N_samples, 32]
                else:
                    all_local_features = np.concatenate([all_local_features, local_feature], axis=0)
        self.model.fit(all_local_features, self.frm2features)
        return

    def search_loops(self, feature_fn,frame_id,batch,step,c2w_est=None):
        '''
        Params:
            feature_fn: feature extractor function from scene_rep.py

            batch['frame_id']: frame id of current frame
            batch['c2w']: Ground truth camera pose [B, 4, 4]
            batch['rgb']: RGB image [B, H, W, 3]
            batch['depth']: Depth image [B, H, W, 1]
            batch['direction']: Ray direction [B, H, W, 3]
            batch['intrinsic']: intrinsic matrix [B, 3, 3]

            c2w_est: estimated or optmized camera pose [4, 4]
        '''

        if c2w_est is None:
                print('Using ground truth pose', batch['frame_id'])
                c2w = batch['c2w'][0]    # [4, 4]
        else:
                c2w = c2w_est

            # edge
        iW = self.config['tracking']['ignore_edge_W']
        iH = self.config['tracking']['ignore_edge_H']
            # get direction rays
        rays_d_cam = batch['direction'].squeeze(0)[iH:-iH, iW:-iW, :].reshape(-1, 3)  # .to(self.device)
            # rgb-d
            # target_s = batch['rgb'].squeeze(0)[iH:-iH, iW:-iW, :]#.to(self.device)              # [N_rays, 3]
        target_d = batch['depth'].squeeze(0)[iH:-iH, iW:-iW].reshape(-1, 1)  # .to(self.device) # [N_rays, 1]

        idxs = random.sample(range(0, rays_d_cam.shape[0]), PT_NUM)
        rays_d_cam = rays_d_cam[idxs, :]
        target_d = target_d[idxs, :]

        rays_o = c2w[:3, -1].repeat(rays_d_cam.shape[0], 1)    # [N_rays, 3]

        rays_d = torch.sum(rays_d_cam[..., None, :] * c2w[:3, :3], -1)        # [N_rays, 3]


        n_rays = rays_o.shape[0]
        z_samples = torch.linspace(-self.config['training']['range_d'], self.config['training']['range_d'], steps=self.config['training']['n_range_d']).to(target_d)
        z_samples = z_samples[None, :].repeat(n_rays, 1) + target_d
            # replace invalid depth pts with sample from range near to far
        z_samples[target_d.squeeze()<=0] = torch.linspace(self.config['cam']['near'], self.config['cam']['far'], steps=self.config['training']['n_range_d']).to(target_d)
        pts = rays_o[...,None,:] + rays_d[...,None,:] * z_samples[...,:,None] # [N_rays, N_samples, 3]
        with torch.no_grad():
            voxel_code, geom_feature = feature_fn(pts.to(self.device))


            ### !!! decide which to use: voxel_code OR geom_feature OR cat[voxel_code, geom_feature] !!! ###
            local_feature = voxel_code.cpu().numpy()
            # local_feature = geom_feature.cpu().numpy()
            # local_feature = torch.cat([voxel_code, geom_feature], dim=-1).cpu().numpy()

            local_feature = local_feature.reshape(-1, local_feature.shape[-1])
            if self.x == 2 :
                placeframe_id, loop_score = self.model.predict(local_feature,frame_id)

                if placeframe_id > 0:
                    if placeframe_id == len(self.frame_ids) - 1:
                        print(f'---- Find loop at: {placeframe_id} - {loop_score}, but it is the last place frame')
                    else:
                        print(f'!!!! Find loop at: {placeframe_id} - {loop_score}')
                        # exit()
                        ### !!! Add code to link the loop in pose graph optimization !!! ###


                return placeframe_id, loop_score

            else:
                if local_feature is not None:

                    placeframe_id, loop_score = self.model.predict(self.frame_ids, frame_id,self.frm2features,local_feature)
                    return placeframe_id, loop_score
                else:
                    return None,None
