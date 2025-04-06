# package imports
import torch
import torch.nn as nn
import numpy as np
import os
# Local imports
from .encodings import get_encoder
from .decoder import ColorSDFNet, ColorSDFNet_v2,ColorSDFSemNet,SemanticNet
from .utils import sample_pdf, batchify, get_sdf_loss, mse2psnr, compute_loss,compute_cross_entropy_loss
import model.point_cloud_icp as image_utils
class JointEncoding(nn.Module):
    def __init__(self, config, bound_box):
        super(JointEncoding, self).__init__()
        self.config = config
        self.bounding_box = bound_box
        self.get_resolution()
        self.num_class = 41  #35
        self.get_encoding(config)
        self.get_decoder(config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num = []

    def get_resolution(self):
        '''
        Get the resolution of the grid
        '''
        dim_max = (self.bounding_box[:,1] - self.bounding_box[:,0]).max()
        if self.config['grid']['voxel_sdf'] > 10:
            self.resolution_sdf = self.config['grid']['voxel_sdf']
        else:
            self.resolution_sdf = int(dim_max / self.config['grid']['voxel_sdf'])
        
        if self.config['grid']['voxel_color'] > 10:
            self.resolution_color = self.config['grid']['voxel_color']
        else:
            self.resolution_color = int(dim_max / self.config['grid']['voxel_color'])
        
        print('SDF resolution:', self.resolution_sdf)

    def get_encoding(self, config):
        '''
        Get the encoding of the scene representation
        '''
        # Coordinate encoding
        self.embedpos_fn, self.input_ch_pos = get_encoder(config['pos']['enc'], n_bins=self.config['pos']['n_bins'])

        # Sparse parametric encoding (SDF)
        self.embed_fn, self.input_ch = get_encoder(config['grid']['enc'], log2_hashmap_size=config['grid']['hash_size'], desired_resolution=self.resolution_sdf)

        # Sparse parametric encoding (Color)
        if not self.config['grid']['oneGrid']:
            print('Color resolution:', self.resolution_color)
            self.embed_fn_color, self.input_ch_color = get_encoder(config['grid']['enc'], log2_hashmap_size=config['grid']['hash_size'], desired_resolution=self.resolution_color)

    def get_decoder(self, config):
        '''
        Get the decoder of the scene representation
        '''
        if not self.config['grid']['oneGrid']:
            print('using colorsdfnet.')
            self.decoder = ColorSDFNet(config, input_ch=self.input_ch, input_ch_pos=self.input_ch_pos)
        else:
            print('using colorsdfnet_v2.')
            self.decoder = ColorSDFSemNet(config,output_ch_semantic =41,input_ch=self.input_ch, input_ch_pos=self.input_ch_pos)

        self.color_net = batchify(self.decoder.color_net, None)
        self.sdf_net = batchify(self.decoder.sdf_net, None)
        self.sem_net = batchify(self.decoder.semantic_net, None)

    def sdf2weights(self, sdf, z_vals, args=None):
        '''
        Convert signed distance function to weights.

        Params:
            sdf: [N_rays, N_samples]
            z_vals: [N_rays, N_samples]
        Returns:
            weights: [N_rays, N_samples]
        '''
        weights = torch.sigmoid(sdf / args['training']['trunc']) * torch.sigmoid(-sdf / args['training']['trunc'])

        signs = sdf[:, 1:] * sdf[:, :-1]
        mask = torch.where(signs < 0.0, torch.ones_like(signs), torch.zeros_like(signs))
        inds = torch.argmax(mask, axis=1)
        inds = inds[..., None]
        z_min = torch.gather(z_vals, 1, inds) # The first surface
        mask = torch.where(z_vals < z_min + args['data']['sc_factor'] * args['training']['trunc'], torch.ones_like(z_vals), torch.zeros_like(z_vals))

        weights = weights * mask
        return weights / (torch.sum(weights, axis=-1, keepdims=True) + 1e-8)
    
    def raw2outputs(self, raw, z_vals,enable_semantic=False,white_bkgd=False, num_sem_class=0):
        '''
        Perform volume rendering using weights computed from sdf.

        Params:
            raw: [N_rays, N_samples, 4]
            z_vals: [N_rays, N_samples]
        Returns:
            rgb_map: [N_rays, 3]
            disp_map: [N_rays]
            acc_map: [N_rays]
            weights: [N_rays, N_samples]
        '''
        rgb = torch.sigmoid(raw[...,:3])  # [N_rays, N_samples, 3]
        weights = self.sdf2weights(raw[..., 3], z_vals, args=self.config)
        rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]

        depth_map = torch.sum(weights * z_vals, -1)
        depth_var = torch.sum(weights * torch.square(z_vals - depth_map.unsqueeze(-1)), dim=-1)
        disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
        acc_map = torch.sum(weights, -1)

        if enable_semantic :
            assert num_sem_class > 0

            sem_logits = raw[..., 4:4 + num_sem_class]  # [N_rays, N_samples, num_class]
            sem_map = torch.sum(weights[..., None] * sem_logits, -2)  # [N_rays, num_class]

        else:
            sem_map = torch.tensor(0)
        if white_bkgd:
            rgb_map = rgb_map + (1.-acc_map[...,None])
            if enable_semantic:
                sem_map = sem_map + (1. - acc_map[..., None])

        return rgb_map, disp_map, acc_map, weights, depth_map, depth_var, sem_map

    
    def query_local_features(self, query_points: object) -> object:
        '''
        Get the compact code and geometric feature of the query points
        Params:
            query_points: [N_rays, N_samples, 3]
        Returns:
            embed: [N_rays, N_samples, 4]?
            geo_feat: [N_rays, N_samples, channel]
        '''

        inputs_flat = torch.reshape(query_points, [-1, query_points.shape[-1]])#将三维坐标点展位二维数组[N_rays * N_samples, 3]

        embedded = self.embed_fn(inputs_flat)
        embedded_pos = self.embedpos_fn(inputs_flat)

        out = self.sdf_net(torch.cat([embedded, embedded_pos], dim=-1))
        geo_feat =out[..., 1:]


        embedded = torch.reshape(embedded, list(query_points.shape[:-1]) + [embedded.shape[-1]])
        geo_feat = torch.reshape(geo_feat, list(query_points.shape[:-1]) + [geo_feat.shape[-1]])
        return embedded, geo_feat

    def query_sem_feature(self, query_points: object) -> object:


        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        model = self.sem_net.to(device)
        # for param in self.sem_net.parameters():
        #     print(param)
        inputs_flat = torch.reshape(query_points, [-1, query_points.shape[-1]])  # 将三维坐标点展位二维数组[N_rays * N_samples, 3]

        embedded = self.embed_fn(inputs_flat)
        embedded_pos = self.embedpos_fn(inputs_flat)

        out = self.sdf_net(torch.cat([embedded, embedded_pos], dim=-1))
        sdf, geo_feat = out[..., :1], out[..., 1:]

        out_sem = model(torch.cat([embedded_pos, geo_feat], dim=-1))
        sem_pre = torch.argmax(out_sem, dim=-1).cpu()

        return sem_pre
    def query_sem_sdf(self, query_points, return_geo=False, embed=False):
        '''
        Get the SDF value of the query points
        Params:
            query_points: [N_rays, N_samples, 3]
        Returns:
            sdf: [N_rays, N_samples]
            geo_feat: [N_rays, N_samples, channel]
        '''
        inputs_flat = torch.reshape(query_points, [-1, query_points.shape[-1]])

        embedded = self.embed_fn(inputs_flat)
        if embed:
            return torch.reshape(embedded, list(query_points.shape[:-1]) + [embedded.shape[-1]])

        embedded_pos = self.embedpos_fn(inputs_flat)
        out = self.sdf_net(torch.cat([embedded, embedded_pos], dim=-1))
        sdf, geo_feat = out[..., :1], out[..., 1:]

        sdf = torch.reshape(sdf, list(query_points.shape[:-1]))
        if not return_geo:
            out_sem = self.sem_net(torch.cat([embedded_pos, geo_feat], dim=-1))
            sem_pre = torch.argmax(out_sem, dim=-1).cpu()
            return sdf, sem_pre

        return sdf, geo_feat,

    def query_sdf(self, query_points, return_geo=False, embed=False):
        '''
        Get the SDF value of the query points
        Params:
            query_points: [N_rays, N_samples, 3]
        Returns:
            sdf: [N_rays, N_samples]
            geo_feat: [N_rays, N_samples, channel]
        '''
        inputs_flat = torch.reshape(query_points, [-1, query_points.shape[-1]])
  
        embedded = self.embed_fn(inputs_flat)
        if embed:
            return torch.reshape(embedded, list(query_points.shape[:-1]) + [embedded.shape[-1]])

        embedded_pos = self.embedpos_fn(inputs_flat)
        out = self.sdf_net(torch.cat([embedded, embedded_pos], dim=-1))
        sdf, geo_feat = out[..., :1], out[..., 1:]

        sdf = torch.reshape(sdf, list(query_points.shape[:-1]))
        if not return_geo:
            return sdf
        geo_feat = torch.reshape(geo_feat, list(query_points.shape[:-1]) + [geo_feat.shape[-1]])

        return sdf, geo_feat
    
    def query_color(self, query_points):

        return torch.sigmoid(self.query_color_sdf(query_points)[..., :3])
    def query_sem_color(self, query_points):
        sdf, sem_pre= self.query_sem_sdf(query_points)
        label_colormap = image_utils.nyu40_colour_code
        color =self.semantic_color_func(sem_pre.flatten(), label_colormap, 41)
        color = torch.from_numpy(color)
        return color


    def semantic_color_func(self,sem_pre, label_colormap, num_classes=None):
        '''
        Map semantic predictions to colors using the predefined colormap.

        sem_pre: flattened array of semantic predictions (category IDs).
        label_colormap: colormap array where each row corresponds to the RGB color of a label.
        num_classes: number of classes in the semantic prediction (if None, use the shape of label_colormap).
        '''
        if isinstance(sem_pre, torch.Tensor):
            sem_pre = sem_pre.cpu().numpy()
        if num_classes is None:
            num_classes = label_colormap.shape[0]

        colors = np.zeros((sem_pre.size, 3), dtype=np.float32)  # 注意这里改为了4通道RGBA

        # Apply the colors based on the semantic predictions (IDs)
        for i, label in enumerate(sem_pre):
                colors[i, :3] = label_colormap[label] / 255.0

        return colors
    def query_color_sdf(self, query_points,enable_semantic=False):
        '''
        Query the color and sdf at query_points.

        Params:
            query_points: [N_rays, N_samples, 3]
        Returns:
            raw: [N_rays, N_samples, 4]
        '''
        inputs_flat = torch.reshape(query_points, [-1, query_points.shape[-1]])

        embed = self.embed_fn(inputs_flat)
        embe_pos = self.embedpos_fn(inputs_flat)

        if enable_semantic:
            if not self.config['grid']['oneGrid']:
                embed_color = self.embed_fn_color(inputs_flat)
                return self.decoder(embed, embe_pos, embed_color, train_semantic=True)
            return self.decoder(embed, embe_pos, train_semantic=True)
        else:
            if not self.config['grid']['oneGrid']:
                embed_color = self.embed_fn_color(inputs_flat)
                return self.decoder(embed, embe_pos, embed_color)
            return self.decoder(embed, embe_pos)
    def run_network(self, inputs,enable_semantic=False):
        """
        Run the network on a batch of inputs.

        Params:
            inputs: [N_rays, N_samples, 3]
        Returns:
            outputs: [N_rays, N_samples, 4]
        """
        inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
        
        # Normalize the input to [0, 1] (TCNN convention)
        if self.config['grid']['tcnn_encoding']:
            inputs_flat = (inputs_flat - self.bounding_box[:, 0]) / (self.bounding_box[:, 1] - self.bounding_box[:, 0])

        outputs_flat = batchify(self.query_color_sdf, None)(inputs_flat,enable_semantic)
        outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])

        return outputs
    
    def render_surface_color(self, rays_o, normal,enable_semantic=False):
        '''
        Render the surface color of the points.
        Params:
            points: [N_rays, 1, 3]
            normal: [N_rays, 3]
        '''
        n_rays = rays_o.shape[0]
        trunc = self.config['training']['trunc']
        z_vals = torch.linspace(-trunc, trunc, steps=self.config['training']['n_range_d']).to(rays_o)
        z_vals = z_vals.repeat(n_rays, 1)
        # Run rendering pipeline
        
        pts = rays_o[...,:] + normal[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples, 3]
        raw = self.run_network(pts, enable_semantic)
        rgb, disp_map, acc_map, weights, depth_map, depth_var, sem_map = self.raw2outputs(raw, z_vals,
                                                                                          self.config['training'][
                                                                                              'white_bkgd'],
                                                                                          enable_semantic,
                                                                                          num_sem_class=self.num_class)
        return rgb
    
    def render_rays(self, rays_o, rays_d,enable_semantic=False, target_d=None):
        '''
        Params:
            rays_o: [N_rays, 3]
            rays_d: [N_rays, 3]
            target_d: [N_rays, 1]

        '''
        n_rays = rays_o.shape[0]

        # Sample depth
        if target_d is not None:
            z_samples = torch.linspace(-self.config['training']['range_d'], self.config['training']['range_d'], steps=self.config['training']['n_range_d']).to(target_d) 
            z_samples = z_samples[None, :].repeat(n_rays, 1) + target_d
            z_samples[target_d.squeeze()<=0] = torch.linspace(self.config['cam']['near'], self.config['cam']['far'], steps=self.config['training']['n_range_d']).to(target_d) 

            if self.config['training']['n_samples_d'] > 0:
                z_vals = torch.linspace(self.config['cam']['near'], self.config['cam']['far'], self.config['training']['n_samples_d'])[None, :].repeat(n_rays, 1).to(rays_o)
                z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
            else:
                z_vals = z_samples
        else:
            z_vals = torch.linspace(self.config['cam']['near'], self.config['cam']['far'], self.config['training']['n_samples']).to(rays_o)
            z_vals = z_vals[None, :].repeat(n_rays, 1) # [n_rays, n_samples]

        # Perturb sampling depths
        if self.config['training']['perturb'] > 0.:
            mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
            upper = torch.cat([mids, z_vals[...,-1:]], -1)
            lower = torch.cat([z_vals[...,:1], mids], -1)
            z_vals = lower + (upper - lower) * torch.rand(z_vals.shape).to(rays_o)

        # Run rendering pipeline
        pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples, 3]
        raw = self.run_network(pts,enable_semantic)
        if enable_semantic:
            rgb_map, disp_map, acc_map, weights, depth_map, depth_var, sem_map = self.raw2outputs(raw, z_vals,enable_semantic,self.config['training'][ 'white_bkgd']
                                                            ,  num_sem_class=self.num_class)
        else:
            rgb_map, disp_map, acc_map, weights, depth_map, depth_var,_ = self.raw2outputs(raw, z_vals,enable_semantic,self.config['training'][ 'white_bkgd'], num_sem_class=self.num_class)


        # Importance sampling
        if self.config['training']['n_importance'] > 0:

            rgb_map_0, disp_map_0, acc_map_0, depth_map_0, depth_var_0 = rgb_map, disp_map, acc_map, depth_map, depth_var

            z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])
            z_samples = sample_pdf(z_vals_mid, weights[...,1:-1], self.config['training']['n_importance'], det=(self.config['training']['perturb']==0.))
            z_samples = z_samples.detach()

            z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
            pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples + N_importance, 3]

            raw = self.run_network(pts,enable_semantic)

            if enable_semantic:
                rgb_map, disp_map, acc_map, weights, depth_map, depth_var, sem_map = self.raw2outputs(raw, z_vals,enable_semantic,self.config[ 'training'][ 'white_bkgd'],
                                                                                                  num_sem_class=self.num_class)


            else:
                rgb_map, disp_map, acc_map, weights, depth_map, depth_var = self.raw2outputs(raw, z_vals,
                                                                                             self.config['training'][
                                                                                                 'white_bkgd'])

        if enable_semantic:
            # Return rendering outputs
            ret = {'rgb': rgb_map, 'depth': depth_map,
                   'disp_map': disp_map, 'acc_map': acc_map,
                   'depth_var': depth_var, 'sem_map': sem_map}
            ret = {**ret, 'z_vals': z_vals}

            ret['raw'] = raw

            if self.config['training']['n_importance'] > 0:
                ret['rgb0'] = rgb_map_0
                ret['disp0'] = disp_map_0
                ret['acc0'] = acc_map_0
                ret['depth0'] = depth_map_0
                ret['depth_var0'] = depth_var_0
                ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)
        else:
            # Return rendering outputs
            ret = {'rgb' : rgb_map, 'depth' :depth_map,
                   'disp_map' : disp_map, 'acc_map' : acc_map,
                   'depth_var':depth_var,}
            ret = {**ret, 'z_vals': z_vals}

            ret['raw'] = raw

            if self.config['training']['n_importance'] > 0:
                ret['rgb0'] = rgb_map_0
                ret['disp0'] = disp_map_0
                ret['acc0'] = acc_map_0
                ret['depth0'] = depth_map_0
                ret['depth_var0'] = depth_var_0
                ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)

        return ret,raw


    def compute_score(self, hist, correct, labeled):
        # 计算 IoU
        iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))

        # 忽略0值和NaN值计算mean_IU
        iu_non_zero_nan = iu[(iu != 0) & ~np.isnan(iu)]
        mean_IU = np.mean(iu_non_zero_nan) if iu_non_zero_nan.size > 0 else 0

        # 忽略0值和NaN值计算mean_IU_no_back, 假设背景是类别索引0
        iu_no_back = iu[1:]  # 去掉背景类别
        iu_no_back_non_zero_nan = iu_no_back[(iu_no_back != 0) & ~np.isnan(iu_no_back)]
        mean_IU_no_back = np.mean(iu_no_back_non_zero_nan) if iu_no_back_non_zero_nan.size > 0 else 0

        # 计算像素准确率，需要确保labeled不为0，以避免除以0
        mean_pixel_acc = correct / labeled if labeled > 0 else 0

        return iu, mean_IU, mean_IU_no_back, mean_pixel_acc


    def forward(self,batch, rays_o, rays_d, target_rgb, target_d,target_semantic,enable_semantic=False, global_step=0):
        '''
        Params:
            rays_o: ray origins (Bs, 3)
            rays_d: ray directions (Bs, 3)
            frame_ids: use for pose correction (Bs, 1)
            target_rgb: rgb value (Bs, 3)
            target_d: depth value (Bs, 1)
            c2w_array: poses (N, 4, 4)
             r r r tx
             r r r ty
             r r r tz
        '''
        # Get render results
        rend_dict,features = self.render_rays(rays_o, rays_d,enable_semantic ,target_d=target_d)
        self.num_class = batch['num_semantic']

        if not self.training:
            return rend_dict,features
        # Get depth and rgb weights for loss
        valid_depth_mask = (target_d.squeeze() > 0.) * (target_d.squeeze() < self.config['cam']['depth_trunc'])
        rgb_weight = valid_depth_mask.clone().unsqueeze(-1)
        rgb_weight[rgb_weight==0] = self.config['training']['rgb_missing']

        # Get render loss
        rgb_loss = compute_loss(rend_dict["rgb"]*rgb_weight, target_rgb*rgb_weight)
        psnr = mse2psnr(rgb_loss)
        depth_loss = compute_loss(rend_dict["depth"].squeeze()[valid_depth_mask], target_d.squeeze()[valid_depth_mask])




        if enable_semantic:
            sem_logits = rend_dict["sem_map"].cpu()
            target_semantic = target_semantic.cpu()
            semantic_loss = compute_cross_entropy_loss(sem_logits.squeeze(), target_semantic.squeeze())
            pred_labels = torch.argmax(sem_logits, dim=-1).cpu()

            num_classes = self.num_class
            pred_labels_np = pred_labels.numpy().flatten()
            target_semantic_np = target_semantic.numpy().flatten()

            bincount_result = np.bincount(self.num_class * target_semantic_np + pred_labels_np,
                                          minlength=num_classes ** 2)

            confusion_matrix = bincount_result.reshape((num_classes, num_classes))
            correct = np.sum(pred_labels_np == target_semantic_np)
            labeled = np.sum(target_semantic_np > 0)
            iu, mean_IU, mean_IU_no_back, mean_pixel_acc = self.compute_score(confusion_matrix, correct, labeled)

        if 'rgb0' in rend_dict:
            rgb_loss += compute_loss(rend_dict["rgb0"]*rgb_weight, target_rgb*rgb_weight)

            depth_loss += compute_loss(rend_dict["depth0"][valid_depth_mask], target_d.squeeze()[valid_depth_mask])

        # Get sdf loss
        z_vals = rend_dict['z_vals']  # [N_rand, N_samples + N_importance]
        sdf = rend_dict['raw'][..., -1]  # [N_rand, N_samples + N_importance]
        truncation = self.config['training']['trunc'] * self.config['data']['sc_factor']
        fs_loss, sdf_loss = get_sdf_loss(z_vals, target_d, sdf, truncation, 'l2', grad=None)


        if enable_semantic:
            ret = {
                "rgb": rend_dict["rgb"],
                "depth": rend_dict["depth"],
                "semantic":pred_labels,
                "rgb_loss": rgb_loss,
                "depth_loss": depth_loss,
                "sdf_loss": sdf_loss,
                "fs_loss": fs_loss,
                "psnr": psnr,
                "sem_loss":semantic_loss,
                "iu":iu,
                "mean_IU":mean_IU,
                "mean_IU_no_back":mean_IU_no_back,
                "mean_pixel_acc":mean_pixel_acc
            }
        else:
            ret = {
                "rgb": rend_dict["rgb"],
                "depth": rend_dict["depth"],
                "rgb_loss": rgb_loss,
                "depth_loss": depth_loss,
                "sdf_loss": sdf_loss,
                "fs_loss": fs_loss,
                "psnr": psnr,
            }

        return ret,features

