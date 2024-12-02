import torch
import numpy as np
from PIL import Image
import os


def mkdir(pth):
    if not os.path.exists(pth):
        os.makedirs(pth)


class VoxelGrid():
    def __init__(self, input_size, normalize=True, norm_type='mean_std', device='cpu', keep_shape=False):
        assert len(input_size) == 3
        assert norm_type in ['mean_std', 'min_max']

        self.voxel_grid = torch.zeros((input_size), dtype=torch.float, requires_grad=False, device=device)
        self.nb_channels = input_size[0]
        self.normalize = normalize
        self.norm_type = norm_type
        self.keep_shape = keep_shape

    def convert(self, events):
        C, H, W = self.voxel_grid.shape
        with torch.no_grad():
            if 'batch_index' not in events.keys():
                bs = 1
                batch_index = torch.zeros_like(events['x'], dtype=torch.long)
            else:
                bs = torch.max(events['batch_index'])+1
                batch_index = events['batch_index']

            voxel_grid = torch.stack([self.voxel_grid]*bs, dim=0)

            t_norm = events['t']
            for i in range(bs):
                mask = batch_index == i
                if torch.sum(mask) < 1: continue
                if torch.sum(mask) < 2:
                    t_norm[mask] = 0
                    continue
                t_min = t_norm[mask][0]
                t_max = t_norm[mask][-1]
                t_norm[mask] = (C - 1) * (t_norm[mask]-t_min) / (t_max-t_min)

            x0 = events['x'].int()
            y0 = events['y'].int()
            t_norm = t_norm.float()
            t0 = t_norm.int()

            value = 2*events['p']-1

            for xlim in [x0,x0+1]:
                for ylim in [y0,y0+1]:
                    for tlim in [t0,t0+1]:

                        mask = (xlim < W) & (xlim >= 0) & (ylim < H) & (ylim >= 0) & (tlim >= 0) & (tlim < self.nb_channels)
                        interp_weights = value * (1 - (xlim-events['x']).abs()) * (1 - (ylim-events['y']).abs()) * (1 - (tlim - t_norm).abs())

                        index = batch_index * C * H * W + \
                                H * W * tlim.long() + \
                                W * ylim.long() + \
                                xlim.long()
                        voxel_grid.put_(index[mask], interp_weights[mask], accumulate=True)

            if self.normalize:
                for i in range(bs):
                    if self.norm_type == 'min_max':
                        maxv = torch.max(voxel_grid[i].abs())
                        voxel_grid[i] = voxel_grid[i] / maxv
                    elif self.norm_type == 'mean_std':
                        mask = torch.nonzero(voxel_grid[i], as_tuple=True)
                        if mask[0].size()[0] > 0:
                            mean = voxel_grid[i].mean()
                            std = voxel_grid[i].std()
                            if std > 0:
                                voxel_grid[i] = (voxel_grid[i] - mean) / std
                            else:
                                voxel_grid[i] = voxel_grid[i] - mean

        if bs == 1 and not self.keep_shape:
            voxel_grid = voxel_grid[0]

        return voxel_grid


def process_image(event_image, threshold):
    img = np.array(event_image).astype(np.float32)
    img_mean = img.mean()
    normalized_img = (img - img_mean)
    
    result_img = np.ones_like(img) * 255
    
    blue_color = [0, 0, 255]
    red_color = [255, 0, 0]
    
    result_img = np.where(np.all(normalized_img > threshold, axis=-1, keepdims=True), red_color, result_img)
    result_img = np.where(np.all(normalized_img < -threshold, axis=-1, keepdims=True), blue_color, result_img)

    result_img = result_img.astype(np.uint8)

    return Image.fromarray(result_img)