import os
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from random import randint



class PlaneDataset(Dataset):
    def __init__(self, subset='train', transform=None, root_dir=None):
        assert subset in ['train', 'val']
        self.subset = subset
        self.transform = transform
        self.root_dir = os.path.join(root_dir, subset)
        self.txt_file = os.path.join(root_dir, subset + '.txt')

        self.data_list = [line.strip() for line in open(self.txt_file, 'r').readlines()]
        self.precompute_K_inv_dot_xy_1()

    def get_plane_parameters(self, plane, plane_nums, segmentation):
        valid_region = segmentation != 20

        plane = plane[:plane_nums]

        tmp = plane[:, 1].copy()
        plane[:, 1] = -plane[:, 2]
        plane[:, 2] = tmp

        # convert plane from n * d to n / d
        plane_d = np.linalg.norm(plane, axis=1)
        # normalize
        plane /= plane_d.reshape(-1, 1)
        # n / d
        plane /= plane_d.reshape(-1, 1)

        h, w = segmentation.shape
        plane_parameters = np.ones((3, h, w))
        for i in range(h):
            for j in range(w):
                d = segmentation[i, j]
                if d >= 20: continue
                plane_parameters[:, i, j] = plane[d, :]

        # plane_instance parameter, padding zero to fix size
        plane_instance_parameter = np.concatenate((plane, np.zeros((20 - plane.shape[0], 3))), axis=0)
        return plane_parameters, valid_region, plane_instance_parameter

    def precompute_K_inv_dot_xy_1(self, h=192, w=256):
        focal_length = 517.97
        offset_x = 320
        offset_y = 240

        K = [[focal_length, 0, offset_x],
             [0, focal_length, offset_y],
             [0, 0, 1]]

        K_inv = np.linalg.inv(np.array(K))
        self.K_inv = K_inv

        K_inv_dot_xy_1 = np.zeros((3, h, w))
        for y in range(h):
            for x in range(w):
                yy = float(y) / h * 480
                xx = float(x) / w * 640

                ray = np.dot(self.K_inv,
                             np.array([xx, yy, 1]).reshape(3, 1))
                K_inv_dot_xy_1[:, y, x] = ray[:, 0]

        # precompute to speed up processing
        self.K_inv_dot_xy_1 = K_inv_dot_xy_1

    def plane2depth(self, plane_parameters, num_planes, segmentation, gt_depth, h=192, w=256):

        depth_map = 1. / np.sum(self.K_inv_dot_xy_1.reshape(3, -1) * plane_parameters.reshape(3, -1), axis=0)
        depth_map = depth_map.reshape(h, w)

        # replace non planer region depth using sensor depth map
        # 做了一个深度修复，并把非平面区域的深度设为0
        depth_map[segmentation == 20] = gt_depth[segmentation == 20]
        return depth_map

    def __getitem__(self, index):
        if self.subset == 'train':
            data_path = self.data_list[index]
        else:
            data_path = str(index) + '.npz'
        data_path = os.path.join(self.root_dir, data_path)
        data = np.load(data_path)

        image = data['image']
        image_path = data['image_path']
        info = data['info']
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)

        if self.transform is not None:
            image = self.transform(image)

        plane = data['plane']
        num_planes = data['num_planes'][0]

        gt_segmentation = data['segmentation']
        gt_segmentation = gt_segmentation.reshape((192, 256))
        segmentation = np.zeros([21, 192, 256], dtype=np.uint8)

        _, h, w = segmentation.shape
        for i in range(num_planes + 1):
            # deal with backgroud
            if i == num_planes:
                seg = gt_segmentation == 20
            else:
                seg = gt_segmentation == i

            segmentation[i, :, :] = seg.reshape(h, w)

        # surface plane parameters
        plane_parameters, valid_region, plane_instance_parameter = \
            self.get_plane_parameters(plane, num_planes, gt_segmentation)

        # since some depth is missing, we use plane to recover those depth following PlaneNet
        gt_depth = data['depth'].reshape(192, 256)
        depth = self.plane2depth(plane_parameters, num_planes, gt_segmentation, gt_depth).reshape(192, 256)

        # Depth图像需要归一化
        if self.transform is not None:
            depth = self.transform(depth)

        sample = {
            'image': image,
            'num_planes': num_planes,
            'instance': torch.FloatTensor(segmentation),
            # one for planar and zero for non-planar
            'semantic': 1 - torch.FloatTensor(segmentation[num_planes, :, :]).unsqueeze(0),
            'gt_seg': torch.LongTensor(gt_segmentation),
            'depth': depth.to(torch.float32),
            'plane_parameters': torch.FloatTensor(plane_parameters),
            'valid_region': torch.ByteTensor(valid_region.astype(np.uint8)).unsqueeze(0),
            'plane_instance_parameter': torch.FloatTensor(plane_instance_parameter),
            'data_path': data_path
        }

        return sample

    def __len__(self):
        return len(self.data_list)

    def masks_to_bboxes(self, masks):
        """
        从掩码张量中计算边界框的左上和右下坐标
        参数:
            masks: 形状为 [N, H, W] 的二进制掩码张量
        返回值:
            bounding_boxes: 形状为 [B, 4] 的边界框坐标张量，包含左上和右下坐标
        """
        batch_size, h, w = masks.size()
        bounding_boxes = torch.zeros((batch_size, 4), dtype=torch.float32)

        for b in range(batch_size):
            mask = masks[b]

            # 找到掩码的非零元素索引
            nonzero_indices = torch.nonzero(mask)

            if nonzero_indices.size(0) == 0:
                # 如果掩码中没有非零元素，则边界框坐标为零
                assert "no mask!"
            else:
                # 计算边界框的左上和右下坐标
                ymin = torch.min(nonzero_indices[:, 0]) / h
                xmin = torch.min(nonzero_indices[:, 1]) / w
                ymax = torch.max(nonzero_indices[:, 0]) / h
                xmax = torch.max(nonzero_indices[:, 1]) / w
                bounding_boxes[b] = torch.tensor([(xmin + xmax) / 2, (ymin + ymax) / 2, xmax - xmin, ymax - ymin], dtype=torch.float32)

        return bounding_boxes