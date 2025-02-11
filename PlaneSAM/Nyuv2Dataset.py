import os
import torch
from torch.utils.data import Dataset
import random
import numpy as np
from torchvision import transforms
import skimage

class Nyuv2Dataset(Dataset):
    def __init__(self, datafolder, subset='train', transform=None):
        self.datafolder = datafolder
        self.subset = subset
        self.transform = transform
        assert subset in ['train', 'val']
        self.data_list = [os.path.join(datafolder, subset, x) for x in os.listdir(os.path.join(datafolder, subset))]

    def __len__(self):
        return len(self.data_list)

    '''
        return:
            image: [3, H, W]
            depth: [H, W]
            segmentation: [H, W]
            instance: [num_planes, H, W]
    '''
    def __getitem__(self, index):
        data_path = self.data_list[index]
        data = np.load(data_path, allow_pickle=True)
        data = np.load(self.data_list[index], allow_pickle=True)
        image = data[:, :, :3].astype(np.uint8)
        depth = data[:, :, 3]
        segmentation = data[:, :, 4].astype(np.uint8)

        sample = {}
        if self.transform:
            sample = self.transform({
                'image': image,
                'depth': depth,
                'segmentation': segmentation
            })
            image = sample['image']
            depth = sample['depth']
            segmentation = sample['segmentation']

        mask = []
        unique_idx = torch.unique(segmentation)
        unique_idx = [x for x in unique_idx if x]
        for i in unique_idx:
            mask.append((segmentation == i).float())
        mask = torch.stack(mask)
        bbox = self.masks_to_bboxes(mask)
        num_planes = len(unique_idx)

        masks = torch.zeros(30, image.shape[1], image.shape[2], dtype=torch.float32)
        masks[:num_planes] = mask

        sample.update({
            'instance': masks,
            'num_planes': num_planes,
            'data_path': data_path
        })

        return sample

    def masks_to_bboxes(self, masks):
        """
        从掩码张量中计算边界框的左上和右下坐标
        参数:
            masks: 形状为 [B, H, W] 的二进制掩码张量
        返回值:
            bounding_boxes: 形状为 [B, 4] 的边界框坐标张量，包含左上和右下坐标
        """
        batch_size, height, width = masks.size()
        device = masks
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
                ymin = torch.min(nonzero_indices[:, 0])
                xmin = torch.min(nonzero_indices[:, 1])
                ymax = torch.max(nonzero_indices[:, 0])
                xmax = torch.max(nonzero_indices[:, 1])
                bounding_boxes[b] = torch.tensor([xmin, ymin, xmax, ymax], dtype=torch.float32)

        return bounding_boxes


class ToTensor(object):
    def __call__(self, sample):
        image, depth, segmentation = sample['image'], sample['depth'], sample['segmentation']
        # [H, W, C] -> [C, H, W], 像素值归一化到0-1之间
        image = transforms.ToTensor()(image)
        # [1, H, W]
        depth = transforms.ToTensor()(depth)
        return {
            'image': image,
            'depth': depth,
            'segmentation': torch.from_numpy(segmentation.astype(np.int16)).float()
        }

class RandomFlip(object):
    def __call__(self, sample):
        image, depth, segmentation = sample['image'], sample['depth'], sample['segmentation']
        if random.random() > 0.5:
            image = np.fliplr(image).copy()
            depth = np.fliplr(depth).copy()
            segmentation = np.fliplr(segmentation).copy()

        return {
            'image': image,
            'depth': depth,
            'segmentation': segmentation
        }

# 随机缩放
class RandomScale(object):
    def __init__(self, scale):
        self.scale_low = min(scale)
        self.scale_high = max(scale)

    def __call__(self, sample):
        image, depth, segmentation = sample['image'], sample['depth'], sample['segmentation']

        target_scale = random.uniform(self.scale_low, self.scale_high)
        # (H, W, C)
        target_height = int(round(target_scale * image.shape[0]))
        target_width = int(round(target_scale * image.shape[1]))
        # Bi-linear
        image = skimage.transform.resize(image, (target_height, target_width),
                                         order=1, mode='reflect', preserve_range=True).astype(np.uint8)
        # Nearest-neighbor
        depth = skimage.transform.resize(depth, (target_height, target_width),
                                         order=0, mode='reflect', preserve_range=True).astype(np.uint8)
        segmentation = skimage.transform.resize(segmentation, (target_height, target_width),
                                        order=0, mode='reflect', preserve_range=True)

        return {'image': image, 'depth': depth, 'segmentation': segmentation}

# 随机裁剪
class RandomCrop(object):
    def __init__(self, th, tw):
        self.th = th
        self.tw = tw

    def __call__(self, sample):
        image, depth, segmentation = sample['image'], sample['depth'], sample['segmentation']
        h = image.shape[0]
        w = image.shape[1]
        i = random.randint(0, h - self.th)
        j = random.randint(0, w - self.tw)

        return {'image': image[i:i + image_h, j:j + image_w, :],
                'depth': depth[i:i + image_h, j:j + image_w],
                'segmentation': segmentation[i:i + image_h, j:j + image_w]}