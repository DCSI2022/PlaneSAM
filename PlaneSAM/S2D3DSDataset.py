import os
import torch
from torch.utils.data import Dataset
import random
import numpy as np
from torchvision import transforms
import skimage
from pycocotools.coco import COCO
from PIL import Image

class S2d3dsDataset(Dataset):
    def __init__(self, datafolder, subset='train', transform=None):
        self.datafolder = datafolder
        self.subset = subset
        self.transform = transform
        assert subset in ['train', 'val']
        if subset == 'train':
            self.coco = COCO(annotation_file=os.path.join(datafolder, 's2d3ds_train.json'))
        else:
            self.coco = COCO(annotation_file=os.path.join(datafolder, 's2d3ds_val.json'))
        self.image_ids = self.coco.getImgIds()

    def __len__(self):
        return len(self.image_ids)

    '''
        return:
            image: [3, H, W]
            depth: [H, W]
            segmentation: [H, W]
            instance: [num_planes, H, W]
    '''

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        if self.subset == 'train':
            image_path = os.path.join(self.datafolder, 'images', self.coco.loadImgs(image_id)[0]['file_name'])
        else:
            image_path = os.path.join(self.datafolder, 'images_val', self.coco.loadImgs(image_id)[0]['file_name'])
        depth_path = image_path.replace('images', 'depths').replace('rgb', 'depth').replace('.jpg', '.png')
        image = np.array(Image.open(image_path), dtype=np.uint8)
        depth = np.array(Image.open(depth_path), dtype=np.float32) / 1000.0

        segmentation = np.zeros(image.shape[:2], dtype=np.uint8)
        annotation_id = self.coco.getAnnIds(imgIds=image_id)
        annotation = self.coco.loadAnns(annotation_id)
        for idx, i in enumerate(annotation):
            segmentation[self.coco.annToMask(i) > 0] = idx + 1
            idx += 1

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
            "data_path": image_path
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