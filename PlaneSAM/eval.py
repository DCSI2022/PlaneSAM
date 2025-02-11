import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim
import torchvision.transforms as transforms
from model import build_efficient_sam_vitt
from FasterRCNN.build_FasterRCNN import create_model
from PlaneDataset import PlaneDataset
from Nyuv2Dataset import Nyuv2Dataset, ToTensor
from S2D3DSDataset import S2d3dsDataset, ToTensor
from torch.utils.data import DataLoader
from utils.make_prompt import preprocess
from utils.train_tools import load_ckpt
from utils.eval_tools import evaluateMasks, match_boxes_gt
from utils.box_ops import masks_to_boxes

parser = argparse.ArgumentParser(description='Segment Any Planes')
parser.add_argument('--data-dir', default='ScanNet', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--num-workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('-o', '--output', default='result', metavar='DIR',
                    help='path to output')
parser.add_argument('--cuda', action='store_true', default=True,
                    help='enables CUDA training')
parser.add_argument('--last-ckpt', default='model/PlaneSAM.pth', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--detector-ckpt', default='weights/FasterRCNN.pth', type=str, metavar='PATH',
                    help='path to detector checkpoint (default: none)')

args = parser.parse_args()
device = torch.device("cuda:0" if args.cuda and torch.cuda.is_available() else "cpu")


def inference():
    model = build_efficient_sam_vitt()
    detector = create_model(num_classes=2, load_pretrain_weights=False)
    detector.load_state_dict(torch.load(args.detector_ckpt)['model'])

    # if torch.cuda.device_count() > 1:
    #     print("Let's use", torch.cuda.device_count(), "GPUs!")
    #     model = nn.DataParallel(model)
    #     detector = nn.DataParallel(detector)

    model.to(device)
    detector.to(device)

    # eval scannet
    val_data = PlaneDataset(subset="val",
                            transform=transforms.Compose([
                                transforms.ToTensor()]),
                            root_dir=args.data_dir
                            )
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=args.num_workers,
                            pin_memory=False)

    # eval mp3d and synthetic
    # val_data = Nyuv2Dataset(subset="val",
    #                          transform=transforms.Compose([
    #                                    ToTensor()]),
    #                          datafolder=args.data_dir
    #                          )
    # val_loader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=args.num_workers,
    #                          pin_memory=False)

    # eval s2d3ds
    # val_data = S2d3dsDataset(subset="val",
    #                          transform=transforms.Compose([
    #                              ToTensor()]),
    #                          datafolder=args.data_dir
    #                          )
    # val_loader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=args.num_workers,
    #                         pin_memory=False)

    if args.last_ckpt:
        load_ckpt(model, None, args.last_ckpt, device, NUM_GPUS=1)
    else:
        print('no ckpt!')

    mRI = .0
    mSC = .0
    mVoI = .0
    num = 0
    total_time = 0
    model.eval()
    detector.eval()
    with torch.no_grad():
        for batch_idx, sample in enumerate(val_loader):
            num_planes = sample['num_planes'][0]
            image = sample['image'].to(device)
            target = sample['instance'].to(device)
            target = target[:, :num_planes, :, :].permute(1, 0, 2, 3)  # [num_planes, B, H, W]
            gt_boxes = masks_to_boxes(target)
            depth = sample['depth'].to(device)

            # use gt box
            # batch_points, batch_labels = preprocess(target.flatten(0, 1), num_points=0, box=True)

            # 预测边界框 to prompt
            start_time = time.time()
            outputs = detector(image)[0]
            boxes, scores = outputs['boxes'], outputs['scores']
            batch_points, batch_labels = match_boxes_gt(boxes, gt_boxes)

            # union prompt
            pred_masks, pred_ious = model(image, depth, batch_points, batch_labels)
            end_time = time.time()
            total_time += end_time - start_time
            pred_masks = pred_masks.permute(1, 0, 2, 3, 4)
            pred_ious = pred_ious.permute(1, 0, 2)

            # # use preprocess
            # batch_points, batch_labels = preprocess(target.flatten(0, 1), num_points=0, box=True)

            # 二值掩码
            pred_masks = (pred_masks > 0.).float()
            b, num_prompt, per_prompt_mask, h, w = pred_masks.shape
            # [3, B, H, W]
            # [3, B]
            pred_masks = pred_masks.view(b, -1, h, w).permute(1, 0, 2, 3)
            pred_ious = pred_ious.view(b, -1).permute(1, 0)
            best_id = torch.argmax(pred_ious, dim=0)
            # [num_preds, H, W]
            best_mask = pred_masks[best_id, torch.arange(b)]
            target = target.squeeze(1)

            # 计算指标
            RI, VoI, SC = evaluateMasks(best_mask, target)
            mRI += RI
            mSC += SC
            mVoI += VoI
            num += 1

            print(f"iter: {batch_idx}  mRI: {RI:.4f}  mSC: {SC:.4f}  VoI: {VoI:.4f}")

    mRI = mRI / num
    mSC = mSC / num
    mVoI = mVoI / num
    print(f"mRI: {mRI:.3f}  mSC: {mSC:.3f}  VoI: {mVoI:.3f}")
    print("img/s:", len(val_loader) / total_time)


if __name__ == '__main__':
    if not os.path.exists(args.output):
        os.mkdir(args.output)
    inference()
