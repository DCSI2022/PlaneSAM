import argparse
import os
import tifffile
import time
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from efficient_sam import build_efficient_sam_vitt
# from raw_efficient_sam import build_efficient_sam_vitt
from FasterRCNN.build_FasterRCNN import create_model
from PlaneDataset import PlaneDataset
from Nyuv2Dataset import Nyuv2Dataset, ToTensor
# from S2D3DSDataset import S2d3dsDataset, ToTensor
from torch.utils.data import DataLoader
from utils.make_prompt import preprocess
from utils.utils import load_ckpt
from utils.visual_tools import map_masks_to_colors
from utils.eval_tools import box_to_prompt, MatchSegmentation, evaluateMasks, match_boxes_gt
from utils.box_ops import masks_to_boxes
from PIL import Image
from tqdm import tqdm


parser = argparse.ArgumentParser(description='Segment Any Planes')
parser.add_argument('--data-dir', default='mp3d-plane', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--num-workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('-o', '--output', default='result', metavar='DIR',
                    help='path to output')
parser.add_argument('--cuda', action='store_true', default=True,
                    help='enables CUDA training')
parser.add_argument('--last-ckpt', default='./model/usepre_best.pth', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--detector-ckpt', default='FasterRCNN_weights/resNet101Fpn-model-9.pth', type=str, metavar='PATH',
                    help='path to detector checkpoint (default: none)')

args = parser.parse_args()
device = torch.device("cuda:0" if args.cuda and torch.cuda.is_available() else "cpu")


def inference():
    model = build_efficient_sam_vitt()
    detector = create_model(num_classes=2, load_pretrain_weights=False)
    detector.load_state_dict(torch.load(args.detector_ckpt)['model'])

    if torch.cuda.device_count() >= 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    model.to(device)
    detector.to(device)

    # val_data = PlaneDataset(subset="train",
    #                         transform=transforms.Compose([
    #                             transforms.ToTensor()]),
    #                         root_dir=args.data_dir
    #                         )
    # val_loader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=args.num_workers,
    #                         pin_memory=False)

    val_data = Nyuv2Dataset(subset="val",
                            transform=transforms.Compose([
                                ToTensor()]),
                            datafolder=args.data_dir
                            )
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=args.num_workers,
                            pin_memory=False)

    # val_data = S2d3dsDataset(subset="val",
    #                          transform=transforms.Compose([
    #                              ToTensor()]),
    #                          datafolder=args.data_dir
    #                          )
    # val_loader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=args.num_workers,
    #                         pin_memory=False)

    if args.last_ckpt:
        load_ckpt(model, None, args.last_ckpt, device)
    else:
        print('no ckpt!')

    model.eval()
    detector.eval()

    total_time = 0

    with torch.no_grad():
        for batch_idx, sample in enumerate(tqdm(val_loader)):
            num_planes = sample['num_planes'][0]
            image = sample['image'].to(device)
            target = sample['instance'].to(device)
            target = target[:, :num_planes, :, :].permute(1, 0, 2, 3)  # [num_planes, B, H, W]
            gt_boxes = masks_to_boxes(target)
            depth = sample['depth'].to(device)

            # # 预测边界框 to prompt
            # outputs = detector(image)[0]
            # boxes = outputs['boxes']
            # batch_points, batch_labels = match_boxes_gt(boxes, gt_boxes)

            # # use preprocess
            # batch_points, batch_labels = preprocess(target.flatten(0, 1), num_points=0, box=True)

            pred_masks = []
            pred_ious = []

            start_time = time.time()

            for input_points, input_labels in zip(batch_points, batch_labels):
                input_points, input_labels = input_points.unsqueeze(0), input_labels.unsqueeze(0)
                # [B, num_prompt, per_prompt_mask, H, W]
                pred_mask, pred_iou = model(image, depth, input_points, input_labels)
                pred_masks.append(pred_mask)
                pred_ious.append(pred_iou)

            end_time = time.time()
            total_time += end_time - start_time

            # 不增加新的维度
            pred_masks = torch.cat(pred_masks, dim=0)
            pred_ious = torch.cat(pred_ious, dim=0)
            # 二值掩码
            pred_masks = (pred_masks > 0.).float()
            b, num_prompt, per_prompt_mask, h, w = pred_masks.shape
            # [3, B, H, W]
            # [3, B]
            pred_masks = pred_masks.view(b, -1, h, w).permute(1, 0, 2, 3)
            pred_ious = pred_ious.view(b, -1).permute(1, 0)
            best_id = torch.argmax(pred_ious, dim=0)
            # [num_planes, H, W]
            best_mask = pred_masks[best_id, torch.arange(b)]

            # # 将pred与gt匹配
            # target = target.squeeze(1)
            # matching = MatchSegmentation(best_mask, target)
            # matched_pred_indices = []
            # matched_gt_indices = []
            # used = []
            # for i, a in enumerate(matching):
            #     if a not in used:
            #         matched_pred_indices.append(i)
            #         matched_gt_indices.append(a)
            #         used.append(a)
            # matched_pred_indices = torch.as_tensor(matched_pred_indices)
            # matched_gt_indices = torch.as_tensor(matched_gt_indices)
            # prediction = torch.zeros_like(target)
            # prediction[matched_gt_indices] = best_mask[matched_pred_indices]

            # RI, VoI, SC = evaluateMasks(best_mask, target)
            # if SC > 0.7:
            #     continue

            # visual
            prediction = best_mask.cpu().numpy().astype(np.uint8)
            # pred
            rgb_image = map_masks_to_colors(prediction)
            # gt_rgb
            image = image.squeeze(0).cpu().numpy()
            image *= 255
            image = image.astype(np.uint8).transpose(1, 2, 0)
            image = np.clip(image, 0, 255)
            # gt
            target = target.squeeze(1)
            target = target.cpu().numpy().astype(np.uint8)
            gt = map_masks_to_colors(target)
            # depth
            depth = depth.squeeze(0).squeeze(0).cpu().numpy()
            depth = (depth * 255 / (depth.max())).astype(np.uint8)
            depth = cv2.applyColorMap(depth, cv2.COLORMAP_JET)
            # # 拼接
            # rgbd_image = np.concatenate((image, depth, rgb_image, gt), axis=1)
            # rgbd_image = Image.fromarray(rgbd_image)
            # rgbd_image.save("raw_sam_result/all/" + str(batch_idx) + '.jpg')
            # print(f"save {batch_idx}.jpg")

            tifffile.imwrite("mp3d_result/input/input_" + str(batch_idx) + '.tif', image, resolution=(600, 600))
            tifffile.imwrite("mp3d_result/gt/gt_" + str(batch_idx) + '.tif', gt, resolution=(600, 600))
            tifffile.imwrite("mp3d_result/predict/predict_" + str(batch_idx) + '.tif', rgb_image, resolution=(600, 600))
            tifffile.imwrite("mp3d_result/depth/depth_" + str(batch_idx) + '.tif', depth, resolution=(600, 600))
            print(f"save {batch_idx}")

    print("total_time: {}".format(total_time))

if __name__ == '__main__':
    if not os.path.exists(args.output):
        os.mkdir(args.output)
    inference()
