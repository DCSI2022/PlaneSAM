import os
import argparse
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from PlaneDataset import PlaneDataset
from model import build_efficient_sam_vitt
from utils.train_tools import save_ckpt, load_ckpt
from utils.eval_tools import compute_iou
from utils.loss import criterion
from utils.make_prompt import preprocess
from torch.optim.lr_scheduler import CosineAnnealingLR
from random import randint


parser = argparse.ArgumentParser(description='Segment Any Planes')
parser.add_argument('--data-dir', default='ScanNet', metavar='DIR',
                    help='path to dataset-D')
parser.add_argument('--cuda', action='store_true', default=True,
                    help='enables CUDA training')
parser.add_argument('--num-workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=15, type=int, metavar='N',
                    help='number of total epochs to run (default: 150)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=8, type=int,
                    metavar='N', help='mini-batch size (default: 10)')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--weight-decay', '--wd', default=0.01, type=float,
                    metavar='W', help='weight decay')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--save-epoch-freq', '-s', default=1, type=int,
                    metavar='N', help='save epoch frequency (default: 5)')
parser.add_argument('--last-ckpt', default='weights/pre.pth', type=str, metavar='PATH')
parser.add_argument('--lr-decay-rate', default=0.8, type=float,
                    help='decay rate of learning rate (default: 0.8)')
parser.add_argument('--lr-epoch-per-decay', default=10, type=int,
                    help='epoch of per decay of learning rate (default: 10)')
parser.add_argument('--ckpt-dir', default='weights', metavar='DIR',
                    help='path to save checkpoints')

args = parser.parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

image_w = 256
image_h = 192

def train():
    train_data = PlaneDataset(subset="train",
                              transform=transforms.Compose([
                                  transforms.ToTensor()]),
                              root_dir=args.data_dir
                              )
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                              pin_memory=False)
    val_data = PlaneDataset(subset="val",
                            transform=transforms.Compose([
                                transforms.ToTensor()]),
                            root_dir=args.data_dir
                            )
    val_loader = DataLoader(val_data, batch_size=1, shuffle=True, num_workers=args.num_workers,
                            pin_memory=False)

    num_train = len(train_data)

    model = build_efficient_sam_vitt()

    for p in model.prompt_encoder.parameters():
        p.requires_grad = False

    if torch.cuda.device_count() >= 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    model.to(device)

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr,
                                  weight_decay=args.weight_decay)

    global_step = 0

    if args.last_ckpt:
        global_step, _ = load_ckpt(model, None, args.last_ckpt, device, NUM_GPUS=2)

    # 余弦学习率调度
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0)

    for _ in range(args.start_epoch):
        optimizer.step()
        scheduler.step()

    for epoch in range(int(args.start_epoch), args.epochs):

        local_count = 0
        if epoch % args.save_epoch_freq == 0 and epoch != args.start_epoch:
            save_ckpt(args.ckpt_dir, model, optimizer, global_step, epoch,
                      local_count, num_train)

        # 训练
        model.train()
        num_batch = 0
        mean_loss = 0
        for sample in tqdm(train_loader, desc=f"Train Epoch [{epoch + 1}/{args.epochs}]"):
            image = sample['image'].to(device)  # [B, C, H, W]
            num_planes = sample['num_planes']
            sel_planes = [randint(0, x - 1) for x in num_planes]
            target = sample['instance'].to(device)  # [B, 21, H, W]，第0-num_planes-1为平面掩码，第num_planes为背景(非平面)
            target = target[torch.arange(len(num_planes)), sel_planes, ...]
            depth = sample['depth'].to(device)  # [B, 1, H, W]

            optimizer.zero_grad()

            input_points, input_labels = preprocess(target, num_points=0, box=True)

            # [B, num_prompt, per_prompt_mask, H, W]
            pred_masks, pred_ious = model(image, depth, input_points, input_labels)
            # 训练只给一个提示
            b, num_prompt, per_prompt_mask, h, w = pred_masks.shape
            #
            pred_masks = pred_masks.view(b, -1, h, w)
            pred_ious = pred_ious.view(b, -1)
            # 遍历3个掩码
            pred_masks = pred_masks.permute(1, 0, 2, 3)
            pred_ious = pred_ious.permute(1, 0)

            loss = []
            for mask, iou in zip(pred_masks, pred_ious):
                loss.append(criterion(mask, iou, target))
            loss = min(loss)

            num_batch += 1
            mean_loss += loss

            loss.backward()
            optimizer.step()

            local_count += image.data.shape[0]
            global_step += 1

        scheduler.step()

        mean_loss /= num_batch
        print('Epoch: {}    mean_loss: {}'.format(epoch + 1, mean_loss))

        # 评估(use box to test)
        model.eval()
        totoal_iou = .0
        totoal_num = 0
        with torch.no_grad():
            for sample in tqdm(val_loader, desc=f"Eval Epoch[{epoch + 1}/{args.epochs}]"):
                image = sample['image'].to(device)
                num_planes = sample['num_planes'][0]
                target = sample['instance'].to(device)
                target = target[:, randint(0, num_planes - 1), ...]
                depth = sample['depth'].to(device)

                # 预处理
                input_points, input_labels = preprocess(target, num_points=0, box=True, noise_ratio=0.1)

                # [B, num_prompt, per_prompt_mask, H, W]
                pred_masks, pred_ious = model(image, depth, input_points, input_labels)
                # 二值掩码
                pred_masks = (pred_masks > 0.).float()
                b, num_prompt, per_prompt_mask, h, w = pred_masks.shape
                # [num_masks, B, H, W]
                pred_masks = pred_masks.view(b, -1, h, w).permute(1, 0, 2, 3)
                pred_ious = pred_ious.view(b, -1).permute(1, 0)

                # 计算iou
                best_id = torch.argmax(pred_ious, dim=0)
                best_mask = pred_masks[best_id, torch.arange(b)]
                best_iou = compute_iou(best_mask, target)
                totoal_num += len(best_iou)
                totoal_iou += sum(best_iou)

            mIou = float(totoal_iou / totoal_num)
            print('Epoch: {}    mIou: {:.2}\n'.format(epoch + 1, mIou))
            with open('log/logger.txt', 'a') as f:
                f.write('Epoch: {}    mIou: {:.4}   mean_loss: {}\n'.format(epoch + 1, mIou, mean_loss))

    save_ckpt(args.ckpt_dir, model, optimizer, global_step, args.epochs,
              0, num_train)

    print("Training completed ")


if __name__ == '__main__':

    if not os.path.exists(args.ckpt_dir):
        os.mkdir(args.ckpt_dir)

    train()
