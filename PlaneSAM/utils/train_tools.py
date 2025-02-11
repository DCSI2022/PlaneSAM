import numpy as np
import torch
import os
from sklearn.metrics import adjusted_rand_score


def print_log(global_step, epoch, local_count, count_inter, dataset_size, loss, time_inter):
    print('Step: {:>5} Train Epoch: {:>3} [{:>4}/{:>4} ({:3.1f}%)]    '
          'Loss: {:.6f} [{:.2f}s every {:>4} data]'.format(
        global_step, epoch, local_count, dataset_size,
        100. * local_count / dataset_size, loss.data, time_inter, count_inter))


def save_ckpt(ckpt_dir, model, optimizer, global_step, epoch, local_count, num_train):
    # usually this happens only on the start of a epoch
    epoch_float = epoch + (local_count / num_train)
    state = {
        'global_step': global_step,
        'epoch': epoch_float,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    ckpt_model_filename = "ckpt_epoch_{:0.2f}.pth".format(epoch_float)
    path = os.path.join(ckpt_dir, ckpt_model_filename)
    torch.save(state, path)
    print('{:>2} has been successfully saved'.format(path))


def load_ckpt(model, optimizer, model_file, device, NUM_GPUS):
    if os.path.isfile(model_file):
        print("=> loading checkpoint '{}'".format(model_file))
        if device.type == 'cuda':
            checkpoint = torch.load(model_file)
        else:
            checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
        model_dict = checkpoint['state_dict']
        model_dict_ = {}
        if NUM_GPUS > 1:
            model.load_state_dict(model_dict)
        else:
            for k, v in model_dict.items():
                k_ = k.replace('module.', '')
                model_dict_[k_] = v
            model.load_state_dict(model_dict_)
        if optimizer:
            optimizer.load_state_dict(checkpoint['optimizer'])
    else:
        print("=> no checkpoint found at '{}'".format(model_file))
        exit(0)


# 计算指标
def compute_iou(x, target):
    """
    :param x: [B, H, W]
    :param target: [B, H, W]
    :return: iou
    """
    b, h, w = x.shape
    x = x.reshape(b, -1)
    target = target.reshape(b, -1)
    intersection = torch.sum(torch.logical_and(x, target).float(), dim=1)
    union = torch.sum(torch.logical_or(x, target).float(), dim=1)
    # [B]
    iou = intersection / (union + 1e-9)
    return iou


def compute_iou_sc(x, target):
    """
    :param x: [B, H, W]
    :param target: [B, H, W]
    :return: iou
    """
    b, h, w = x.shape
    x = x.reshape(b, -1)
    target = target.reshape(b, -1)
    intersection = np.sum(np.logical_and(x, target).astype(np.float32), axis=1)
    union = np.sum(np.logical_or(x, target).astype(np.float32), axis=1)
    iou = np.max(intersection / (union + 1e-9))
    return iou


def compute_acc(x, target):
    """
    :param x: [B, H, W]
    :param target: [B, H, W]
    :return: acc
    """
    b, h, w = x.shape
    x = x.reshape(b, -1)
    target = target.reshape(b, -1)

    acc = torch.mean((x == target).float(), dim=1)
    return acc


def compute_RI(x, target):
    """
    :param x: [B, H, W]
    :param target: [B, H, W]
    :return: RI
    """
    num_planes, h, w = x.shape
    pred_map = np.zeros((h, w), dtype=np.uint8)
    gt_map = np.zeros((h, w), dtype=np.uint8)

    # [H, W]
    for i in range(num_planes):
        pred_map[x[i] == 1] = i + 1
        gt_map[target[i] == 1] = i + 1

    pred_map = pred_map.flatten()
    gt_map = gt_map.flatten()
    RI = adjusted_rand_score(pred_map, gt_map)

    return RI


def compute_SC(x, target):
    """
    :param x: [B, H, W]
    :param target: [B, H, W]
    :return: RI
    """
    iou_1 = []
    iou_2 = []
    num_planes, h, w = x.shape
    for per_plane in x:
        plane = np.repeat(per_plane[np.newaxis, ...], num_planes, axis=0)
        iou_1.append(compute_iou_sc(plane, target))
    iou_1 = sum(iou_1) / len(iou_1)

    for per_plane in target:
        plane = np.repeat(per_plane[np.newaxis, ...], num_planes, axis=0)
        iou_2.append(compute_iou_sc(plane, x))
    iou_2 = sum(iou_2) / len(iou_2)

    return (iou_1 + iou_2) / 2