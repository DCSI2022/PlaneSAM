import torch
import torch.nn as nn
import torch.nn.functional as F
import utils.box_ops as box_ops


# 输入[B, H, W]每一个像素是一个整数表示一个类
# 输出[B, C, H, W]共C个类，每一个类对应一个二值掩码
def build_target(target, num_classes=2, ignore_index=-100):
    dice_target = target.clone()
    if ignore_index >= 0:
        ignore_mask = torch.eq(target, ignore_index)
        dice_target[ignore_mask] = 0
        # [B, H, W] -> [B, H, W, C]
        dice_target = nn.functional.one_hot(dice_target, num_classes).float()
        dice_target[ignore_mask] = ignore_index
    else:
        dice_target = nn.functional.one_hot(dice_target, num_classes).float()

    return dice_target.permute(0, 3, 1, 2)


# 输入[B, H, W]
def dice_coeff(x, target, ignore_index=-100, epsilon=1e-6):
    d = 0.
    batch_size = x.shape[0]
    for i in range(batch_size):
        x_i = x[i].reshape(-1)
        t_i = target[i].reshape(-1)
        if ignore_index >= 0:
            # 找出像素值不为ignore_index的位置
            roi_mask = torch.ne(t_i, ignore_index)
            x_i = x_i[roi_mask]
            t_i = t_i[roi_mask]
        inter = torch.dot(x_i, t_i)
        sets_sum = torch.sum(x_i) + torch.sum(t_i)
        # 如果sets_sum为0，说明预测和实际值都为0，预测百分百正确，dice系数为1
        if sets_sum == 0:
            sets_sum = 2 * inter

        d += (2 * inter + epsilon) / (sets_sum + epsilon)

    return d / batch_size


def multiclass_dice_coeff(x, target, ignore_index=-100, epsilon=1e-6):
    dice = 0.
    for channel in range(x.shape[1]):
        dice += dice_coeff(x[:, channel, ...], target[:, channel, ...], ignore_index, epsilon)


def dice_loss(x, target, multiclass=False, ignore_index=-100):
    x = torch.sigmoid(x)
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(x, target, ignore_index=ignore_index)


def criterion(x_mask, x_iou, target, alpha=1, gamma=2):
    """
    :param x_mask: [B, H, W]SAM输出的掩码
    :param x_iou: [B,]掩码对应的预测iou
    :param target: [B, H, W]标签
    :param alpha: focalloss参数
    :param gamma: focalloss参数
    :return: 返回综合损失
    """
    batch_size, h, w = x_mask.shape
    binary_mask = (x_mask > 0).float()

    ce_loss = F.binary_cross_entropy_with_logits(x_mask, target, reduction='mean')
    pt = torch.exp(-ce_loss)
    focal_loss = alpha * (1 - pt) ** gamma * ce_loss

    Dice_loss = dice_loss(x_mask, target)

    binary_mask = binary_mask.reshape(batch_size, -1)
    target = target.reshape(batch_size, -1)
    intersection = torch.sum(torch.logical_and(binary_mask, target).float(), dim=1)
    union = torch.sum(torch.logical_or(binary_mask, target).float(), dim=1)
    iou = intersection / (union + 1e-6)
    binary_mask = binary_mask.reshape(batch_size, h, w)
    target = target.reshape(batch_size, h, w)

    mse_Loss = F.mse_loss(x_iou, iou, reduction='mean')

    return 20 * focal_loss + Dice_loss