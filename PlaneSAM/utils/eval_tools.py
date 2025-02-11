import numpy as np
import torch
from sklearn.metrics import adjusted_rand_score, mutual_info_score
from torchvision.ops import box_iou


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


def masks_to_map(masks):
    """
    :param masks: [B, H, W]
    :return: [H, W]
    """
    num_planes, h, w = masks.shape
    semantic_map = np.zeros((h, w), dtype=np.uint8)
    # [H, W]
    for i in range(num_planes):
        semantic_map[masks[i] == 1] = i + 1

    return semantic_map


def match_boxes_gt(pred_boxes, gt_boxes):
    # pred_boxes has been sorted by score
    num_gts = gt_boxes.shape[0]
    num_preds = pred_boxes.shape[0]
    device = gt_boxes.device
    batched_points = torch.full((num_gts, 2, 2), fill_value=-1, dtype=torch.float32, device=device)
    batched_labels = torch.full((num_gts, 2), fill_value=-1, dtype=torch.float32, device=device)
    iou_matrix = box_iou(pred_boxes, gt_boxes)

    _, best_match_indices = iou_matrix.max(dim=1)
    unique_masks = torch.zeros(num_preds, dtype=torch.bool, device=device)
    tmp = []
    for i, t in enumerate(best_match_indices):
        if t not in tmp:
            tmp.append(t)
            unique_masks[i] = True

    pred_boxes = pred_boxes[unique_masks]
    best_match_indices = best_match_indices[unique_masks]

    batched_points[best_match_indices, 0] = pred_boxes[:, :2]
    batched_points[best_match_indices, 1] = pred_boxes[:, 2:]
    batched_labels[best_match_indices, 0] = 2
    batched_labels[best_match_indices, 1] = 3

    return batched_points.unsqueeze(1), batched_labels.unsqueeze(1)


def evaluateMasks(predMasks, gtMasks):
    """
    :param predMasks: [N, H, W]
    :param gtMasks: [N, H, W]
    :return:
    """
    valid_mask = (gtMasks.max(0)[0]).unsqueeze(0)

    gtMasks = torch.cat([gtMasks, torch.clamp(1 - gtMasks.sum(0, keepdim=True), min=0)], dim=0)  # M+1, H, W
    predMasks = torch.cat([predMasks, torch.clamp(1 - predMasks.sum(0, keepdim=True), min=0)], dim=0)  # N+1, H, W

    intersection = (gtMasks.unsqueeze(1) * predMasks * valid_mask).sum(-1).sum(-1).float()
    union = (torch.max(gtMasks.unsqueeze(1), predMasks) * valid_mask).sum(-1).sum(-1).float()

    N = intersection.sum()

    RI = 1 - ((intersection.sum(0).pow(2).sum() + intersection.sum(1).pow(2).sum()) / 2 - intersection.pow(2).sum()) / (
            N * (N - 1) / 2)
    joint = intersection / N
    marginal_2 = joint.sum(0)
    marginal_1 = joint.sum(1)
    H_1 = (-marginal_1 * torch.log2(marginal_1 + (marginal_1 == 0).float())).sum()
    H_2 = (-marginal_2 * torch.log2(marginal_2 + (marginal_2 == 0).float())).sum()

    B = (marginal_1.unsqueeze(-1) * marginal_2)
    log2_quotient = torch.log2(torch.clamp(joint, 1e-8) / torch.clamp(B, 1e-8)) * (torch.min(joint, B) > 1e-8).float()
    MI = (joint * log2_quotient).sum()
    voi = H_1 + H_2 - 2 * MI

    IOU = intersection / torch.clamp(union, min=1)
    SC = ((IOU.max(-1)[0] * torch.clamp((gtMasks * valid_mask).sum(-1).sum(-1), min=1e-4)).sum() / N + (
            IOU.max(0)[0] * torch.clamp((predMasks * valid_mask).sum(-1).sum(-1), min=1e-4)).sum() / N) / 2
    info = [RI.item(), voi.item(), SC.item()]

    return info