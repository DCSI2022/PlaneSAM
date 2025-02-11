import random
import torch

def make_point_pt(masks, num_samples=1):
    """
    masks: [B, H, W]

    return:
        batched_points: [B, num_pt, per_pt_points, 2]
        batched_label: [B, num_pt, per_pt_points]
    """
    device = masks.device
    batched_points = []
    batched_labels = []

    for mask in masks:
        # instance_ids, counts = torch.unique(mask, return_counts=True)
        index = torch.nonzero(mask, as_tuple=True)
        num_points = len(index[0])
        if num_points >= num_samples:
            # diff = num_samples - num_points if num_samples > num_points else 0
            # num_samples = num_points if num_samples > num_points else num_samples
            select_idx = torch.tensor(random.sample(range(num_points), num_samples))
            # [num_samples, 2]
            point_coords = torch.zeros((num_samples, 2))
            # h -> y and w -> x
            point_coords[..., 1], point_coords[..., 0] = index[0][select_idx], index[1][select_idx]
            point_coords = point_coords[None, ...]
            # 前景是1，背景为0
            point_labels = torch.ones(1, num_samples)

            # 如果diff存在，就要确保batch的格式一致
            # if diff:
            #     diff_point_coords = torch.full((1, diff, 2), fill_value=-1)
            #     diff_point_labels = torch.full((1, diff), fill_value=-1)
            #     point_coords = torch.cat((point_coords, diff_point_coords), dim=1)
            #     point_labels = torch.cat((point_labels, diff_point_labels), dim=1)

        # 没有实例
        else:
            # 点提示全部用-1填充
            point_coords = torch.full((1, num_samples, 2), fill_value=-1)
            point_labels = torch.full((1, num_samples), fill_value=-1)

        batched_points.append(point_coords)
        batched_labels.append(point_labels)

    batched_points = torch.stack(batched_points, dim=0).to(device)
    batched_labels = torch.stack(batched_labels, dim=0).to(device)

    return batched_points, batched_labels


def make_box_pt(batched_masks, noise_ratio=0.):
    """
    batched_mask: [B, H, W]

    return:
        左上点的标签是2，右下点的标签是3
        batched_points: [B, num_pt, per_pt_points, 2]
        batched_label: [B, num_pt, per_pt_points]
    """
    device = batched_masks.device

    batched_points = []
    batched_labels = []

    for per_mask in batched_masks:
        indices = torch.nonzero(per_mask)
        h, w = per_mask.shape

        # 存在前景
        if indices.numel():
            # 计算边界框的左下和右上点的坐标
            # bbox_min:[x_min, y_min]
            bbox_min = torch.min(indices, dim=0).values.flip(0)
            bbox_max = torch.max(indices, dim=0).values.flip(0)

            # 加10%边界框长度的噪声
            bbox_w, bbox_h = bbox_max - bbox_min
            noise_x1 = int(random.uniform(-noise_ratio, noise_ratio) * bbox_w)
            noise_y1 = int(random.uniform(-noise_ratio, noise_ratio) * bbox_h)
            noise_x2 = int(random.uniform(-noise_ratio, noise_ratio) * bbox_w)
            noise_y2 = int(random.uniform(-noise_ratio, noise_ratio) * bbox_h)
            bbox_min[0] = bbox_min[0] + noise_x1
            bbox_min[1] = bbox_min[1] + noise_y1
            bbox_max[0] = bbox_max[0] + noise_x2
            bbox_max[1] = bbox_max[1] + noise_y2
            bbox_min[0] = torch.clip(bbox_min[0], 0, w)
            bbox_min[1] = torch.clip(bbox_min[1], 0, h)
            bbox_max[0] = torch.clip(bbox_max[0], 0, w)
            bbox_max[1] = torch.clip(bbox_max[1], 0, h)

            # [num_pt, 2, 2]
            per_box_pt = torch.cat((bbox_min[None, None, :], bbox_max[None, None, :]), dim=1)

            # 左上对应最小点，右下对应最大点
            bottomright_label = torch.full((1, 1), fill_value=3).to(device)
            topleft_label = torch.full((1, 1), fill_value=2).to(device)
            # [1, 2, 1]
            per_box_label = torch.cat((topleft_label, bottomright_label), dim=1)

            batched_points.append(per_box_pt)
            batched_labels.append(per_box_label)
        else:
            batched_points.append(torch.full((1, 2, 2), fill_value=-1).to(device))
            batched_labels.append(torch.full((1, 2), fill_value=-1).to(device))

    batched_points = torch.stack(batched_points, dim=0)
    batched_labels = torch.stack(batched_labels, dim=0)

    return batched_points, batched_labels


def preprocess(masks, num_points=1, box=False, noise_ratio=0.):
    """
    masks: [B, H, W]

    return:
    point_prompts: [B, num_pts, num_points, 2]
    prompt_labels: [B, num_pts, num_points]
    """
    assert 0 <= num_points + 2 * box <= 6, "num_points shouldn't be greater than 6 or less than 0!"

    device = masks.device

    batched_points = None
    batched_pt_labels = None
    batched_boxes = None
    batched_box_labels = None

    if num_points:
        batched_points, batched_pt_labels = make_point_pt(masks, num_samples=num_points)
    if box:
        batched_boxes, batched_box_labels = make_box_pt(masks, noise_ratio)

    # 分配到指定设备上
    if batched_points is not None:
        batched_points = batched_points.to(device)
        batched_pt_labels = batched_pt_labels.to(device)
    if batched_boxes is not None:
        batched_boxes = batched_boxes.to(device)
        batched_box_labels = batched_box_labels.to(device)

    # 分情况返回值
    if batched_points is not None and batched_boxes is not None:
        # [B, num_pts, num_points, 2]
        # [B, num_pts, num_points]
        return torch.cat((batched_boxes, batched_points), dim=2), torch.cat(
            (batched_box_labels, batched_pt_labels), dim=2)
    elif batched_points is not None:
        return batched_points, batched_pt_labels
    elif batched_boxes is not None:
        return batched_boxes, batched_box_labels