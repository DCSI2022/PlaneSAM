import numpy as np
import cv2

colors = np.array([
    [255, 0, 0],
    [0, 255, 0],
    [0, 0, 255],
    [255, 255, 0],
    [255, 0, 255],
    [0, 255, 255],
    [255, 128, 0],
    [128, 0, 255],
    [0, 128, 255],
    [255, 0, 128],
    [128, 255, 0],
    [0, 255, 128],
    [255, 128, 128],
    [128, 255, 128],
    [128, 128, 255],
    [255, 255, 128],
    [255, 128, 255],
    [128, 255, 255],
    [192, 192, 192],
    [128, 0, 0],
    [0, 128, 0],
    [0, 0, 128],
    [128, 128, 0],
    [128, 0, 128],
    [0, 128, 128],
    [64, 0, 0],
    [0, 64, 0],
    [0, 0, 64],
    [64, 64, 0],
    [64, 0, 64],
    [0,64, 64],
    [0, 0, 0]
], dtype=np.uint8)

def map_masks_to_colors(masks):
    """
    :param masks: [num_planes, H, W]
    :return: rgb_image[H, W, 3]
    """
    num_masks = len(masks)
    # 随机生成类别对应的颜色

    # 创建空白RGB图像
    height, width = masks[0].shape
    rgb_image = np.zeros((height, width, 3), dtype=np.uint8)

    for i, mask in enumerate(masks):
        color = colors[i]
        # mask_rgb = np.zeros((height, width, 3), dtype=np.uint8)
        # mask_rgb[mask > 0] = color
        # rgb_image += mask_rgb
        rgb_image[mask > 0] = color

    return rgb_image


def draw_bounding_boxes(image, boxes):
    """
    在图像上绘制边界框 (使用 OpenCV)。

    :param image: RGB图像 (ndarray, shape [H, W, 3])。
    :param boxes: 边界框数组 (ndarray, shape [B, 4], 每行 [xmin, ymin, xmax, ymax])。
    :return: 带有边界框的图像。
    """
    image_copy = image.copy()
    for box in boxes:
        xmin, ymin, xmax, ymax = map(int, box)
        cv2.rectangle(image_copy, (xmin, ymin), (xmax, ymax), color=(0, 255, 0), thickness=2)  # 绿色边框
    return image_copy