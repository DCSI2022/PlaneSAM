import os
import torch
from FasterRCNN import FasterRCNN, FastRCNNPredictor
from backbone import resnet101_fpn_backbone

def create_model(num_classes, load_pretrain_weights=False):
    backbone = resnet101_fpn_backbone(pretrain_path="",
                                     norm_layer=torch.nn.BatchNorm2d,
                                     trainable_layers=5)
    # 训练自己数据集时不要修改这里的91，修改的是传入的num_classes参数
    model = FasterRCNN(backbone=backbone, num_classes=91)

    if load_pretrain_weights:
        weights_dict = torch.load("./backbone/fasterrcnn_resnet50_fpn_coco.pth", map_location='cpu')
        missing_keys, unexpected_keys = model.load_state_dict(weights_dict, strict=False)
        if len(missing_keys) != 0 or len(unexpected_keys) != 0:
            print("missing_keys: ", missing_keys)
            print("unexpected_keys: ", unexpected_keys)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model