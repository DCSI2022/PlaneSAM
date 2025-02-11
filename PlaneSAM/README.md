# PlaneSAM: Multimodal Plane Instance Segmentation Using the Segment Anything Model



## Getting Start
Build the Pytorch Environment:
```bash
conda create -n PlaneSAM python=3.9.16
conda activate PlaneSAM
pip install -r requirements.txt
```

## Data Preparation

We train and test our network using the same plane dataset as [PlaneTR](https://github.com/IceTTTb/PlaneTR3D).
You can access the dataset from [here](https://pan.baidu.com/s/1pyx-Ou3SLq7XG5NIMqC2cQ?pwd=in3b)

## Training

Our training process consists of two steps: </br>
 - First, we pretrain on a large-scale RGB-D dataset. The pretrained weights can be obtained from [here](https://pan.baidu.com/s/1NarX09MpkDDsBr7WWI0mzw?pwd=pvkj) and placed in the weights directory. </br>
 - Second, load the pretrained weights into the network and run the train.py script.The trained weights can be obtained from [here](https://pan.baidu.com/s/1O4zygzKL13obNMAB2kuxkg?pwd=nrwj). </br>

## Evaluation

During the evaluation, we use Faster R-CNN as the plane detector. The trained weights can be obtained from [here](https://pan.baidu.com/s/1uO1pqs2B4R5IPKQgU0fPTg?pwd=26jr) and placed in the weights directory.The unseen test dataset can be obtained from [here](https://pan.baidu.com/s/1ywNjTRCzXfuxb2VHPGTGzg?pwd=9qcm).</br>
To evaluate the plane segmentation capabilities of PlaneSAM, please run the eval.py script.

## Acknowledgements
This code is based on the [EfficientSAM](https://github.com/yformer/EfficientSAM) repository. We would like to acknowledge the authors for their work.

## Additional Note
Due to the author's current busy schedule, we apologize for the possibly poor code quality. Optimizations will be made in the future. If you encounter any questions or bugs in the code, feel free to ask.