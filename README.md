# Multimodal plane instance segmentation with the Segment Anything Model

## Getting Start

Build the Pytorch Environment:

```bash
conda create -n PlaneSAM python=3.9.16
conda activate PlaneSAM
pip install -r requirements.txt
```

## Data Preparation

We train and test our network using the same plane dataset as [PlaneTR](https://github.com/IceTTTb/PlaneTR3D).
You can access the dataset from [here](https://pan.baidu.com/s/1-1w5dFULfLbxVrfUd6Gmig?pwd=cyye).

## Training

Our training process consists of two steps: 

- First, we pretrain on a large-scale RGB-D dataset. The pretrained weights can be obtained from [here](https://pan.baidu.com/s/1Dw4mrCJliEXC6ZT4BOdsuA?pwd=chnk) and placed in the weights directory. 
- Second, load the pretrained weights into the network and run the train.py script.The trained weights can be obtained from [here](https://pan.baidu.com/s/1jZcYI9YbD4K9B6HZqLfJ3Q?pwd=1c6p). 

## Evaluation

During the evaluation, we use Faster R-CNN as the plane detector. The trained weights can be obtained from [here](https://pan.baidu.com/s/1sHXdjAry2RIc1xHBQXaB7w?pwd=p1kt) and placed in the weights directory. The unseen test dataset can be obtained from [here](https://pan.baidu.com/s/1BIwpigGtfPMxhAmj3M6E6Q?pwd=t6e2).
To evaluate the plane segmentation capabilities of PlaneSAM, please run the eval.py script.

## Acknowledgements

This code is based on the [EfficientSAM](https://github.com/yformer/EfficientSAM) repository. We would like to acknowledge the authors for their work.

## Additional Note

Due to the author's current busy schedule, we apologize for the possibly poor code quality. Optimizations will be made in the future. If you encounter any questions or bugs in the code, feel free to ask.
