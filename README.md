# PlaneSAM: Multimodal Plane Instance Segmentation Using the Segment Anything Model

This is the official PyTorch implementation for our paper "PlaneSAM: Multimodal Plane Instance Segmentation Using the Segment Anything Model". The paper has been uploaded to Arxiv (https://arxiv.org/abs/2410.16545) and is currently under review by the journal Automation in Construction.


## 🔭 Introduction
<strong>Abstract:</strong> Plane instance segmentation from RGB-D data is a crucial research topic for
many downstream tasks, such as indoor 3D reconstruction. However, most existing deep-learning-based methods utilize only information within the RGB bands,
neglecting the important role of the depth band in plane instance segmentation.
Based on EfficientSAM, a fast version of the Segment Anything Model (SAM),
we propose a plane instance segmentation network called PlaneSAM, which can
fully integrate the information of the RGB bands (spectral bands) and the D band
(geometric band), thereby improving the effectiveness of plane instance segmentation in a multimodal manner. Specifically, we use a dual-complexity backbone,
with primarily the simpler branch learning D-band features and primarily the
more complex branch learning RGB-band features. Consequently, the backbone
can effectively learn D-band feature representations even when D-band training
data is limited in scale, retain the powerful RGB-band feature representations of
EfficientSAM, and allow the original backbone branch to be fine-tuned for the current task. To enhance the adaptability of our PlaneSAM to the RGB-D domain,
we pretrain our dual-complexity backbone using the segment anything task on
large-scale RGB-D data through a self-supervised pretraining strategy based on
imperfect pseudo-labels. To support the segmentation of large planes, we optimize
the loss function combination ratio of EfficientSAM. In addition, Faster R-CNN is
used as a plane detector, and its predicted bounding boxes are fed into our dualcomplexity network as prompts, thereby enabling fully automatic plane instance
segmentation. Experimental results show that the proposed PlaneSAM sets a new
state-of-the-art (SOTA) performance on the ScanNet dataset, and outperforms
previous SOTA approaches in zero-shot transfer on the 2D-3D-S, Matterport3D,and ICL-NUIM RGB-D datasets, while only incurring a 10% increase in computational overhead compared to EfficientSAM. Our code and trained model will be
released publicly.

</p>

## 🔗 Related Works
<strong>Dataset:</strong>

[<u>WHU-Helmet Dataset</u>](https://github.com/kafeiyin00/WHU-HelmetDataset): A helmet-based multi-sensor SLAM dataset for the evaluation of real-time 3D mapping in large-scale GNSS-denied environments

## 💻 Requirements
The code has been tested on:
- Ubuntu 18.04
- ROS melodic
- GTSAM 4.0.3
- Ceres 2.1.0

## ✏️ Build & Run
### 1. How to build this project

```bash
cd ~/catkin_ws/src
git clone https://github.com/DCSI2022/AFLI_Calib.git
cd AFLI_Calib
catkin_make
```
Need solve the dependency before catkin_make, or use Docker

### Docker (Recommended)

```
docker build -t $image_name:tag . #build custom name and tag from Dockerfile
docker run -it -v ~/catkin_ws/src/AFLI_Calib:/home/catkin_ws/src/AFLI_Calib --network host $image_name:tag
cd /home/catkin_ws # in container
catkin_make # in container 
```

### RUN AFLO
  ```
  rosrun afli_calib afl_lidarOdometry $rosbag_path $lidar_type $lidar_topic $match_stability_threshold $motion_linearity_threshold $rosbag_start $rosbag_end
  ```
  
lidar_type: 1 LIVOX 2 VELODYNE 3 OUSTER 4 HESAI

### RUN LiDAR-IMU extrinsic calibration
  ```
  rosrun afli_calib tight_licalib $rosbag_path $lo_path $lidar_type $lidar_topic $rosbag_start $rosbag_end %still_time
  ```
## Todo
- [ ] Modify parameters using yaml file

## 💡 Citation
If you find this repo helpful, please give us a star .
Please consider citing AFLI-Calib if this program benefits your project
```
@article{wu2023afli,
  title={AFLI-Calib: Robust LiDAR-IMU extrinsic self-calibration based on adaptive frame length LiDAR odometry},
  author={Wu, Weitong and Li, Jianping and Chen, Chi and Yang, Bisheng and Zou, Xianghong and Yang, Yandi and Xu, Yuhang and Zhong, Ruofei and Chen, Ruibo},
  journal={ISPRS Journal of Photogrammetry and Remote Sensing},
  volume={199},
  pages={157--181},
  year={2023},
  publisher={Elsevier}
}
```

## 🔗 Acknowledgments
We sincerely thank the excellent projects:
- [loam_livox](https://github.com/hku-mars/loam_livox) for inspiring the idea of the adaptive frame length
- [ikd-Tree](https://github.com/hku-mars/ikd-Tree) for point cloud map management;
- [GTSAM](https://github.com/borglab/gtsam) for IMU pre-integration and factor graph optimization;
- [Ceres](https://github.com/ceres-solver/ceres-solver) for auto-diff.
- [Sopuhs](https://github.com/strasdat/Sophus)
- [PCL](https://github.com/PointCloudLibrary/pcl)
- [A-LOAM](https://github.com/HKUST-Aerial-Robotics/A-LOAM)
