# PlaneSAM: Multimodal Plane Instance Segmentation Using the Segment Anything Model

This is the official PyTorch implementation for our paper "PlaneSAM: Multimodal Plane Instance Segmentation Using the Segment Anything Model". The paper has been uploaded to Arxiv (https://arxiv.org/abs/2410.16545) and is currently under review by the journal Automation in Construction.


## 🔭 Introduction
<strong>Abstract:</strong> Plane instance segmentation from RGB-D data is a crucial research topic for
many downstream tasks, such as indoor 3D reconstruction. However, most existing deep-learning-based methods utilize only information within the RGB bands,
neglecting the important role of the depth band in plane instance segmentation.
Based on EfficientSAM, a fast version of the Segment Anything Model (SAM),
we propose a plane instance segmentation network called PlaneSAM, which can
fully integrate the information of the RGB bands (spectral bands) and the D band
(geometric band), thereby improving the effectiveness of plane instance segmentation in a multimodal manner. Specifically, we use a dual-complexity backbone,
with primarily the simpler branch learning D-band features and primarily the
more complex branch learning RGB-band features. Consequently, the backbone
can effectively learn D-band feature representations even when D-band training
data is limited in scale, retain the powerful RGB-band feature representations of
EfficientSAM, and allow the original backbone branch to be fine-tuned for the current task. To enhance the adaptability of our PlaneSAM to the RGB-D domain,
we pretrain our dual-complexity backbone using the segment anything task on
large-scale RGB-D data through a self-supervised pretraining strategy based on
imperfect pseudo-labels. To support the segmentation of large planes, we optimize
the loss function combination ratio of EfficientSAM. In addition, Faster R-CNN is
used as a plane detector, and its predicted bounding boxes are fed into our dualcomplexity network as prompts, thereby enabling fully automatic plane instance
segmentation. Experimental results show that the proposed PlaneSAM sets a new
state-of-the-art (SOTA) performance on the ScanNet dataset, and outperforms
previous SOTA approaches in zero-shot transfer on the 2D-3D-S, Matterport3D,and ICL-NUIM RGB-D datasets, while only incurring a 10% increase in computational overhead compared to EfficientSAM. Our code and trained model will be
released publicly.

</p>
