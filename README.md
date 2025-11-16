# Multimodal plane instance segmentation with the Segment Anything Model

This is the official PyTorch implementation for our paper "Multimodal plane instance segmentation with the Segment Anything Model". This paper has been accepted for publication in Automation in Construction.
You may also learn about our algorithm from the preprint version of our paper on arXiv, titled “PlaneSAM: Multimodal Plane Instance Segmentation Using the Segment Anything Model” (https://arxiv.org/abs/2410.16545 ). 
However, the version we published in the journal Automation in Construction is more formal and provides a more complete description of the algorithm.


## 🔭 Introduction
<strong>Abstract:</strong> Plane instance segmentation from RGB-D data is critical for BIM-related tasks. However, existing deep-learning methods rely on only RGB bands, overlooking depth information. To address this, PlaneSAM, a Segment-Anything-Model-based network, is proposed. It fully integrates RGB-D bands using a dual-complexity backbone: a simple branch primarily for the D band and a high-capacity branch mainly for RGB bands. This structure facilitates effective D-band learning with limited data, preserves EfficientSAM's RGB feature representations, and enables task-specific fine-tuning. To improve adaptability to RGB-D domains, a self-supervised pretraining strategy is introduced. EfficientSAM’s loss is also optimized for large-plane segmentation. Additionally, plane detection is performed using Faster R-CNN, enabling fully automatic segmentation. State-of-the-art performance is achieved on multiple datasets, with <10% additional overhead compared to EfficientSAM. The proposed dual-complexity backbone shows strong potential for transferring RGB-based foundation models to RGB+X domains in other scenarios, while the pretraining strategy is promising for other data-scarce tasks.

## 🔭 Citation
If you find our work useful for your research, please consider citing our paper.
Deng, Z., Yang, Z., Chen, C., Zeng, C., Meng, Y., Yang, B., 2025. Multimodal plane instance segmentation with the Segment Anything Model. Automation in Construction 180, 106541.

</p>
