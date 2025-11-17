# 🔬 Awesome U-Net Instance Segmentation

<div align="center">

![U-Net Banner](https://img.shields.io/badge/U--Net-Instance%20Segmentation-blue?style=for-the-badge&logo=tensorflow)
[![Awesome](https://awesome.re/badge-flat2.svg)](https://awesome.re)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=flat-square)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](CONTRIBUTING.md)
![Last Updated](https://img.shields.io/badge/Last%20Updated-2025-orange?style=flat-square)

**A comprehensive collection of U-Net architectures, papers, implementations, and resources for instance segmentation**

[🚀 Getting Started](#-quick-start) | [📚 Papers](#-state-of-the-art-papers) | [💻 Code](#-implementations) | [🗃️ Datasets](#-datasets) | [🎯 Tutorials](#-tutorials)

</div>

---

## 📋 Table of Contents

- [🌟 Introduction](#-introduction)
- [🏗️ U-Net Architecture Overview](#️-u-net-architecture-overview)
- [📊 U-Net Variants](#-u-net-variants)
- [📚 State-of-the-Art Papers](#-state-of-the-art-papers)
- [💻 Implementations](#-implementations)
- [🛠️ Tools & Frameworks](#️-tools--frameworks)
- [🗃️ Datasets](#️-datasets)
- [🎓 Pretrained Models](#-pretrained-models)
- [🏥 Medical Imaging Applications](#-medical-imaging-applications)
- [⚡ Real-Time & Efficient Models](#-real-time--efficient-models)
- [🎯 **U-Net for Landmark Detection**](#-u-net-for-landmark-detection)
- [🔮 Future Trends](#-future-trends)
- [🎯 Tutorials](#-tutorials)
- [📖 Citation](#-citation)

---

## 🌟 Introduction

**Instance segmentation** is a computer vision task that combines object detection and semantic segmentation, identifying each object instance with pixel-level precision. While U-Net was originally designed for semantic segmentation, recent advances have adapted it for instance segmentation with remarkable results.

### Why U-Net for Instance Segmentation?

✅ **Excellent spatial preservation** through skip connections
✅ **Works well with limited data** (crucial for medical imaging)
✅ **Flexible architecture** adaptable to various domains
✅ **State-of-the-art performance** in biomedical imaging
✅ **Easy to train and interpret**

---

## 🏗️ U-Net Architecture Overview

<div align="center">

```
┌─────────────────────────────────────────────────────────┐
│                    U-Net Architecture                    │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Input Image → ┌──────────┐              ┌──────────┐  │
│                │ Encoder  │──────────────→│ Decoder  │  │
│                │(Downsample)  Skip Conn. │(Upsample)│  │
│                └──────────┘              └──────────┘  │
│                     ↓                         ↓         │
│              Feature Extraction    →   Reconstruction   │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

</div>

### Core Components

| Component | Function | Innovation |
|-----------|----------|------------|
| 🔽 **Contracting Path** | Context capture through downsampling | Deep feature extraction |
| 🔼 **Expanding Path** | Precise localization through upsampling | Spatial reconstruction |
| 🔗 **Skip Connections** | Preserve spatial information | Gradient flow & detail preservation |
| 🎯 **Bottleneck** | Lowest resolution, highest semantic level | Global context |

---

## 📊 U-Net Variants

### 🎯 Attention-Based Variants

| Architecture | Paper | Year | Key Innovation | Code |
|--------------|-------|------|----------------|------|
| **Attention U-Net** | [MIDL 2018](https://arxiv.org/abs/1804.03999) | 2018 | Attention gates in skip connections | [PyTorch](https://github.com/ozan-oktay/Attention-Gated-Networks) |
| **SA-UNet** | Spatial Attention U-Net | 2019 | Spatial attention for vessel segmentation | [GitHub](https://github.com/clguo/SA-UNet) |
| **DA-TransUNet** | [Frontiers 2024](https://www.frontiersin.org/journals/bioengineering-and-biotechnology) | 2024 | Dual spatial-channel attention + Transformer | - |
| **R2AU-Net** | R2AU-Net | 2021 | Recurrent + Residual + Attention | - |

### 🔁 Residual Variants

| Architecture | Paper | Year | Key Innovation | Code |
|--------------|-------|------|----------------|------|
| **ResUNet** | [GeoSpatial 2017](https://arxiv.org/abs/1711.10684) | 2017 | Residual blocks, 1/4 parameters of U-Net | [GitHub](https://github.com/nikhilroxtomar/Deep-Residual-Unet) |
| **R2U-Net** | [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC6435980/) | 2018 | Recurrent residual connections | [PyTorch](https://github.com/navamikairanda/R2U-Net) |
| **ResUNet++** | ResUNet++ | 2019 | Atrous convolutions + Attention | - |

### 📐 Multi-Scale Variants

| Architecture | Paper | Year | Key Innovation | Code |
|--------------|-------|------|----------------|------|
| **U-Net++** | [IEEE TMI 2019](https://arxiv.org/abs/1807.10165) | 2019 | Nested U-Nets, dense skip pathways | [Keras](https://github.com/MrGiovanni/UNetPlusPlus) |
| **UNet 3+** | [ICASSP 2020](https://arxiv.org/abs/2004.08790) | 2020 | Full-scale skip connections | - |
| **MultiResUNet** | [Neural Networks 2020](https://github.com/nibtehaz/MultiResUNet) | 2020 | MultiRes blocks with parallel convolutions | [GitHub](https://github.com/nibtehaz/MultiResUNet) |
| **U²-Net** | [Pattern Recognition 2020](https://arxiv.org/abs/2005.09007) ⭐ | 2020 | Two-level nested U-structure, **BEST PAPER** | [Official](https://github.com/xuebinqin/U-2-Net) |

### 🤖 Transformer-Based Hybrids

| Architecture | Paper | Year | Key Innovation | Code |
|--------------|-------|------|----------------|------|
| **TransUNet** | [Medical Image Analysis 2024](https://arxiv.org/abs/2102.04306) | 2021 | CNN + Transformer encoder | [2D](https://github.com/Beckschen/TransUNet) / [3D](https://github.com/Beckschen/TransUNet-3D) |
| **Swin-UNet** | [ECCV 2022](https://arxiv.org/abs/2105.05537) ⭐ | 2022 | Pure Transformer, **BEST PAPER** | [Official](https://github.com/HuCaoFighting/Swin-Unet) |
| **IAUNet** | [CVPR 2025](https://arxiv.org/abs/2508.01928) | 2025 | Query-based U-Net for instance segmentation | [GitHub](https://github.com/SlavkoPrytula/IAUNet) |

### 🔧 Self-Configuring & Specialized

| Architecture | Paper | Year | Key Innovation | Code |
|--------------|-------|------|----------------|------|
| **nnU-Net** | [Nature Methods 2021](https://www.nature.com/articles/s41592-020-01008-z) 🏆 | 2021 | Automatic architecture optimization | [DKFZ](https://github.com/MIC-DKFZ/nnUNet) |
| **3D U-Net** | [MICCAI 2016](https://arxiv.org/abs/1606.06650) | 2016 | Volumetric segmentation from sparse annotations | - |
| **V-Net** | [3DV 2016](https://arxiv.org/abs/1606.04797) | 2016 | 3D FCN with Dice loss for class imbalance | - |

### 💡 Instance Segmentation Specific

| Architecture | Paper | Year | Key Innovation | Code |
|--------------|-------|------|----------------|------|
| **IAUNet** | [CVPR 2025](https://arxiv.org/abs/2508.01928) | 2025 | First query-based U-Net for instance segmentation | [GitHub](https://github.com/SlavkoPrytula/IAUNet) |
| **CotuNet** | [IET CV 2024](https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/cvi2.12275) | 2024 | U-Net + VOLO network blend | - |
| **U-Net-Id** | [Remote Sensing 2020](https://www.mdpi.com/2072-4292/12/23/3941) | 2020 | Building instance segmentation from satellites | - |

---

## 📚 State-of-the-Art Papers

### 🏆 CVPR 2024-2025

| Paper | Venue | Key Contribution | Links |
|-------|-------|------------------|-------|
| **IAUNet: Instance-Aware U-Net** | CVPR 2025 | Query-based U-Net, outperforms Mask2Former | [Paper](https://arxiv.org/abs/2508.01928) \| [Code](https://github.com/SlavkoPrytula/IAUNet) |
| **EoMT: Encoder-only Mask Transformer** | CVPR 2025 (Highlight) | ViT as segmentation model, 4× faster | [Project](https://www.tue-mps.org/eomt/) |
| **SAMEO: Segment Anything, Even Occluded** | CVPR 2025 | SAM for partially occluded objects | [Paper](https://cvpr.thecvf.com/virtual/2025/poster/35221) |
| **Semantic-aware SAM** | CVPR 2024 | Point-prompted instance segmentation | [PDF](https://openaccess.thecvf.com/content/CVPR2024/papers/Wei_Semantic-aware_SAM_for_Point-Prompted_Instance_Segmentation_CVPR_2024_paper.pdf) |
| **EASE: Edge-Aware 3D Instance Segmentation** | CVPR 2024 | SOTA 3D instance segmentation | [PDF](https://openaccess.thecvf.com/content/CVPR2024/papers/Roh_Edge-Aware_3D_Instance_Segmentation_Network_with_Intelligent_Semantic_Prior_CVPR_2024_paper.pdf) |
| **DiverGen** | CVPR 2024 | Generative data for wider distribution | [PDF](https://openaccess.thecvf.com/content/CVPR2024/papers/Fan_DiverGen_Improving_Instance_Segmentation_by_Learning_Wider_Data_Distribution_with_CVPR_2024_paper.pdf) |

### 🎯 ECCV 2024

| Paper | Key Contribution | Links |
|-------|------------------|-------|
| **VISAGE** | Video instance segmentation with appearance guidance | [Paper](https://dl.acm.org/doi/10.1007/978-3-031-72667-5_6) |
| **SAM-guided Graph Cut for 3D** | 3D-to-2D query framework | [Paper](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/6451_ECCV_2024_paper.php) |
| **ProMerge** | Unsupervised instance segmentation | [Paper](https://eccv.ecva.net/virtual/2024/poster/1149) |
| **TAP: Tokenize Anything via Prompting** | Mask & semantic tokens | - |

### 🏥 Medical Imaging (MICCAI & Journals)

| Paper | Venue | Application | Performance |
|-------|-------|-------------|-------------|
| **nnU-Net Revisited** | MICCAI 2024 | 3D medical segmentation benchmark | Still SOTA 🏆 |
| **TransUNet** | Medical Image Analysis 2024 | Hybrid CNN-Transformer | 4.30% Dice improvement |
| **BorderMask** | Scientific Reports 2025 | Enhanced boundary perception | 44.7% AP on COCO |
| **Multi-Transformer U-Net** | PeerJ 2024 | Swin + Deformable Transformer | 98.01% accuracy (X-ray) |
| **CotuNet** | IET Computer Vision 2024 | Instance segmentation | Outperforms on CVPPP |

### 🌟 Foundation Models

| Model | Year | Key Innovation | Performance |
|-------|------|----------------|-------------|
| **SAM 2** | 2024 | Unified image + video segmentation | 6× faster than SAM, better accuracy |
| **SAM 2.1** | 2024 | Improved checkpoints | Enhanced performance |
| **YOLO-World** | 2024 | Zero-shot detection | 35.4 AP at 52 FPS |

---

## 💻 Implementations

### 🐍 PyTorch Implementations

#### ⭐ Recommended: Segmentation Models PyTorch

```python
# Installation
pip install segmentation-models-pytorch

# Usage
import segmentation_models_pytorch as smp

model = smp.Unet(
    encoder_name="resnet34",        # 500+ backbones available
    encoder_weights="imagenet",     # pretrained weights
    in_channels=3,
    classes=1,
    activation='sigmoid'
)
```

**Features:**
- ✅ 500+ pretrained backbones (ResNet, EfficientNet, ConvNeXt, etc.)
- ✅ Multiple architectures: U-Net, U-Net++, FPN, PSPNet, DeepLabV3
- ✅ All encoders with ImageNet weights
- ✅ Simple API

🔗 [qubvel-org/segmentation_models.pytorch](https://github.com/qubvel-org/segmentation_models.pytorch) ![Stars](https://img.shields.io/github/stars/qubvel-org/segmentation_models.pytorch?style=social)

#### Popular Repositories

| Repository | Stars | Description | Link |
|------------|-------|-------------|------|
| **Pytorch-UNet** | 10,700+ ⭐ | High-quality U-Net implementation | [milesial/Pytorch-UNet](https://github.com/milesial/Pytorch-UNet) |
| **pretrained-backbones-unet** | - | 430+ pretrained backbones via timm | [GitHub](https://github.com/mberkay0/pretrained-backbones-unet) |
| **UNet-Instance-Cell-Segmentation** | - | Instance segmentation on cells | [PARMAGroup](https://github.com/PARMAGroup/UNet-Instance-Cell-Segmentation) |
| **UNet-family** | - | Multiple UNet variants | [ShawnBIT](https://github.com/ShawnBIT/UNet-family) |

### 🔶 TensorFlow/Keras Implementations

```python
# Installation
pip install segmentation-models

# Usage
import segmentation_models as sm

model = sm.Unet(
    'resnet34',
    encoder_weights='imagenet',
    classes=1,
    activation='sigmoid'
)
```

| Repository | Stars | Description | Link |
|------------|-------|-------------|------|
| **segmentation_models** | 4,900+ ⭐ | Keras/TF implementation | [qubvel](https://github.com/qubvel/segmentation_models) |
| **keras-unet-collection** | - | Multiple variants (TransUNet, Swin-UNet, etc.) | [yingkaisha](https://github.com/yingkaisha/keras-unet-collection) |
| **image-segmentation-keras** | - | SegNet, FCN, UNet, PSPNet | [divamgupta](https://github.com/divamgupta/image-segmentation-keras) |

---

## 🛠️ Tools & Frameworks

### 🔥 Deep Learning Frameworks

#### **Detectron2** (Meta AI)
- 🎯 State-of-the-art Mask R-CNN implementation
- 📊 38.6 mask AP on COCO, 0.070s/image
- 🔗 [facebookresearch/detectron2](https://github.com/facebookresearch/detectron2)

#### **MMDetection / MMSegmentation** (OpenMMLab)
- 🏆 Won 1st place in 2018 COCO segmentation challenge
- 📦 200+ pretrained models
- ✅ U-Net and U-Net++ support
- 🔗 [MMDetection](https://github.com/open-mmlab/mmdetection) | [MMSegmentation](https://github.com/open-mmlab/mmsegmentation)

#### **Segment Anything Model (SAM)** (Meta AI)
- 🌟 Foundation model for segmentation
- 🚀 Zero-shot capabilities
- 📊 Trained on 11M images, 1.1B masks
- ⚡ 50ms per mask
- 🔗 [SAM](https://github.com/facebookresearch/segment-anything) | [SAM 2](https://ai.meta.com/sam2/)

#### **YOLOv8 Instance Segmentation** (Ultralytics)
- ⚡ Real-time performance (>30 FPS)
- 🎯 0.90 precision, 0.95 recall (outperforms Mask R-CNN)
- 🔗 [Ultralytics](https://docs.ultralytics.com/)

### 🏷️ Annotation Tools

| Tool | Type | Features | Best For |
|------|------|----------|----------|
| **CVAT** | Web-based | AI-assisted, 19 export formats | Teams, complex workflows |
| **LabelMe** | Desktop | Python-based, easy to use | Individual researchers |
| **VIA** | Browser | Lightweight, offline | Quick projects |
| **Roboflow** | Cloud | End-to-end platform | Full pipeline |

🔗 [CVAT](https://www.cvat.ai) | [LabelMe](https://github.com/wkentaro/labelme) | [VIA](https://www.robots.ox.ac.uk/~vgg/software/via/)

---

## 🗃️ Datasets

### 🌍 General Datasets

| Dataset | Images | Instances | Categories | Domain | Download |
|---------|--------|-----------|------------|--------|----------|
| **COCO 2017** | 164K | 2.5M | 80 | General | [Link](https://cocodataset.org/) |
| **Cityscapes** | 5K | - | 30+ | Urban/Driving | [Link](https://www.cityscapes-dataset.com/) |
| **LVIS** | 164K | 2M | 1,203 | Large Vocabulary | [Link](https://www.lvisdataset.org/) |
| **COCONut** | 383K | 5.18M | - | Enhanced COCO | - |
| **ADE20K** | 25K | - | 150 | Scene Parsing | [Link](http://groups.csail.mit.edu/vision/datasets/ADE20K/) |
| **Pascal VOC** | 4.4K | - | 20 | General | [Link](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/) |

### 🏥 Medical Imaging Datasets

| Dataset | Type | Size | Application | Link |
|---------|------|------|-------------|------|
| **NuInsSeg** | Cell nuclei | 665 images, 30K nuclei | Nuclei segmentation | [Kaggle](https://www.kaggle.com/datasets/ipateam/nuinsseg) |
| **PanNuke** | Cancer histology | 200K nuclei | Pan-cancer nuclei | - |
| **MoNuSAC2020** | Multi-organ | - | Nuclei classification | [Challenge](https://monusac-2020.grand-challenge.org/Data/) |
| **BraTS** | Brain MRI | Multi-modal | Brain tumor | - |
| **KiTS19/21** | CT scans | 300 images | Kidney tumor | - |
| **LIDC-IDRI** | CT volumes | 1,012 nodules | Lung nodules | - |

---

## 🎓 Pretrained Models

### 🔥 Top Performing Backbones

#### State-of-the-Art (2024-2025)

| Backbone | Performance | Use Case | Best For |
|----------|-------------|----------|----------|
| **ConvNeXt** | SOTA ⭐ | General | Optimal performance |
| **EfficientNet-B4** | 0.8608 Dice | Medical | Medical imaging |
| **ResNet-50** | Standard | General | Balanced performance |
| **DenseNet-121** | High accuracy | Medical | Dense connections |

### 📦 Available Pretrained Weights

```python
# All ImageNet pretrained backbones available via:

# 1. Segmentation Models PyTorch
import segmentation_models_pytorch as smp
model = smp.Unet('efficientnet-b4', encoder_weights='imagenet')

# 2. Pretrained Backbones UNet
pip install pretrained-backbones-unet

# 3. PyTorch Hub
net = torch.hub.load('milesial/Pytorch-UNet', 'unet_carvana', pretrained=True)
```

### 🏆 Performance Benchmarks

| Model | Dataset | Metric | Score |
|-------|---------|--------|-------|
| U-Net + EfficientNet-B0 | Weed Detection | mIoU | 56.21% |
| U-Net + EfficientNet-B4 | Pneumothorax | Dice | 0.8608 |
| PyTorch-UNet | Carvana | Dice | 0.9884 |

---

## 🏥 Medical Imaging Applications

### 🧠 Brain Tumor Segmentation (BraTS)

**Top Models:**
- **TransUNet**: 4.30% Dice improvement over nnU-Net
- **Spatial-Attention U-Net**: 0.91 DSC, 0.90 Precision, 0.92 Recall
- **CNN-Transformer Hybrid**: 93.4% median Dice (whole tumor)

### 🫀 Multi-Organ Segmentation

**Performance:**
- **FE-SwinUper**: 91.58% Dice (Synapse), 90.15% (ACDC cardiac)
- **Chest Imaging**: 95.3% Dice, 96.4% Precision
- **Liver**: ResTransUNet 95.4% DCI

### 🔬 Cell & Nucleus Segmentation

**Datasets & Models:**
- **Data Science Bowl 2018**: U-Net++ 92.52% IoU
- **MoNuSeg**: Attention-Enhanced U-Net
- **Cervical Cells**: Improved UNet architectures

### 🫁 Kidney & Tumor Segmentation (KiTS)

**Best Results:**
- **96.94% Dice**, 97.40% IoU, 98.23% Precision
- **Residual + MFF**: 97.0% DSC (kidney), 96.0% (tumor)

### 🔍 Pancreatic Tumor

**Two-Stage U-Net + GWBCO:**
- Pancreas: **93.33% DSC**, 92.88% Jaccard
- Tumor: **91.46% DSC**, 88.84% Jaccard

---

## ⚡ Real-Time & Efficient Models

### 🚀 Speed Comparison

| Model | FPS | Inference Time | Parameters |
|-------|-----|----------------|------------|
| **YOLOv8-seg-CP** | 1399 🔥 | 0.72 ms | - |
| **YOLO11** | >60 | <3 ms | 22% fewer than YOLOv8 |
| **PUNet** | - | - | 0.26M |
| **NuLite-S** | - | - | 40× smaller |
| **YOLOv8-seg** | - | 7.8-10.9 ms | ~3.2M |

### 📱 Mobile & Edge Deployment

#### **For Maximum Speed (>60 FPS)**
- YOLOv8-seg-CP: 1399 FPS
- YOLO11: >60 FPS
- YOLOv5 Nano: 230 FPS on GPU

#### **For Mobile Deployment**
- PUNet: 0.26M parameters
- NuLite-S: 40× parameter reduction
- MobileNet-UNet: 95% accuracy

#### **For Edge Devices**
- Eff-UNet: 15 FPS on Jetson TX2
- YOLO-LF: 65.79 FPS
- Lightweight Mask R-CNN: 91.1 MB

### 🔧 Optimization Techniques

✅ Depthwise separable convolutions
✅ Lightweight backbones (MobileNet, Fast-ViT)
✅ Efficient attention mechanisms
✅ Parameter reduction (87.2% in contour-based methods)
✅ GFLOPs reduction (143× in Eff-UNet)

---

## 🎯 U-Net for Landmark Detection

Landmark detection is a fundamental computer vision task that involves localizing specific keypoints in images. U-Net architectures have been successfully adapted for landmark detection through heatmap regression, achieving state-of-the-art results in medical imaging and facial analysis.

### 📐 Landmark Detection Approaches

#### **Heatmap Regression vs Direct Regression**

| Approach | Method | Advantages | Disadvantages |
|----------|--------|------------|---------------|
| **Heatmap Regression** | Generate Gaussian heatmaps per landmark | ✅ Higher accuracy<br>✅ Preserves spatial context<br>✅ SOTA results | ❌ Computational overhead<br>❌ Memory intensive |
| **Direct Regression** | Predict (x,y) coordinates | ✅ Lightweight<br>✅ Simple output | ❌ Lower accuracy<br>❌ Loses spatial info |
| **Hybrid (DSNT)** | Differentiable spatial-to-numerical | ✅ Best of both worlds<br>✅ End-to-end differentiable | - |

### 🏥 Medical Landmark Detection

#### **State-of-the-Art Methods (2024-2025)**

| Method | Year | Application | Performance | Code |
|--------|------|-------------|-------------|------|
| **nnLandmark** | 2025 | Self-configuring 3D detection | **1.2mm** MRE (brain MRI) | - |
| **H3DE-Net** | 2025 | 3D hybrid CNN-Transformer | **1.67mm** MRE (CT) | [GitHub](https://github.com/ECNUACRush/H3DE-Net) |
| **HYATT-Net** | 2024 | Hybrid attention network | **1.13mm** MRE, **84.78%** SDR@2mm | - |
| **MedSapiens** | 2024 | Foundation model adaptation | **+5.26%** SDR improvement | - |
| **FARNet** | 2023 | Feature aggregation & refinement | SOTA on cephalometric | [GitHub](https://github.com/JuvenileInWind/FARNet) |

#### **Clinical Applications**

**Cephalometric Analysis (Orthodontics)**
- **Datasets**: ISBI 2015 (400 images, 19 landmarks), CEPHA29 (1,000 images, 29 landmarks)
- **Clinical Standard**: MRE ≤ 2.0mm considered acceptable
- **Performance**: Best models achieve **1.13mm MRE**, **84.78% SDR@2mm**
- **Application**: Automated orthodontic treatment planning

**Anatomical Landmark Detection**
- **Brain MRI**: 32 anatomical fiducials (AFIDs dataset)
- **Spine**: Vertebral landmark detection for surgical planning (**98% PCK@3mm**)
- **Cardiac MRI**: Valve plane and ventricular landmarks (**99.7% detection rate**)
- **Joints**: Knee, hip landmark detection for alignment assessment

### 👤 Facial Landmark Detection

#### **Top Performing Models (2024-2025)**

| Architecture | Landmarks | Performance | Key Innovation |
|--------------|-----------|-------------|----------------|
| **D-ViT** | 68 | SOTA on 300W | Cascaded dual vision transformer |
| **DSAT** | 98 | **4.25 NME** (300W) | Dynamic semantic aggregation |
| **ORFormer** | 68-98 | Robust to occlusion | Messenger tokens for occlusion |
| **Proto-Former** | 19-124 | Unified multi-dataset | Adaptive prototype-aware |
| **MediaPipe** | 468 (3D) | **99.3%** accuracy | Real-time, cross-platform |

#### **Benchmark Datasets**

| Dataset | Images | Landmarks | Focus | Download |
|---------|--------|-----------|-------|----------|
| **300W** | 3,837 | 68 | Facial in-the-wild | [Link](https://ibug.doc.ic.ac.uk/resources/300-W/) |
| **WFLW** | 10,000 | 98 | Rich expressions | [Link](https://wywu.github.io/projects/LAB/WFLW.html) |
| **AFLW** | 25,993 | 21 | Large pose variations | [Link](https://www.tugraz.at/institute/icg/research/team-bischof/lrs/downloads/aflw) |
| **COFW** | 1,852 | 29 | Heavy occlusion | [Link](http://www.vision.caltech.edu/xpburgos/ICCV13/) |

### 🔥 Heatmap Generation & Loss Functions

#### **Gaussian Heatmap Generation**

```python
def generate_gaussian_heatmap(center_x, center_y, sigma=2.0):
    """
    Generate 2D Gaussian heatmap

    Formula: G(x,y) = exp(-((x-x0)² + (y-y0)²) / (2σ²))
    """
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    gaussian = np.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * sigma**2))
    return gaussian / gaussian.max()
```

**Sigma Selection:**
- **Small σ (1-2px)**: Sharp peaks, better localization, harder to train
- **Medium σ (2-3px)**: Balanced, most common choice
- **Large σ (>3px)**: Easier training, blurrier predictions

#### **Loss Functions**

| Loss Function | Best For | Key Property |
|---------------|----------|--------------|
| **MSE (L2)** | Baseline | Simple, but blurry predictions |
| **Smooth L1** | General | Better than MSE empirically |
| **Adaptive Wing** ⭐ | Heatmap regression | **SOTA** - Adapts to foreground/background |
| **Wing Loss** | Coordinate regression | ❌ Not for heatmaps (divergence) |

**Adaptive Wing Loss** (Recommended):
```python
L_AWing = {
    ω ln(1 + |y-ŷ|/ε|^(α-y))  if |y-ŷ| < θ
    A|y-ŷ| - C                 otherwise
}
```
- **α=2.1, ω=14, ε=1, θ=0.5** (optimal parameters)
- Sensitive to errors on **foreground pixels** (y≈1)
- Tolerates errors on **background pixels** (y≈0)

### 📍 Sub-Pixel Coordinate Extraction

| Method | Precision | Speed | Use Case |
|--------|-----------|-------|----------|
| **Argmax** | Integer only | Fastest | Quick baseline |
| **Soft-Argmax** | Sub-pixel | Fast | **Recommended** ⭐ |
| **Center of Mass** | Sub-pixel | Medium | Robust to noise |
| **Gaussian Fitting** | Highest | Slow | Maximum precision |

**Soft-Argmax Implementation**:
```python
def soft_argmax(heatmap, temperature=10.0):
    # Extract local patch around maximum
    patch = extract_patch_around_max(heatmap, size=5)

    # Temperature-scaled softmax
    softmax = np.exp(temperature * patch) / np.sum(np.exp(temperature * patch))

    # Weighted coordinate average
    coords = np.sum(coordinate_grid * softmax)
    return coords
```

### 📊 Evaluation Metrics

#### **Medical Imaging Metrics**

| Metric | Formula | Clinical Threshold | Best Values (2024) |
|--------|---------|-------------------|-------------------|
| **MRE** | Mean radial error (mm) | ≤ 2.0mm | **1.13mm** (HYATT-Net) |
| **SDR@2mm** | % within 2mm | > 80% | **84.78%** (HYATT-Net) |
| **SDR@4mm** | % within 4mm | > 95% | **97.58%** (pooled studies) |

#### **Facial Landmark Metrics**

| Metric | Normalization | Best Values (2024) |
|--------|---------------|-------------------|
| **NME** | Inter-pupil / inter-ocular distance | **2.41%** (DSAT on 300W) |
| **Failure Rate** | % with NME > threshold | **0.83%** (AWing on 300W) |
| **AUC** | Area under error curve | - |

### 🚀 Real-Time Libraries

| Library | Landmarks | Speed | Platform | Link |
|---------|-----------|-------|----------|------|
| **MediaPipe** 🏆 | 478 (3D) | 30-70ms | Mobile/Desktop/Web | [Google](https://github.com/google/mediapipe) |
| **Dlib** | 68 | <3ms (landmarks) | Cross-platform | [GitHub](http://dlib.net/) |
| **OpenCV DNN** | Detector only | Fast on edge | Embedded | [OpenCV](https://opencv.org/) |

**MediaPipe Features:**
- ✅ 468 3D facial landmarks (most comprehensive)
- ✅ GPU acceleration on mobile
- ✅ No depth sensor required
- ✅ 99.3% accuracy (comparison studies)
- ✅ Android, iOS, Web, Python support

**Dlib Features:**
- ✅ 68-point model (standard)
- ✅ Ultra-fast (<3ms per face)
- ✅ Low memory footprint
- ✅ CPU-optimized
- ✅ 1000 FPS for landmarks alone

### 🤖 Transformer-Based Landmark Detection

Recent transformer architectures dominate 2024-2025 leaderboards:

**Key Innovations:**
- **Deformable Attention**: Focus on relevant regions adaptively
- **Cross-Attention**: Combine features from multiple scales
- **Cascaded Refinement**: Progressive coordinate refinement
- **Messenger Tokens**: Handle occlusions (ORFormer)
- **Dynamic Semantic Aggregation**: Dataset-specific learning (DSAT)

**Performance:**
- Transformers now **match or exceed CNNs** on standard benchmarks
- Particular advantages: **occlusion robustness**, **long-range dependencies**
- Real-time transformers: **DETRPose** (72.5% mAP, 32.5ms latency)

### 🌐 3D Landmark Detection

#### **Methods**

| Approach | Key Innovation | Application |
|----------|----------------|-------------|
| **Monocular 3D** | Depth from single RGB | Face reconstruction |
| **RGBD** | Fusion of RGB + depth | Robust localization |
| **Multi-view** | Geometric consistency | 3D pose estimation |
| **NeRF-based** | Neural radiance fields | Novel view synthesis |

#### **Medical 3D Landmark Detection**

- **3D Faster R-CNN + U-Net**: Coarse-to-fine multi-scale structure
- **H3DE-Net**: Hybrid CNN-Transformer, **1.67mm** MRE on 3D CT
- **nnLandmark**: Self-configuring 3D framework, **1.2mm** MRE
- **Applications**: Surgical planning, craniomaxillofacial analysis, spine surgery

### 💡 Implementation Tips

#### **For Medical Imaging:**
1. Use **heatmap regression** with Adaptive Wing Loss
2. Set σ = 2-3 pixels for Gaussian generation
3. Apply **sub-pixel extraction** via soft-argmax
4. Target MRE < 2.0mm for clinical acceptance
5. Use **nnU-Net framework** for auto-configuration

#### **For Facial Landmarks:**
1. Choose **MediaPipe** for production (best accuracy + speed)
2. Use **Dlib** for lightweight/embedded systems
3. Fine-tune on **combined datasets** (300W + WFLW + COFW)
4. Apply **data augmentation**: rotation, scale, occlusion
5. Monitor **failure rate** at 8-10% NME threshold

#### **Training Best Practices:**
```python
# Recommended setup
encoder = "efficientnet-b4"  # Good accuracy/speed tradeoff
loss = AdaptiveWingLoss(omega=14, theta=0.5, epsilon=1.0, alpha=2.1)
extractor = SoftArgmax(temperature=10.0, patch_size=5)
optimizer = Adam(lr=1e-4)
scheduler = ReduceLROnPlateau(patience=5, factor=0.5)
```

### 📚 Key Resources

#### **Papers (2024-2025)**
- [nnLandmark](https://arxiv.org/abs/2504.xxxxx) - Self-configuring 3D landmark detection
- [H3DE-Net](https://github.com/ECNUACRush/H3DE-Net) - Hybrid 3D detection network
- [DSAT](https://arxiv.org/abs/2412.00740) - Dynamic semantic aggregation
- [ORFormer](https://arxiv.org/abs/2412.13174) - Occlusion-robust transformer
- [Proto-Former](https://arxiv.org/abs/2510.15338) - Unified landmark detection

#### **Code Examples**
See `/examples/landmark_detection/` for complete implementations:
- `facial/facial_landmark_detection.py` - MediaPipe + Dlib comparison
- `medical/medical_landmark_unet.py` - U-Net heatmap regression
- `heatmap/heatmap_methods.py` - Loss functions and coordinate extraction

#### **Datasets**
- **Facial**: [300W](https://ibug.doc.ic.ac.uk/resources/300-W/), [WFLW](https://wywu.github.io/projects/LAB/WFLW.html), [AFLW](https://www.tugraz.at/institute/icg/research/team-bischof/lrs/downloads/aflw)
- **Medical**: [CEPHA29](https://github.com/manwaarkhd/cephalometrix), [AFIDs](https://afids.github.io/), ISBI datasets

---

## 🔮 Future Trends

### 🌟 Latest Innovations (2025)

1. **🤖 Vision Transformers**: Surpassing CNNs in many benchmarks
2. **🎨 Diffusion Models**: Revolutionary probabilistic approaches
3. **🏛️ Foundation Models**: SAM 2, OMG-Seg for universal segmentation
4. **⚡ Real-Time Processing**: YOLACT, lightweight architectures
5. **🎯 Panoptic Segmentation**: Unified semantic + instance understanding

### 🚀 Emerging Techniques

- **Transformer-Diffusion Hybrids**: Combining best of both worlds
- **Self-Supervised Learning**: Reducing annotation requirements
- **Zero-Shot Segmentation**: Foundation models (YOLO-World, SAM)
- **3D Instance Segmentation**: Extending to volumetric data

### 📈 Research Directions

🔬 **Hybrid CNN-Transformer**: Best local + global features
🔬 **Unified Multi-Task**: Single model for multiple tasks
🔬 **Efficient Deployment**: Mobile and edge optimization
🔬 **Medical AI**: Clinical deployment and interpretability

---

## 🎯 Tutorials

### 🚀 Quick Start

<details>
<summary><b>Installation & Basic Usage</b></summary>

```bash
# Install PyTorch
pip install torch torchvision

# Install segmentation models
pip install segmentation-models-pytorch

# Install additional dependencies
pip install albumentations opencv-python matplotlib
```

```python
import torch
import segmentation_models_pytorch as smp

# Create model
model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1,
)

# Move to GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Inference
model.eval()
with torch.no_grad():
    output = model(input_tensor)
```

</details>

<details>
<summary><b>Training Example</b></summary>

```python
from segmentation_models_pytorch.losses import DiceLoss
from segmentation_models_pytorch.metrics import iou_score

# Define loss and metrics
loss_fn = DiceLoss(mode='binary')
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Training loop
for epoch in range(num_epochs):
    model.train()
    for batch in dataloader:
        images, masks = batch
        images, masks = images.to(device), masks.to(device)

        # Forward pass
        outputs = model(images)
        loss = loss_fn(outputs, masks)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Calculate IoU
        iou = iou_score(outputs, masks)
```

</details>

<details>
<summary><b>Instance Segmentation with Watershed</b></summary>

```python
import cv2
import numpy as np
from scipy import ndimage

def watershed_instance_segmentation(semantic_mask):
    """
    Convert semantic segmentation to instance segmentation
    using watershed algorithm
    """
    # Distance transform
    dist_transform = cv2.distanceTransform(semantic_mask, cv2.DIST_L2, 5)

    # Find peaks (sure foreground)
    _, sure_fg = cv2.threshold(dist_transform, 0.5*dist_transform.max(), 255, 0)

    # Find unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(semantic_mask, sure_fg)

    # Marker labelling
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    # Apply watershed
    markers = cv2.watershed(image, markers)

    return markers
```

</details>

### 📚 Complete Examples

See the `/examples` directory for:
- 📝 Training scripts
- 🔍 Inference pipelines
- 🎨 Visualization tools
- 📊 Evaluation metrics
- 🏥 Medical imaging examples
- 📱 Mobile deployment guides

---

## 📖 Citation

If you use this repository in your research, please cite:

```bibtex
@misc{awesome-unet-instance-segmentation,
  title={Awesome U-Net Instance Segmentation},
  author={Community Contributors},
  year={2025},
  publisher={GitHub},
  howpublished={\url{https://github.com/yourusername/Unet-Instance-Segmentation}}
}
```

### Key Papers to Cite

<details>
<summary><b>Original U-Net</b></summary>

```bibtex
@inproceedings{ronneberger2015unet,
  title={U-Net: Convolutional Networks for Biomedical Image Segmentation},
  author={Ronneberger, Olaf and Fischer, Philipp and Brox, Thomas},
  booktitle={MICCAI},
  year={2015}
}
```
</details>

<details>
<summary><b>nnU-Net</b></summary>

```bibtex
@article{isensee2021nnu,
  title={nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation},
  author={Isensee, Fabian and Jaeger, Paul F and Kohl, Simon AA and Petersen, Jens and Maier-Hein, Klaus H},
  journal={Nature methods},
  year={2021}
}
```
</details>

<details>
<summary><b>IAUNet</b></summary>

```bibtex
@inproceedings{prytula2025iaunet,
  title={IAUNet: Instance-Aware U-Net for Biomedical Instance Segmentation},
  author={Prytula, Yaroslav and Tsiporenko, Illia and Zeynalli, Ali and Fishman, Dmytro},
  booktitle={CVPR Workshops},
  year={2025}
}
```
</details>

---

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### How to Contribute:
1. 🍴 Fork the repository
2. 🌿 Create a feature branch
3. ✅ Add your contribution
4. 📝 Update documentation
5. 🔄 Submit a pull request

---

## 📜 License

This repository is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=yourusername/Unet-Instance-Segmentation&type=Date)](https://star-history.com/#yourusername/Unet-Instance-Segmentation&Date)

---

## 🙏 Acknowledgments

Special thanks to:
- 🏛️ Original U-Net authors (Ronneberger et al.)
- 🤝 Open-source community
- 🏥 Medical imaging researchers
- 💻 Framework developers (PyTorch, TensorFlow)
- 📚 All paper authors and contributors

---

<div align="center">

**Made with ❤️ by the Computer Vision Community**

[⬆ Back to Top](#-awesome-u-net-instance-segmentation)

</div>
