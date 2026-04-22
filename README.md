# WRD-RGB-Thermal-Dataset



# Introduction

This repository provides a method for processing dual-modality images, built upon [MMPretrain](https://github.com/open-mmlab/mmpretrain). 



# Example environment setup



This project is based on the RTX 4090, running on Windows 11. The detailed system configuration is as follows, with a GTX 1060 or higher recommended as the minimum requirement.

```
Python	3.8.20
PyTorch	2.1.0 + cu118
TorchVision	0.16.0 + cu118
CUDA Runtime / Toolkit	11.8
cuDNN	8.7.0
NVCC	11.8.89
OpenCV	4.11.0
MMEngine	0.10.7pip install
```


# Dataset

The WRD  used in this project was collected in Harbin, China, and is hosted on Zenodo. The dataset consists of nearly 200,000 annotated image pairs categorized into 8 different classes.

![alt text](picture/tree.png)

# Pre-trained Model Weights

All model weights are hosted on Hugging Face. You can visit the [Hugging Face Repository](https://huggingface.co/hitxiaoming/WRD-RGB-Thermal-Dataset-and-Models) to view them, or download them directly using the links below:

* [`sp_convnextv2.pth` (150 MB)](https://huggingface.co/hitxiaoming/WRD-RGB-Thermal-Dataset-and-Models/resolve/main/sp_convnextv2.pth?download=true)




# Run the train/val script

dual_stream_modules.py: This newly added script contains specific processing methods and architectures designed for handling dual-modality data inputs, applicable to common model types (CNNs, Transformers, and hybrid architectures).

dual_stream_modulesxx.py: This series consists of data processing scripts designed for specific network architectures with shared parameters.

```
python train.py \ --config configs/db_convnextv2.py \ --work-dir db_convnextv2 
```

```
python demo.py configs/sp_convnextv2.py weight/sp_convnextv2.pth
```