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



# Run the validation script



dual_stream_modules4m.py: This newly added script contains the specific processing methods and architecture for handling the dual-modality data inputs.



```
python test.py configs/mynet.py epoch_150.pth
```