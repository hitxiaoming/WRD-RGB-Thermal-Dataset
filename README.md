<div align="right">
  <strong>English</strong> | <a href="README_zh-CN.md">简体中文</a>
</div>

# WRD-RGB-Thermal-Dataset


# Dataset Introduction

The  road-surface classification dataset (WRD)  used in this project was collected in Harbin, China, and hosted on [Zenodo](https://zenodo.org/records/19503581). This dataset contains nearly 200,000 labeled image pairs, divided into 8 different categories.

![alt text](picture/tree.png)

To facilitate the use of this bimodal dataset, we provide a scheme based on bimodal image processing methods, implemented using [MMPretrain](https://github.com/open-mmlab/mmpretrain).



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
MMEngine	0.10.7
```




# Pre-trained Model Weights

All model weights are hosted on Hugging Face. You can visit the [Hugging Face Repository](https://huggingface.co/hitxiaoming/WRD-RGB-Thermal-Dataset-and-Models) to view them, or download them directly using the links below:

* [`sp_convnextv2.pth` (150 MB)](https://huggingface.co/hitxiaoming/WRD-RGB-Thermal-Dataset-and-Models/resolve/main/sp_convnextv2.pth?download=true)




# Run the train/val script

dual_stream_modules.py: The new scripts include specific processing methods and architectures designed for handling bimodal data inputs, applicable to common model types (CNN, Transformer, and hybrid), and can be used directly without complex configuration.

dual_stream_modulesxx.py: This series of scripts contains data processing scripts designed for specific network architectures, which can reduce network parameters.

```
python train.py \ --config configs/sp_convnextv2.py \ --work-dir sp_convnextv2 
```
Using a trained model for inference, taking ConvNeXtV2 with shared-param as an example.
```
python val_demo.py 
```

# Citation

If you use this dataset in your research, please cite:

> L. Mingwu, Y. Yunfei, L. Wantong, Z. Wanying, W. Jundong, and D. Zejiao. **WRD: RGB-IR dual-spectral dataset for road surface classification in severe winter**. Zenodo, 2026. doi:[10.5281/zenodo.19503581](https://doi.org/10.5281/zenodo.19503581).