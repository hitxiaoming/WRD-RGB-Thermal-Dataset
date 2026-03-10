# WRD-RGB-Thermal-Dataset



# Introduction

This repository provides a method for processing dual-modality images, built upon [MMPretrain](https://github.com/open-mmlab/mmpretrain). 



# Example environment setup



This project is based on the RTX 4090, running on Windows 11. The detailed system configuration is as follows, with a GTX 1060 or higher recommended as the minimum requirement.

```
function(){

​ console.log(”代码块示例“)

}
```



```
pip install
```





# Dataset

The WRD  used in this project was collected in Harbin, China, and is hosted on Zenodo. The dataset consists of nearly 200,000 annotated image pairs categorized into 8 different classes.

![alt text](C:\Users\STI\Desktop\小论文2\WRD-RGB-Thermal-Dataset\figure\tree.png)



# Run the validation script



dual_stream_modules4m.py: This newly added script contains the specific processing methods and architecture for handling the dual-modality data inputs.



```
python tools/test.py configs/your_config_file.py xxx.pth
```