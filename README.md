
# Dog Breed Classification - Based on ViT

## Note

This project is a group project for the 2022 training program at Shanxi Agricultural University. The code was completed by the group leader within two days, implementing only basic logic. Due to limited computational resources and time, the model is not optimized. Increasing the number of `epochs` during training is recommended to improve performance.

## Project Overview

This project fine-tunes the `vit-base-patch16-224-in21k` model provided by Google to classify dog breeds.
- **Training Duration**: 5 epochs.
- **Deep Learning Framework**: PyTorch.
- **Frontend and Backend**: Flask and HTML.

With simple steps, you can access the WebUI on local port `5050` and drag and drop images for dog breed classification.

## Project Structure

```
.
├── app/                # Frontend and backend code
│   ├── app.py          # Flask backend main program
│   ├── index.html      # Frontend HTML file
│   ├── static/         # Static resources
│   └── uploads/        # Directory for uploaded images
├── models/             # Directory for fine-tuned models
│   └── vit_finetuned_StanfordDogs_ep5
├── training/           # Training-related code
│   ├── eva.ipynb
│   ├── training_vit_dogs.ipynb
│   ├── vit_on_cifar10.ipynb
│   └── vit_on_stanford_dogs-Copy1.ipynb
├── uploads/            # Upload directory
└── README.md           # Project description file
```

## Quick Start

### Environment Requirements
- **Operating System**: Linux Ubuntu 24.04
- **CUDA**: 12.1
- **Python**: 3.11
- **Torch**: 2.5.0
- **Torchvision**: 0.20.0
- **Transformers**: 4.47.1

### Install Dependencies
Ensure your Python environment meets the requirements and install the necessary dependencies:

```bash
pip install torch torchvision transformers flask
```

### Start the Project

1. Navigate to the project directory:
   ```bash
   cd vit_dogs_classifier
   ```

2. Navigate to the `app/` directory:
   ```bash
   cd app/
   ```

3. Start the service:
   ```bash
   python app.py
   ```

4. Open your browser and visit [http://localhost:5050](http://localhost:5050).

5. Drag and drop an image to the interface for classification.

![WebUI Interface](./imgs/webui.jpg)

## Training Parameters

- **Data Path**: `data_dir = "./data/Images"`
- **Hyperparameters**:
  ```python
  learning_rate = 5e-5
  batch_size = 128
  training_steps = 1000
  num_classes = 120
  num_epochs = 5
  ```
- **Adam Optimizer Parameters**:
  ```python
  betas = (0.9, 0.999)
  eps = 1e-8
  ```

## Training Details

The training code is located in the `./training/` directory and can be executed directly.

Dataset download link: [Stanford Dogs Dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/).

Model source: [Hugging Face ViT Base](https://huggingface.co/google/vit-base-patch16-224-in21k).

Parameter reference: [Training Parameters](https://huggingface.co/amaye15/google-vit-base-patch16-224-batch64-lr0.005-standford-dogs).

## Frontend Design

The frontend mouse trail effect design is inspired by: [Chokcoco CodePen](https://codepen.io/Chokcoco/pen/XgvjQM).

---
# 狗狗识别——基于ViT的犬类品种分类

## 注意

本项目是山西农业大学2022级实训的小组项目，代码由组长一人在两天内赶工完成，仅实现了基本逻辑，整体功能未完全完善。由于算力和时间限制，模型并未达到最优效果，建议设置更多的`epoch`进行训练以提升性能。

## 项目简介

本项目使用Google提供的`vit-base-patch16-224-in21k`模型进行微调，完成了犬类品种分类任务。
- **训练时长**: 5个epochs。
- **深度学习框架**: PyTorch。
- **前后端实现**: Flask 和 HTML。

通过简单的操作，即可在本地端口`5050`访问前端WebUI，并拖拽图片实现犬类品种的识别。

## 项目结构

```
.
├── app/                # 前端及后端代码目录
│   ├── app.py          # Flask后端主程序
│   ├── index.html      # 前端HTML文件
│   ├── static/         # 静态资源目录
│   └── uploads/        # 上传的图片存放目录
├── models/             # 微调后的模型存放目录
│   └── vit_finetuned_StanfordDogs_ep5
├── training/           # 训练相关代码目录
│   ├── eva.ipynb
│   ├── training_vit_dogs.ipynb
│   ├── vit_on_cifar10.ipynb
│   └── vit_on_stanford_dogs-Copy1.ipynb
├── uploads/            # 上传目录
└── README.md           # 项目说明文件
```

## 快速开始

### 环境要求
- **操作系统**: Linux Ubuntu 24.04
- **CUDA**: 12.1
- **Python**: 3.11
- **Torch**: 2.5.0
- **Torchvision**: 0.20.0
- **Transformers**: 4.47.1

### 安装依赖
确保Python环境满足上述要求后，使用以下命令安装所需依赖：

```bash
pip install torch torchvision transformers flask
```

### 启动项目

1. 切换到项目目录：
   ```bash
   cd vit_dogs_classifier
   ```

2. 切换到`app/`目录：
   ```bash
   cd app/
   ```

3. 启动服务：
   ```bash
   python app.py
   ```

4. 在浏览器中访问[http://localhost:5050](http://localhost:5050)。

5. 拖拽图片到界面完成识别。

![WebUI界面](./imgs/webui.jpg)

## 训练参数

- **数据路径**: `data_dir = "./data/Images"`
- **超参数**:
  ```python
  learning_rate = 5e-5
  batch_size = 128
  training_steps = 1000
  num_classes = 120
  num_epochs = 5
  ```
- **Adam优化器参数**:
  ```python
  betas = (0.9, 0.999)
  eps = 1e-8
  ```

## 训练相关

在`./training/`目录中放置了训练所需的代码，直接运行即可进行训练。

数据集下载地址：[Stanford Dogs Dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/)。

模型来源：[Hugging Face ViT Base](https://huggingface.co/google/vit-base-patch16-224-in21k)。

参数参考：[Training Parameters](https://huggingface.co/amaye15/google-vit-base-patch16-224-batch64-lr0.005-standford-dogs)。

## 前端设计

前端设计的鼠标拖尾思路来源于：[Chokcoco CodePen](https://codepen.io/Chokcoco/pen/XgvjQM)。

---

Thank you for using this project! Please provide feedback if you encounter any issues.

