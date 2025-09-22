[English](README.md)|[中文](README_zh.md)
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

同时，安装并初始化 Git LFS：

```bash
git lfs install
```

在项目目录下，使用 Git LFS 下载大文件：

```bash
git lfs pull
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

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Thank you for using this project! Please provide feedback if you encounter any issues.

