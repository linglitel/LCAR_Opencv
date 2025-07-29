# LCAR_Opencv

本项目是 LCAR 项目的一部分，代号为 **LCAR_Opencv**，用于在 RK3588 平台上基于 RKNN 和 OpenCV 实现水果目标检测推理。适用于边缘侧部署场景，支持实时摄像头采集与识别显示。

## 🚀 项目简介

运行于 **鲁班猫4 RK3588 (8GB+64GB)**，采用 **RKNN SDK + YOLOv11n 模型** 实现水果识别，使用 C++ 单线程配合 OpenCV 进行画面采集与渲染。

识别类别如下：

- Watermelon 🍉  
- Orange 🍊  
- Grape 🍇  
- Apple 🍎  
- Peach 🍑  
- Banana 🍌  
- Pineapple 🍍

模型经过轻量化处理（去除后处理模块），经 RKNN 工具链优化后部署，兼容 RK3588 NPU 推理。

## ⚙️ 项目特点

- 📷 支持实时摄像头图像识别（640x640）
- ⚡ 初步估计单帧推理时间约为 20ms
- 🧠 轻量 YOLOv11n 模型，适用于近距离识别
- 🔧 模型输入输出自定义后处理流程
- 📚 数据集包含约 **1000+ 张水果图像**，涵盖 7 类水果

## 📦 模型与数据

- 模型：YOLOv11n，去除后处理模块
- 格式：RKNN，适用于 RK3588
- 输入张量：
  - shape: `[1, 640, 640, 3]`
  - format: `NHWC`
  - dtype: `INT8`
  - quant: scale=0.003922, zero_point=-128
- 输出张量：共 9 个，需结合后处理解码

## 🧰 依赖环境

- RKNN-Toolkit-Lite2
- OpenCV
- C++17 或更高
- CMake 构建工具

## 🏁 快速开始

1. 克隆项目：

```bash
git clone https://github.com/linglitel/LCAR_Opencv.git
cd LCAR_Opencv
```

2. 编译项目：

```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
```

3. 运行摄像头推理：

```bash
./build/fruit_detector
```

## 📂 项目结构

```
LCAR_Opencv/
├── include/            # 头文件（后处理接口等）
├── lib/                # RKNN SDK 库文件
├── model/              # YOLOv11n.rknn 模型文件
├── src/                # 核心代码逻辑（main.cpp, postprocess.cpp）
├── build/              # 构建输出目录（可忽略）
├── CMakeLists.txt      # 构建配置
└── README.md           # 项目说明
```

## 📌 注意事项

- 模型精度受限于训练数据，仅用于演示用途
- 摄像头采集建议分辨率为 640x480 或 1280x720（需 resize）
- 数据集较小，适合做样例或测试用途
- 若用于量产或商用，请使用更大规模数据集进行完整训练

## 👤 开发者信息

由 [linglitel](https://github.com/linglitel) 开发维护，隶属于 **LinCore 工作室**。  
项目部分代码来源自 [rknn_model_zoo](https://github.com/airockchip/rknn_model_zoo)，归属于 Rockchip 所有，已根据需要适配与修改。

如有侵权问题，请及时联系，我们将依法删除。

## 📄 协议

本项目基于 [MIT License](LICENSE) 开源。

---

🧠 Powered by RKNN • YOLOv11 • OpenCV • LinCore Studio

