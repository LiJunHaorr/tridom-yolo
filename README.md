# TriDom-YOLO｜三域增强轻量级目标检测网络

基于 PyTorch 与 Ultralytics YOLO11 构建的复杂场景视觉检测架构。项目围绕小目标、弱纹理、边缘模糊、密集遮挡和周期性背景干扰等视觉难题，引入频率域增强、空间域细化与门控域筛选，形成三域渐进式特征优化链。

> 本仓库为作品集精简版，仅展示模型主体架构、自研算子和必要训练入口。

## 模型架构

```text
Input
  -> YOLO11 Backbone
  -> PFAE Frequency Enhancement
  -> DI-SpAM Spatial Refinement
  -> Gated CNN Feature Selection
  -> Multi-scale Neck
  -> Detect Head
```

## 核心设计

| 模块 | 技术作用 |
| --- | --- |
| PFAE | 融合多尺度卷积与二维频率注意力，通过 FFT/IFFT 建模强化结构边缘并抑制周期性噪声 |
| DI-SpAM | 组合膨胀深度卷积、SimpleGate 与通道建模，在受控计算量下聚合多尺度空间上下文 |
| Gated CNN | 对高层语义特征执行位置相关的动态筛选，保留高价值响应并削弱冗余背景激活 |
| YOLO11 Neck + Head | 完成跨尺度特征融合、候选区域建模与目标分类回归 |

## 技术体系

- Python、PyTorch、Ultralytics YOLO11
- FFT/IFFT 频率建模、二维频率注意力与多尺度卷积分支
- Depthwise Convolution、Dilated Convolution、SimpleGate
- Gated CNN、PSA、跨尺度特征融合与小目标表征
- CUDA、混合精度计算与边缘推理适配

## 公开结构

| 路径 | 内容 |
| --- | --- |
| `ultralytics/change_model/` | PFAE、DI-SpAM 与 Gated CNN 自研模块 |
| `ultralytics/cfg/models/11/` | TriDom-YOLO 网络拓扑配置 |
| `ultralytics/nn/` | 模型解析、算子注册和基础网络组件 |
| `ultralytics/models/yolo/detect/` | 检测训练、验证与推理主体 |
| `train_yolov11.py` | 参数化训练入口 |

## 公开边界

当前版本仅用于呈现算法设计、模型结构与代码组织。

## License

本项目基于 Ultralytics YOLO 修改，公开代码遵循仓库中的 [AGPL-3.0 License](LICENSE)。
