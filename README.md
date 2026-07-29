# TriDom-YOLO 水稻病虫害检测

面向复杂稻田环境设计的轻量级视觉检测架构。项目以 PyTorch 与 Ultralytics YOLO11 为基础，通过频率域增强、空间域细化和门控域筛选构建三域渐进式特征链，重点强化小病斑、弱边缘、叶片遮挡、周期性背景纹理与运动模糊场景下的特征表达。

> 本仓库为作品集精简版，仅保留模型主体架构、自研模块和必要的训练入口。数据集、权重、训练日志、实验结果、评估图表、复现实验脚本及完整研究文档不对外公开。

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

| 模块 | 作用 |
| --- | --- |
| PFAE | 融合多尺度卷积与二维频率注意力，抑制周期性背景干扰并增强病斑边缘 |
| DI-SpAM | 结合膨胀深度卷积、SimpleGate 与通道建模，聚合多尺度空间上下文 |
| Gated CNN | 对高层特征进行位置相关的动态筛选，保留有效响应并削弱冗余背景 |
| YOLO11 Neck + Head | 完成跨尺度特征融合与水稻病虫害目标预测 |

## 技术体系

- Python、PyTorch、Ultralytics YOLO11
- FFT/IFFT 频率建模与注意力机制
- Depthwise Convolution、Dilated Convolution、SimpleGate
- Gated CNN、PSA、多尺度特征融合与小目标检测
- CUDA 训练、混合精度计算、模型评估与边缘部署适配

## 公开结构

| 路径 | 内容 |
| --- | --- |
| `ultralytics/change_model/` | PFAE、DI-SpAM 与 Gated CNN 自研模块 |
| `ultralytics/cfg/models/11/` | TriDom-YOLO 网络结构配置 |
| `ultralytics/nn/` | 模型解析和基础网络组件 |
| `ultralytics/models/yolo/detect/` | 检测训练、验证与推理主体 |
| `train_yolov11.py` | 参数化训练入口 |

## 公开范围

当前版本用于呈现算法设计与代码组织。原始图像、标注信息、类别统计、数据划分、训练权重、超参数复现细节、消融实验、指标曲线和预测样例均已从公开仓库移除。

## License

本项目基于 Ultralytics YOLO 修改，公开代码遵循仓库中的 [AGPL-3.0 License](LICENSE)。
