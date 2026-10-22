# rPPG-GFVC 工作记录

## 1. 总体目标

我们的目标是在 GFVC/生成式人脸视频编码的基础上，加入 **rPPG 相关能力**，形成一套既保留人脸生理信号、又支持隐私保护与可控编辑的框架。

核心目标是：

1. **关键帧直接传输**
   - 关键帧不经过复杂压缩，直接传给解码端。
   - 保留参考人脸的静态外观信息。

2. **后续视频片段特征化**
   - 把后续人脸视频送入编码端。
   - 提取：
     - 人脸外观/几何特征；
     - 表情与头部姿态；
     - rPPG/脉搏相关特征。
   - 传输这些紧凑特征。

3. **解码端恢复带 rPPG 信号的视频**
   - 解码端利用关键帧和接收到的特征。
   - 恢复出带有 rPPG 信号的人脸视频。
   - 保证 rPPG 可检测、可评估。

4. **动漫风格隐私保护**
   - 将关键帧替换为动漫角色脸部。
   - 解码端仍然能够进行 rPPG 检测。
   - 实现人脸隐私保护。

5. **rPPG 特征编辑 / 篡改**
   - 在编码特征或解码过程中修改 rPPG 特征。
   - 可用于可控 rPPG 合成、匿名化或伪装。

## 2. 当前已确认状态

### 基线模型

当前使用：

```text
/opt/data/private/project/One-Shot_Free-View_Neural_Talking_Head_Synthesis-main/
```

该方法是基于：

```text
One-Shot Free-View Neural Talking-Head Synthesis for Video Conferencing
```

特点：

- 单张人脸图 + 驱动视频 → 生成人脸视频；
- 使用 3D 关键点；
- 支持头部姿态与表情控制；
- 视觉效果较好，适合作为高质量 GFVC baseline。

### rPPG 参考项目

另一个参考项目：

```text
/opt/data/private/project/rePPG-Rewriting-Photoplethysmography-Signal-to-Video-master/
```

后续可以从这里提取或借鉴：

- rPPG 特征提取方式；
- rPPG 信号重建方式；
- 生理信号与视频帧之间的映射方式；
- 可控 rPPG 编辑思路。

## 3. 期望整体流程

```text
原始人脸视频
    |
    |--- 关键帧 ------------------------------→ 直接传输
    |
    |--- 后续视频片段 --------------------------→ 编码端特征提取
                                                  |
                                                  |-- 人脸外观/几何特征
                                                  |-- 表情/头部姿态
                                                  |-- rPPG/脉搏特征
                                                  |
                                                  ▼
                                              紧凑特征码流
                                                  |
                                                  ▼
                                       解码端人脸视频重建
                                                  |
                                                  ▼
                                     带 rPPG 信号的人脸视频
```

隐私/风格化部分：

```text
关键帧
  |
  |--- 保留原始人脸
  |--- 替换为动漫风格人脸
  |
  ▼
解码端仍支持 rPPG 检测
```

## 4. 需要加入的关键模块

### 4.1 rPPG 特征提取模块

负责从人脸视频中提取脉搏相关特征。

可能的输入：

- 人脸区域图像；
- 面部 ROI；
- 皮肤色度变化；
- 局部纹理/微弱运动。

可能的输出：

- rPPG 信号；
- rPPG 特征向量；
- 心率/脉搏相关表征。

### 4.2 rPPG 注入模块

将 rPPG 特征注入生成器。

可能注入位置：

1. 表情/头部姿态之后；
2. 3D 关键点变换之后；
3. generator 的中间特征；
4. 解码器的纹理/颜色通道。

目前更倾向于：

```text
rPPG feature
    |
    |--- 注入 generator 的中间特征
    |--- 或注入局部颜色/纹理通道
```

### 4.2.1 当前倾向：显式 rPPG 信号

参考项目使用的是显式 rPPG 信号，而不是纯隐式特征。相关结构是：

```text
cPPG / rPPG signal
    ↓
recon_ffn
    ↓
ReconstructionNet
    ↓
per-frame feature map
    ↓
Hourglass 中间层
    ↓
生成带目标 rPPG 的视频
```

这个方案更适合我们的目标，因为：

- 可以直接编辑目标 rPPG 波形；
- 可以直接评估心率/脉搏指标；
- 可以在码流端显式控制 rPPG；
- 比纯隐式特征更容易解释；
- 更适合做隐私保护与生理信号伪装。

隐式 rPPG 特征可以以后作为对比，但当前优先使用显式信号。

### 4.3 动漫风格渲染模块

将原始人脸关键帧替换为动漫风格。

目标是：

- 隐藏真实身份；
- 保持面部几何；
- 保持表情和头部姿态；
- 保持 rPPG 可检测性。

### 4.4 rPPG 编辑模块

用于修改 rPPG 特征。

可能的操作：

- 心率偏移；
- 周期性强度变化；
- 生理信号替换；
- 生理信号伪装。

## 5. 当前环境

远程服务器：

```text
172.25.100.130:25281
root@172.25.100.130
```

当前环境：

```bash
/opt/conda/envs/fv2v/bin/python
```

已确认可用：

- Python 3.10.22；
- PyTorch 2.1.2+cu121；
- torchvision 0.16.2+cu121；
- OpenCV；
- scikit-image；
- pandas；
- scipy；
- sklearn；
- GPU：NVIDIA GeForce RTX 4090。

## 6. 下一步计划

1. 确认 FV2V 代码可运行。
2. 用预训练模型跑通基础推理。
3. 分析 rPPG 参考项目，确定可复用模块。
4. 设计 rPPG 特征提取与注入接口。
5. 在 FV2V 中加入 rPPG 分支。
6. 构建带 rPPG 信号的解码端。
7. 加入动漫风格关键帧替换。
8. 加入 rPPG 编辑能力。
9. 设计评估指标：
   - 重建质量；
   - rPPG 精度；
   - 隐私保护强度；
   - rPPG 可控性。

## 7. 需要后续讨论的问题

1. rPPG 特征是显式信号还是隐式特征？
2. 是否需要显式心率作为可编辑参数？
3. 动漫风格转换放在编码端还是解码端？
4. rPPG 编辑是在编码端、码流端还是解码端？
5. 是否需要加入 VVC/FOMM/CFTE 等对比基线？
6. 数据集使用 VoxCeleb、UBFC-rPPG 还是自采数据？

## 8. 当前进度补充

### 数据处理已完成

原始 UBFC2 数据集位于：

```text
/opt/data/private/datasets/UBFC2/
```

共 42 个 subject。

subject 级别划分文件：

```text
/opt/data/private/project/One-Shot_Free-View_Neural_Talking_Head_Synthesis-main/splits.json
```

划分结果：

```text
train: 32 subjects
val:   5 subjects
test:  5 subjects
```

### 5 秒片段

预处理脚本：

```text
tools/preprocess_ubfc2.py
```

输出目录：

```text
/opt/data/private/datasets/UBFC2_processed/
```

每个 subject 下包含：

```text
clip_0000.face.npy
clip_0000.sig.npy
clip_0001.face.npy
clip_0001.sig.npy
...
meta.json
```

片段参数：

```text
window_size = 150
step_size   = 75
size        = 256
```

### FV2V 可读取划分

已生成 FV2V 训练目录：

```text
/opt/data/private/datasets/UBFC2_fv2v/
```

片段数量：

```text
train: 778 clips
val:   120 clips
test:  128 clips
```

已抽查训练片段，可以正常加载原始帧数组：

```text
frames = 150
shape  = (150, 256, 256, 3)
dtype  = uint8
```
