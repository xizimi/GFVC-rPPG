# Motion-Aligned Physiological Modulation：实际实现与受控训练

2026-11-03。项目路径：`/opt/data/private/project/One-Shot_Free-View_Neural_Talking_Head_Synthesis-main`。
修改前源代码保存在 `architecture_history/20261103_pre_motion_aligned/`。

## 1. 现有网络与实际形状

以下形状对应已确认的原始 SPADE 权重、256×256 RGB 和任意 batch B。

| 步骤 | 实际张量/作用 |
|---|---|
| 参考图像 | `[B,3,256,256]`，float32 RGB，0–1 |
| `first` | `[B,64,256,256]` |
| 两个 `down_blocks` | `[B,128,128,128]`，`[B,256,64,64]` |
| `second` 1×1 Conv2D | `[B,512,64,64]` |
| reshape + 6 个 `resblocks_3d` | 参考体 `F0=[B,32,16,64,64]` |
| 原始 DenseMotionNetwork | 压缩32→4，既有稀疏关键点运动、3D hourglass、softmax混合；不增加第二个运动网络 |
| 稀疏混合 mask | `[B,16,16,64,64]`，第2维是15关键点加背景，不是语义皮肤mask |
| deformation | `[B,16,64,64,3]`，最后一维为 `(x,y,z)`，坐标归一化到约[-1,1]，允许越界采样 |
| occlusion_map | `[B,1,64,64]`，原始视觉遮挡/重建权重，不视为真实皮肤可见性标签 |
| `deform_input(F0,D)` | 5D grid_sample，输出仍 `[B,32,16,64,64]` |
| 深度展平 | 以 channel/depth 顺序展平为 `[B,512,64,64]` |
| `third` + `fourth` + 原有遮挡乘法 | `Ft=[B,256,64,64]`，作为单一新融合位置 |
| SPADE `fc` | 256→512，64×64 |
| `G_middle_0...5` | 6个512通道SPADE残差块，64×64 |
| up + `up_0` | 256通道，128×128 |
| up + `up_1` | 64通道，256×256 |
| `conv_img` + sigmoid | `[B,3,256,256]` |

SPADE 内部各块均以同一个256通道输入特征作为语义条件，按原实现 nearest resize；其 InstanceNorm、spectral norm 和层次结构均未重设计。

**采样约定**：原 `grid_sample` 未显式给参数，当前 fv2v/PyTorch2.1.2 的有效默认值是 mode=bilinear（5D时为三线性）、padding=zeros、align_corners=False。
新 SPADE `deform_input` 显式写出这些相同参数，网格尺寸不同时用三线性插值、align_corners=False。
DenseMotion 和 `make_coordinate_grid` 未修改。原GFVC的linspace端点网格与现代像素中心的差别作为既有约定保留，没有趁本次改动改变原运动表示。
受控位移单测使用明确的像素中心网格验证x/z轴和平移方向，同时真实预训练模型测试确认未改变原GFVC输出。

## 2. 新分支

`modules/physiology_motion.py::MotionAlignedPhysioModulator`：

1. **参考基底**：`Conv3D(32,16,1) → SiLU → Conv3D(16,K,3,pad1) → tanh`，由 `F0` 产生 `B0=[B,K,16,64,64]`。
   它是可学习的调制基底，不是经物理验证的真实血流分布。它不接收逐帧BVP或driving图像。
2. **共享运动**：aligned模式直接调用 `generator.deform_input(B0, D)`，D就是同一帧外观Warp使用的那一个deformation对象；不再次调用DenseMotion。
3. **深度投影**：将 `[K,D]` 展平到K×D，以group=K的1×1 Conv2D学习每个基底的深度汇聚，初始化为深度平均，再tanh，得到 `[B,K,64,64]`。
4. **时序条件**：保持原 `TemporalRPPGEncoder`，整段 `[B,T]→[B,T,64]`，无绝对时间位置编码。
   将当前向量z和显式p拼接送轻量MLP，得到K×256系数；系数再乘 `tanh(p)`。
5. **门控**：由现有Ft预测 `[B,1,64,64]` 的soft skin gate；再乘原GFVC遮挡图，组成g。遮挡图只是辅助约束，未假定它是准确皮肤标签。
6. **融合**：

   `a = tanh(MLP([z,p])) * tanh(p)`

   `deltaF = 0.1 * sigmoid(strength_logit) * g * tanh(sum_k(B2D_k * a_k)/K)`

   `prediction = original_SPADE(Ft + deltaF)`

   strength初始化为0.1，MLP末层采用非零小方差初始化，确保时序、空间、门控和调制头一开始即可收到梯度。
   显式p=0时deltaF严格为0。没有新的自由RGB残差输出头。

**损失语义**：保留原Pearson/waveform/spectral/KL/BPM、visual、outside-skin、skin BCE、residual、temporal smooth公式及权重/epoch调度。
新模式中的 `physiology_delta = prediction - prediction_base` 只是诊断与图像约束所需差值。
`physiology_skin_mask` 是网络预测gate上采样到256；语义真mask只在训练损失和诊断ROI中使用，不传入生成分支。

## 3. 四模式与消融

| 模式 | 生理位置 | 基底共享运动 | 训练参数 |
|---|---|---|---|
| none | 无 | 不适用 | 无，直接使用原GFVC权重推理 |
| rgb_residual | 原RGB输出低秩残差 | 否 | 原temporal encoder、SkinPhysiologyResidual |
| feature_unaligned | SPADE输入特征 | 否，直接投影参考B0 | temporal encoder、新modulator |
| feature_motion_aligned | SPADE输入特征 | 是，同D同Warp | temporal encoder、新modulator |
| feature_motion_aligned + joint_up1 | 同上 | 是 | 以上参数 + 仅decoder.up_1，lr1e-5 |

aligned与unaligned使用完全相同类、参数形状、初始化seed、数据顺序、改写概率和损失，唯一结构开关是是否Warp生理基底。
现有 `SkinPhysiologyResidual` 保留；`RPPGFusion` 仍用于历史显式rppg_feat调用，独立于新的主分支。
feature模式拒绝同时使用另一套legacy rppg_feat注入，避免混淆消融。

K=2时新modulator实测 **43,062** 参数，原TemporalRPPGEncoder **104,000** 参数，Stage1新模式训练参数为 **147,062**。
generator总参数 **101,184,981**，其中旧RGB模块保留但在feature模式冻结且不参与预测。aligned/unaligned参数量完全相同；K=4也可配置，首组正式实验先用K=2。

## 4. 文件与接口变更

| 文件 | 修改 |
|---|---|
| `modules/physiology_motion.py` | 新参考基底/共享Warp/深度投影/时间系数/门控；严格checkpoint schema核对 |
| `modules/rppg.py` | 导出新modulator；旧teacher、temporal encoder、RGB residual、损失实现保留 |
| `modules/generator.py` | SPADE增加模式与prepare_reference/motion_context/decode_context；融合点在fourth/occlusion之后、decoder之前；原SPADEDecoder结构未变 |
| `modules/model.py` | 原逐帧序列接口保持；每段共享参考特征与可微B0；mode-aware训练状态；记录feature norm/gate；损失公式未改 |
| `train_gfvc_rppg.py` | 增加mode/rank/stage/decoder低学习率/当前native数据入口；KP/HE strict=True；生成器显式白名单后strict=True合并加载 |
| `tools/motion_physio/data.py` | 按既有索引加载，不预处理；仅在明确改写模式将Train接触PPG插值到driving相对时间戳；原标签保留 |
| `tools/motion_physio/train_native.py` | 当前划分训练、完整验证、冻结权重核验、断点恢复、每epoch不可变权重和详细说明 |
| `tools/motion_physio/diagnostics.py` | 不依赖teacher的原始Green波形相关性、RGB/皮肤内外变化、PSNR、饱和比例 |
| `tools/motion_physio/inspect_checkpoint.py` | 固定参考及运动，完整12秒BVP-A/B/zero控制；同时测float和量化uint8的Green波形，保存对照图 |
| `tools/motion_physio/run_experiments.py` | 顺序执行受控消融及可选局部微调，失败即停止并保留记录 |

兼容接口仍有 source_image、kp_source、kp_driving、target_rppg、physiology_condition、physiology_value。
`skin_mask` 参数保留但生成器不使用它，防止推理泄漏真实driving mask。
`reference_context` 只在一个明确source张量/版本/形状上复用，不保存在跨样本的全局缓存中。
`prediction_base_override` 是训练时复用已验证原始GFVC float32输出的可选优化，不能替换成原driving视频。
没有冻结或离线缓存新的可训练B0；其梯度仍连接到basis encoder。

旧RGB checkpoint默认mode仍为rgb_residual，参数键不增加新的feature模块。
载入原始GFVC时只允许明确列出的新生理参数缺失；未知key、原主干缺失、shape不匹配以及错误mode恢复均报错。
新权重恢复训练与新/旧模式迁移是不同操作，不以无条件strict=False忽略不匹配。

## 5. 数据、教师和实际训练

数据固定 `/opt/data/private/datasets/GFVC-rPPG_processed/11-3-pro`。
只训练UBFC30人717段native5s，验证6人28段固定12s；LGI和Test不参与训练/选模。
`--sequence_length 150` 兼容已有命令；当前native入口以既有5秒索引为准，UBFC原生T可变，不截断/补帧/重采样。
传统`.face.npy`入口仍保持原150帧行为。

教师复用 `checkpoint/ubfc_11_3_subject_independent_20261103/teacher/teacher_best.pt`，不重新训练、不按结果调教师。
它的独立UBFC验证波形PCC约0.868仅说明既有校准表现，不保证在生成图像上的测量无偏。
新实验保留原损失，同时单列teacher-independent Green/视觉/uint8诊断。

首组新消融改写概率统一0.3；旧已完成RGB实验是preserve-only，因此队列额外重训一个**同配置RGB对照**。
改写只选Train被试有效连续片段中的接触PPG，按真实时间戳插值到当前视频时间轴；不改原PPG、数据划分或视频，不做相关性最大化/平移标签。
验证固定包含28段preserve和28段Train-donor rewrite；选模分数为两组完整验证总损失均值的等权平均。
所有mask仅作监督/诊断，推理端没有真实driving mask输入。

Stage1冻结KP、HE、DenseMotion、参考编码和完整SPADE；仅训练现有时序编码器及相应生理模块。
Stage2仅解冻 `decoder.up_1`，它仍保持eval模式以固定spectral-norm估计buffer；参数梯度有效。
视觉约束的baseline锚定此前验证过的原始GFVC float32缓存，防止把共同漂移的微调模型误当视觉基准。

原始GFVC缓存目录：`/opt/data/private/datasets/GFVC-rPPG_generated/ubfc_11_3_frozen_base_float32_cache_loader_layout_v2`。
本任务不改变该缓存或预处理数据。训练使用decoder激活重计算以保留完整5秒的时间梯度。

## 6. 可执行命令

从项目根目录运行（这些文件和权重均已在服务器核实）：

```bash
/opt/conda/envs/fv2v/bin/python train_gfvc_rppg.py \
  --config config/vox-256-motion-physio.yaml --generator spade \
  --gfvc_checkpoint checkpoint/00000189-checkpoint.pth.tar \
  --rppg_backend skin \
  --rppg_checkpoint checkpoint/ubfc_11_3_subject_independent_20261103/teacher/teacher_best.pt \
  --contact_clip_root /opt/data/private/datasets/GFVC-rPPG_processed/11-3-pro \
  --sequence_length 150 --rewrite_probability 0.3 --require_semantic_masks \
  --signal_preprocess none --physiology_fusion_mode feature_motion_aligned \
  --physiology_rank 2 --training_stage physiology --checkpoint_decoder \
  --epochs 30 --patience 8 --seed 42 --num_workers 1 --lr 0.0001 \
  --output checkpoint/motion_physio_20261103_v2/feature_motion_aligned_stage1_seed42_rank2
```

注意：native入口的output是**运行目录**，不是单个文件；该目录非空时必须使用明确的resume。
顺序消融由 `python tools/motion_physio/run_experiments.py` 启动，避免同GPU多个训练互相挤占显存。
它运行aligned Stage1、unaligned Stage1、匹配RGB Stage1，最后用aligned最佳权重初始化5轮joint_up1。
若局部微调未提升预先声明的验证指标/视觉约束，则保留Stage1最佳权重，比较结果单独保存。

## 7. checkpoint说明与检索

见项目根 `CHECKPOINT_CATALOG.md`。
每轮生成 `epoch_XXXX.pth.tar` 和同名 `.json`、`.md` 说明；文件不可变。
best/latest仅为指向实际epoch文件的符号链接，不会将不同实验互相覆盖。
每个sidecar记录mode、rank、阶段、epoch/step、初始/父权重、SHA256、精确训练/验证被试、数据索引哈希、
接触标签与改写规则、可训练/冻结参数名、优化器、逐样本验证诊断和选模分数。
每个run有 `run_config.json`、`checkpoint_index.json`、`history.json`、`code_snapshot/`。
smoke权重和正式训练目录分开，不能用于宣称正式训练结果。

## 8. 已验证结果与研究边界

模块单测覆盖已知x/z位移、零BVP/零遮挡、batch2形状、换BVP、各模块梯度、K=4、参考缓存样本/版本保护和严格加载；
真实预训练256模型的batch2兼容检查；旧RGB权重输出最大误差0；原GFVC无生理输入误差0；新模式无输入误差0；
零BVP额外调制误差0；同一网格对象确实用于两次Warp；更换BVP后像素最大差约0.000211；真实mask不改变推理输出。
完整native5秒2步训练 + 12秒验证检查通过，各新子模块梯度非零，原始主干与教师未变；峰值显存8,086,498,304字节。
保存/恢复到下一epoch的检查也已通过。旧rPPG pipeline回归测试通过。

这些检查证明实现接口、受控响应和可训练性，不证明最终生理或视觉质量优于旧模型。
当前皮肤门控为64×64学习后上采样，SPADE也可传播空间影响；是否主要限制在真实皮肤、是否在眼睛/头发/背景产生异常，需要完整训练后的定量和可视化诊断。
同时记录float与uint8波形，避免把量化后消失的极小扰动误认为可用生理信息。
不能把新的基底解释为真实血流，不宣称率失真优化，也未修改量化、熵编码、码率控制或关键点运动表示。
默认诊断输出包含原始baseline解码以便求RGB差，feature模式推理会有这次额外解码开销；训练用已核验baseline缓存避免重复baseline解码。
新modulator参数很少并不意味着完整SPADE反向传播免费。单步耗时、显存和训练后控制报告保留在每次实验记录中。

## 当前审查版本

正式修订运行目录为 `checkpoint/motion_physio_20261103_v2/`。
初版 `checkpoint/motion_physio_20261103/` 保留检查与中断记录；中断原因及最后状态见其 `TRAINING_INTERRUPTED.json`。
本次审查包含已实现的架构、测试和训练记录机制。新架构完整训练及最终性能比较尚未完成。
