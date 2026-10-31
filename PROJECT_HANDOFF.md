# GFVC-rPPG 项目交接与阶段性成果

本文档面向后续接手本项目的 AI/研究者。目标是让接手者能快速定位代码、数据、模型、实验协议和当前结论。

> 不在仓库、日志或文档中保存服务器密码、GitHub Token 等密钥。

## 1. 研究目标

在 GFVC（关键帧 + 每帧运动参数）视频传输框架中附加一维目标 rPPG/BVP 波形，使接收端生成视频同时满足：

1. 保留模式（preserve）：目标波形等于源视频自身的 BVP；
2. 改写模式（rewrite）：源人脸外观与运动保持不变，但生成视频携带另一个供体的 BVP；
3. 后续隐私扩展：将关键帧替换为二维动漫头像，保留运动参数和 BVP 条件，实现匿名化 rPPG 传输。

期望的通信抽象为：

```text
发送端: 关键帧 I_key + GFVC 运动参数 M_1:T + 目标波形 y_1:T
                                     ↓
接收端: GFVC(I_key, M_1:T, y_1:T) → 带目标 rPPG 的视频 V_hat
```

## 2. 当前模型结构

主模型定义：`modules/model.py` 中的 `GFVCRPPGSequenceModel`。

现阶段采用低秩、显式颜色调制，而不是任意 RGB 残差。核心思想为在基础 GFVC 帧上叠加受皮肤门控约束的生理扰动：

```text
目标 BVP y_t
  → 时间颜色分支（每时刻 RGB 调制向量）
空间响应图 A_k(x,y) + 皮肤权重 M_t(x,y)
  → rank=2 的低秩颜色调制
  → V_hat_t = V_base_t + ΔV_t
```

其中调制强度通过 `tanh` 有界，避免生成器以过大的任意颜色残差欺骗 rPPG Teacher。当前默认 rank 为 2，且自由 RGB 残差分支关闭；这是一种需要后续消融验证的设计选择。

训练时的关键组件：

- GFVC 基础部件：Generator、KPDetector、HEEstimator；
- `SkinRPPGTeacher`：从“视频 × 预计算皮肤掩膜”提取完整 BVP；
- `SkinPhysiologyResidual`：根据目标 BVP 向皮肤区域注入低秩颜色变化；
- 皮肤掩膜：离线由 MediaPipe convex-hull 得到，作为门控和生理损失 ROI；不是改变输入视频内容的预处理。

## 3. 服务器、环境、数据与权重

以下均为服务器路径（SSH 主机信息和凭据不写入本文件）：

| 用途 | 位置 |
|---|---|
| 主项目 | `/opt/data/private/project/One-Shot_Free-View_Neural_Talking_Head_Synthesis-main` |
| pyVHR 源码 | `/opt/data/private/project/pyVHR-master` |
| GFVC Python 环境 | `/opt/conda/envs/fv2v/bin/python` |
| pyVHR Python 环境 | `/opt/conda/envs/pyvhr/bin/python` |
| 原始 UBFC2（AVI/raw，不能改压缩格式） | `/opt/data/private/datasets/UBFC2` |
| 旧固定裁剪数据 | `/opt/data/private/datasets/UBFC2_fixed256_fv2v` |
| 480 帧候选上下文数据 | `/opt/data/private/datasets/UBFC2_PCC075_candidates` |
| 当前 CHROM 筛选数据集 | `/opt/data/private/datasets/UBFC2_CHROM_PCC075` |
| 低秩模型最佳权重 | `checkpoint/gfvc_rppg_stage2_lowrank_v2.pth.tar.best` |
| rPPG Teacher 权重 | `checkpoint/skin_rppg_teacher.pt` |
| GFVC 基础权重 | `checkpoint/00000189-checkpoint.pth.tar` |

当前 CHROM 数据集的规模：train 146、val 23、test 10 个片段；test 覆盖 4 个 subject。它适合阶段性验证，不足以作为最终论文统计量。

## 4. 规范数据集流程

完整说明和可复制命令见：`tools/UBFC_CHROM_DATASET.md`。

### 4.1 处理原则

1. 从原始 `vid.avi` 解码；不把原始视频转为有损 MP4；
2. 每个 subject 做固定 256×256 人脸裁剪，保留原始帧序；
3. train/val/test 必须按 subject 划分，禁止身份泄漏；
4. CHROM 标签使用 `60 + 360 + 60 = 480` 帧上下文提取，再只保留中间 360 帧；
5. 保留 signed zero-lag PCC，不通过时移或极性翻转提高筛选结果；
6. `rppg_signal` 保存 CHROM 波形；`reference_rppg_signal` 保留原有 BVP，供独立核查；
7. 面部皮肤掩膜与图像帧一一对应，且在候选/最终数据之间必须做像素一致性验证。

### 4.2 关键脚本

| 功能 | 文件 |
|---|---|
| 原始 UBFC2 → 固定裁剪、分割及上下文窗口 | `tools/preprocess_ubfc2.py` |
| 旧掩膜严格复用到新窗口 | `tools/reuse_ubfc_skin_masks.py` |
| 旧版 CHROM 标签筛选 | `tools/curate_ubfc_chrom.py` |
| 规范 480→中央 360 CHROM 标签构建 | `tools/curate_ubfc_chrom_context.py` |
| 最终数据集完整性/标签复现验证 | `tools/validate_ubfc_chrom_dataset.py` |
| 480 帧上下文 CHROM 预计算 | `tools/precompute_pyvhr_context_chrom.py` |
| 数据流程文档 | `tools/UBFC_CHROM_DATASET.md` |
| 数据流程测试 | `tests/test_preprocess_split_windows.py`、`tests/test_curate_ubfc_chrom*.py`、`tests/test_validate_ubfc_chrom_dataset.py` |

## 5. 训练与验证流程

### 5.1 Teacher 先验验证

先确认 Teacher 能从真实视频及对应皮肤掩膜恢复 BVP；若这一点不成立，不能把后续生成器训练结果解释为可靠 rPPG。

相关文件：

- Teacher 和损失：`modules/rppg.py`
- Teacher/模型全量检查：`tools/evaluate_stage2_full.py`
- 训练入口：`train_gfvc_rppg.py`
- 数据加载：`rppg_gfvc_dataset.py`

### 5.2 训练

训练主要以 5 秒（150 帧、约 30 fps）序列进行，以控制显存；信号预处理对 CHROM 标签应使用 `signal_preprocess=none`，避免破坏标签定义。训练目标包含视觉重建与 rPPG 一致性，生成器只在皮肤门控区域写入小幅生理颜色变化。

### 5.3 长序列测试

训练可以是 150 帧，但测试可使用 480 帧上下文；模型、pyVHR 都看完整 480 帧，最终只对中间 360 帧计算指标。这样可避免滤波边界效应，又不扩大评估目标片段。

## 6. pyVHR 独立评价规范

主评价脚本：`tools/evaluate_pyvhr_bvp.py`。

它实际调用 pyVHR `Pipeline`：

```python
Pipeline().run_on_video(
    video, roi_method='convexhull', roi_approach='patches',
    method='cpu_CHROM' or 'cpu_POS', estimate='clustering',
    patch_size=30, pre_filt=(method == 'cpu_CHROM'), post_filt=True,
    cuda=True)
```

规则：

- 每个 patch 单独 z-score，再按时刻取中位数；不根据参考标签挑选 patch；
- 主波形 PCC 是零时延、有符号 PCC；不平移波形，也不翻转 POS 极性；
- 波形 `mae_z`/`rmse_z` 是预测和目标各自 z-score 后的逐帧误差；POS/CHROM 原始振幅没有统一物理尺度，因此不报告不可比的原始振幅 MAE；
- 同时报出 DTW、SNR，以及诊断性 `hr_mae_bpm`；短片段上的 HR 指标不作为主结论；
- 汇总规则：window median → sample mean → subject mean → subject 等权总均值。

测试和可视化辅助：

| 功能 | 文件 |
|---|---|
| 480 帧生成并写出评价 manifest | `tools/render_stage2_pyvhr_context_set.py` |
| pyVHR 指标 | `tools/evaluate_pyvhr_bvp.py` |
| 波形图 | `tools/plot_pyvhr_bvp_example.py` |
| 指标单元测试 | `tests/test_pyvhr_bvp_metrics.py` |

## 7. 已完成实验与结果

### 7.1 Preserve（正常保留）

协议：10 个 `UBFC2_CHROM_PCC075/test` 样本；完整 480 帧生成和 pyVHR 提取；只统计中间 360 帧。使用 `gfvc_rppg_stage2_lowrank_v2.pth.tar.best`。

结果（按 4 个 subject 等权统计）：

| 方法 | 视频 | PCC | DTW-normalized | SNR (dB) |
|---|---|---:|---:|---:|
| CHROM | 输入 | 1.000 | ≈0 | 2.30 |
| CHROM | 生成 | 0.790 ± 0.036 | 0.320 ± 0.075 | 1.06 ± 1.44 |
| POS | 输入 | -0.982 ± 0.009 | 0.384 ± 0.067 | 1.59 ± 0.93 |
| POS | 生成 | -0.776 ± 0.034 | 0.427 ± 0.056 | 0.52 ± 1.50 |

POS 的负号是原生极性差异；可作为跨算法的绝对相关性证据（约 0.776），但不得在主指标中偷偷翻转极性。输入 CHROM=1 是数据集标签由同配置 CHROM、同 480 帧上下文构造而来的预期一致性检查，不应被表述为模型收益。

服务器结果：

```text
visualization/ubfc2_chrom_context480_test/pyvhr_summary.json
visualization/ubfc2_chrom_context480_test/pyvhr_details.json
```

### 7.2 Rewrite（随机 BVP 改写）

协议：同一 10 个测试视频，保持源人脸和运动不变；固定种子 `20261031` 对 10 条波形做无自配对随机置换，供体波形作为 `target_rppg` 注入。源/供体标签 PCC 为 `0.020 ± 0.087`，故不是源波形偶然相似造成的结果。

生成视频对供体 BVP 的独立 pyVHR 结果：

| 方法 | PCC | mae_z | rmse_z | DTW-normalized | SNR (dB) |
|---|---:|---:|---:|---:|---:|
| CHROM | 0.757 ± 0.047 | 0.552 ± 0.049 | 0.692 ± 0.069 | 0.339 ± 0.045 | -0.51 ± 1.68 |
| POS | -0.743 ± 0.047 | 1.528 ± 0.054 | 1.867 ± 0.025 | 0.440 ± 0.032 | -1.23 ± 1.75 |

CHROM 的改写 PCC 只比 preserve 低约 0.033，说明当前框架已能将与源波形近似无关的供体 BVP 写入生成视频。POS 的负号仍是原生极性。

服务器结果：

```text
visualization/ubfc2_chrom_context480_rewrite_seed20261031/manifest.json
visualization/ubfc2_chrom_context480_rewrite_seed20261031/pyvhr_summary.json
visualization/ubfc2_chrom_context480_rewrite_seed20261031/pyvhr_details.json
```

代表性改写可视化（样本 `0000_clip_0001`，CHROM PCC=0.759）：

```text
visualization/ubfc2_chrom_context480_rewrite_seed20261031/demo_0000/
├── 0000_clip_0001_source_vs_rewrite.avi
├── 0000_clip_0001_source_vs_rewrite.gif
└── 0000_clip_0001_donor_vs_generated_chrom.png
```

## 8. 复现实验命令

在服务器项目根目录执行。先生成/确认 480 帧上下文 CHROM：

```bash
PYTHONPATH=.:tools /opt/conda/envs/pyvhr/bin/python \
  tools/precompute_pyvhr_context_chrom.py \
  --target-root /opt/data/private/datasets/UBFC2_CHROM_PCC075/test \
  --context-root /opt/data/private/datasets/UBFC2_PCC075_candidates/test \
  --output-dir visualization/ubfc2_chrom_context480_test/context_signals \
  --manifest visualization/ubfc2_chrom_context480_test/context_signals.json
```

保留实验生成：

```bash
PYTHONPATH=.:tools /opt/conda/envs/fv2v/bin/python \
  tools/render_stage2_pyvhr_context_set.py \
  --checkpoint checkpoint/gfvc_rppg_stage2_lowrank_v2.pth.tar.best \
  --output-dir visualization/ubfc2_chrom_context480_test \
  --manifest visualization/ubfc2_chrom_context480_test/manifest.json \
  --context-signal-manifest visualization/ubfc2_chrom_context480_test/context_signals.json \
  --indices 0,1,2,3,4,5,6,7,8,9
```

改写实验仅需增加 `--rewrite-seed 20261031` 并换一个输出目录。最后运行 pyVHR：

```bash
PYTHONPATH=.:/opt/data/private/project/pyVHR-master /opt/conda/envs/pyvhr/bin/python \
  tools/evaluate_pyvhr_bvp.py \
  --manifest <output-dir>/manifest.json \
  --output <output-dir>/pyvhr_details.json \
  --summary-output <output-dir>/pyvhr_summary.json
```

## 9. 当前边界与下一步

当前结果是强的概念验证，但还不足以作为最终论文结论。优先级如下：

1. 在当前 CHROM 筛选分布上重新训练低秩模型，并报告训练曲线；
2. 与原始 GFVC（无生理分支）、自由 RGB 残差、不同 rank（1/2/4）做消融；
3. 强制跨 subject 供体改写，避免同一 subject 不同片段带来的潜在相似性；
4. 在有可靠 cPPG 的独立数据集上评价，避免仅依赖 CHROM 构造标签；
5. 报告视觉保真度（PSNR/SSIM/LPIPS）、身份保持和运动一致性；
6. 扩大 subject 与测试片段数，给出置信区间/显著性检验；
7. 最后再做动漫关键帧隐私传输；其接口无需改变，只替换 `I_key`。

## 10. 版本状态

- 本规范流程和 pyVHR 评价已于 Git commit `b33977e` 推送至 `main`。
- 该提交刻意不包含尚在迭代的模型、训练、可视化改动，避免把未验证工作混进规范数据/评价流程。
