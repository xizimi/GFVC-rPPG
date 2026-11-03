# GFVC-rPPG checkpoint 目录与区别

2026-11-03起统一记录。相对路径均以本项目根目录为基准。

| 位置 | 架构/用途 | 数据与资格 |
|---|---|---|
| `checkpoint/00000189-checkpoint.pth.tar` | 原GFVC SPADE，预训练epoch189；KP/HE/主干初始化来源 | 本任务冻结，不是本次训练结果 |
| `checkpoint/gfvc_rppg_stage2_skin.pth.tar[.best]` | 历史RGB生理分支 | 历史划分，不能直接视为当前被试独立验证结果 |
| `checkpoint/gfvc_rppg_stage2_lowrank_v2.pth.tar[.best]` | 历史rank2 RGB低秩调制；best内部epoch26 | 已审计出与当前UBFC验证集5个被试重叠；未作为新生理权重初始化 |
| `checkpoint/skin_rppg_teacher.pt` | 历史Skin teacher | 旧划分包含当前验证被试；新实验不用 |
| `checkpoint/ubfc_11_3_subject_independent_20261103/teacher/teacher_best.pt` | 当前独立划分的固定Skin teacher | UBFC Train30人717段；13轮早停，最佳被试平均波形PCC0.86798；新实验复用并冻结 |
| `checkpoint/ubfc_11_3_subject_independent_20261103/branch/branch_best.pth.tar` | 已完成旧RGB residual基线，23轮早停，最佳epoch15 | 当前UBFC独立训练，preserve-only；LGI仅跨数据集；已有指标属于接触PPG条件生成 |
| `checkpoint/motion_physio_20261103_v2/*_smoke_*` | 完整窗口训练/恢复/接口检查权重 | 只有少量优化步，不是正式结果 |
| `checkpoint/motion_physio_20261103_v2/feature_motion_aligned_stage1_seed42_rank2/` | 新运动对齐特征调制 | 新时序编码器+新modulator；原GFVC/teacher冻结；UBFC训练，rewrite_probability0.3 |
| `checkpoint/motion_physio_20261103_v2/feature_unaligned_stage1_seed42_rank2/` | 同结构但生理基底不Warp | 与aligned同参数量/初始化seed/数据/损失/改写设置的受控消融 |
| `checkpoint/motion_physio_20261103_v2/rgb_residual_stage1_seed42_rank2/` | 重新训练的匹配RGB对照 | 使用与新结构相同的0.3改写概率，区别于此前preserve-only RGB结果 |
| `checkpoint/motion_physio_20261103_v2/feature_motion_aligned_stage2_up1_seed42_rank2/` | aligned基础上的可选局部联合微调 | 从Stage1最佳权重初始化，只额外训练SPADE up_1，lr1e-5；未改善则保留Stage1 |

**目录存在不代表训练完成**。正式完成必须有 `TRAINING_COMPLETE.json`；当前运行阶段看 `experiment_progress.json`，异常看 `experiment_failure.json`。
smoke完成标记为 `SMOKE_COMPLETE.json`，不与正式完成混用。

## 每个新checkpoint的说明

例如 `epoch_0012.pth.tar` 必须配套：

- `epoch_0012.pth.tar.json`：机器可读的SHA256、架构模式、rank、训练阶段、epoch/全局步数、初始化父权重、原GFVC和固定teacher、
  原始数据与标签定义、精确被试列表/索引哈希、训练和冻结参数、优化器、验证选模分数、逐样本诊断。
- `epoch_0012.pth.tar.md`：便于快速辨认的中文项目索引对应的人类可读说明。
- `run_config.json`：一次训练唯一的完整参数和来源记录。
- `code_snapshot/`：该次训练实际代码副本及对应哈希。
- `checkpoint_index.json`：该run的全部epoch文件、分数和当时是否最佳。
- `best.pth.tar` / `latest.pth.tar`：仅链接到实际epoch文件；解释实际权重时应解析到对应epoch sidecar。

训练保存了模型、优化器、epoch、stale count、随机数状态、配置、完整验证记录；同模式同阶段严格恢复。
Stage1→Stage2被明确记录为父权重初始化，重新创建优化器，不能冒充同一训练过程的无缝resume。

## 如何比较

无生理Baseline使用原始GFVC权重，不训练一个空分支。
先比较本目录内的匹配RGB、feature_unaligned、feature_motion_aligned，保持UBFC Train/Validation、teacher、seed42、rank2、0.3改写和损失不变。
历史preserve-only的CSV属于另一配置，不能作为这组架构消融唯一的公平对照。
新正式训练固定验证28段preserve +28段rewrite；rewrite donor只能来自Train接触PPG。
LGI和Test不参与训练或选模，LGI仍须标记estimated_sync。

评估时明确区分“输入同段接触PPG的条件生成/保留”和“输入指定不同BVP的改写”。
teacher内部相关性、Green波形诊断、正式pyVHR指标和视觉指标分开报告，不能相互替代。

原始GFVC SHA256：`fbad01d46f0510276dc4521322dde6824a873a4222cd0740c85762e7067ea71d`。
已完成独立RGB最佳权重 SHA256：`9a4d7a4ad43218c29db3f21c7bc034be07b862709e21cd7689593d0626958a8a`。

原旧划分审计记录：`/opt/data/private/project/GFVC-PLAN/gfvc_rppg_lowrank_v2_best_audit_20261103.json`。
更详细网络/命令/测试与研究边界见 `MOTION_PHYSIO_ARCHITECTURE.md`。

## 当前审查版本

正式修订运行目录为 `checkpoint/motion_physio_20261103_v2/`。
初版 `checkpoint/motion_physio_20261103/` 保留检查与中断记录；中断原因及最后状态见其 `TRAINING_INTERRUPTED.json`。
本次审查包含已实现的架构、测试和训练记录机制。新架构完整训练及最终性能比较尚未完成。
