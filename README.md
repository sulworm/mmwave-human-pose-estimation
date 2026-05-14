# 雷达 3D 人体姿态回归项目

本项目的目标是训练一个只依赖雷达输入的 3D 人体姿态回归模型。IMU 不作为实时推理输入，只在离线阶段用于构建监督标签、做坐标对齐和评估对比。

当前仓库就是项目初版。以后以现有文件名为唯一基准，直接在当前主链路上覆盖式测试迭代，不再新增版本化脚本名。

## 当前主流程

1. `Data_Matched`：已经完成帧对齐的雷达和 IMU 数据。
2. `data_align.py`：对 IMU 标签做坐标/质心对齐，输出到 `Data_Aligned`。
3. `process_data.py`：从 `Data_Aligned` 打包训练样本，覆盖输出到 `Dataset_Ready`。
4. `train_net.py`：读取 `Dataset_Ready` 训练模型，最佳权重覆盖保存到 `Training_Results/best_model.pth`。
5. `inference.py`：只使用雷达数据做批量推理，覆盖输出到 `Data_With_Pred`。
6. `visualize.py`：读取 `Data_With_Pred` 中的结果做可视化检查。

辅助检查工具：

- `viz_check_data.py`：检查 `Data_Aligned` 中雷达点云和 IMU 骨架是否对齐合理。

## 运行环境

本项目已配置的 Python 环境：

```powershell
C:\Apps\anaconda3\envs\pose\python.exe
```

如果当前终端没有激活该环境，可以直接用完整路径运行命令：

```powershell
C:\Apps\anaconda3\envs\pose\python.exe process_data.py
C:\Apps\anaconda3\envs\pose\python.exe train_net.py
C:\Apps\anaconda3\envs\pose\python.exe inference.py
C:\Apps\anaconda3\envs\pose\python.exe visualize.py
```

## Current Implementation Note

- Default usable groups are now `54-108,131-204`.
- `process_data.py` and `inference.py` exclude `5-44,111-126` by default, even if those folders are still present.
- `Dataset_Ready/split_info.json` records per-sample action buckets for balanced training and diagnostics.
- `train_net.py` defaults to action-bucket balanced sampling, train-only point-cloud augmentation, joint-weighted pose loss, and extra lower-body temporal losses.
- `inference.py` and `visualize.py` report lower-body MPJPE, knee/foot MPJPE, leg velocity error, foot-motion ratio, and phase-delay diagnostics.
- `model.py` is the shared model definition used by both `train_net.py` and `inference.py`.
- `--model_type baseline` keeps the original PointNet + Transformer model.
- `--model_type edgeconv_anchor` enables RadarChainPoseNet-lite with EdgeConv point geometry, latent anchor tokens, spatial token mixing, and temporal token mixing.
- `--model_type mmchain_lite` enables the mmChainPose-inspired path with EdgeConv point features, fixed 3D anchor aggregation, geometry-aware chained cross-attention, and an MLP pose head.
- `--model_type mmchain_lite_st` enables the mmChainPose-lite spatio-temporal variant that flattens time and anchor tokens before Transformer mixing.
- `process_data.py --radar_channels xyzvsnr --seq_len 5` is now the default; `xyz` and `xyzv` remain available for ablation.

## 当前训练逻辑

`process_data.py` 会读取 `Data_Aligned` 当前所有可用数据组，包括 `54-108` 和已有的 `131-204` 数据。

核心处理方式：

- 按数据组切分训练/测试，避免同组相邻滑窗同时进入训练集和测试集。
- 根据雷达点云估计平面人体中心 `center=(x,y,0)`。
- 雷达点云和 IMU 标签都减去该中心，模型学习局部坐标中的人体骨架。
- 少点雷达帧用重复采样补足 128 点，只有空帧才补零。
- 额外写出 `Dataset_Ready/split_info.json`，记录训练/测试组和样本数。

`train_net.py` 默认进行快速冒烟训练：

- 默认 `8` 个 epoch。
- loss = `MPJPE + 0.2 * bone_length_loss + 0.05 * temporal_velocity_loss`。
- 训练前会计算 mean-pose baseline，用于判断模型是否只是学到平均姿态。
- 最佳模型覆盖保存到 `Training_Results/best_model.pth`。

`inference.py` 使用同样的中心估计逻辑：

- 模型预测局部骨架。
- 推理时加回每帧中心，输出全局坐标 `prediction.npy`。
- 同时保存 `prediction_local.npy`、`processed_radar_local.npy`、`center.npy`、`gt_local.npy` 方便诊断。

## 快速上手

如果需要先更新对齐数据：

```powershell
C:\Apps\anaconda3\envs\pose\python.exe data_align.py --groups 59-108
```

打包训练数据：

```powershell
C:\Apps\anaconda3\envs\pose\python.exe process_data.py
```

快速训练：

```powershell
C:\Apps\anaconda3\envs\pose\python.exe train_net.py
```

训练新网络：

```powershell
C:\Apps\anaconda3\envs\pose\python.exe train_net.py --model_type edgeconv_anchor
```

训练 mmChainPose-lite 优化网络：

```powershell
C:\Apps\anaconda3\envs\pose\python.exe process_data.py --radar_channels xyzvsnr --seq_len 5
C:\Apps\anaconda3\envs\pose\python.exe train_net.py --model_type mmchain_lite
```

推理并可视化：

```powershell
C:\Apps\anaconda3\envs\pose\python.exe inference.py
C:\Apps\anaconda3\envs\pose\python.exe visualize.py
```

优先检查这些组：`54, 59, 84, 104, 131, 168, 191, 201`。

## 数据目录说明

- `Data_Matched`：帧对齐后的输入数据。
- `Data_Aligned`：经过 `data_align.py` 处理后的标签对齐数据。
- `Dataset_Ready`：训练和测试张量数据，会被 `process_data.py` 覆盖。
- `Training_Results`：训练输出和最佳模型权重，会被 `train_net.py` 覆盖。
- `Data_With_Pred`：推理结果和可视化所需数据，会被 `inference.py` 覆盖。

## 仓库数据发布说明

本仓库当前不上传原始数据集、对齐后的数据集、打包后的训练张量，以及 `Data_With_Pred*` 推理输出目录。

原因是这些目录包含雷达点云、IMU/骨架标签、处理后的中间数据或模型推理结果，属于实验数据和派生数据。为避免在论文正式产出前暴露数据细节、实验划分和中间结果，数据集将在后续论文完成后再整理并完整上传。

当前会保留并上传 `Training_Results*` 中的训练结果，包括训练日志、loss 曲线、模型配置和最佳模型权重，用于记录不同模型版本的实验结果。

## 数据可用性记录

来自历史记录和 `Data_Matched/log.txt` 的当前判断：

- `5-44`：不能用于训练，IMU 和点云没有对齐，偏差过大。
- `54-108`：已经完成 IMU 和点云对齐，重合度比较理想；部分角度存在偏移，训练效果需要继续验证。
- `111-126`：暂不建议使用，问题类似 `5-44`，IMU 和点云坐标偏差过大，且 IMU 似乎没有剪枝。
- `131-204`：目前已确认可以用于训练，但需要注意这些数据以原地动作为主，训练时要考虑它和运动姿态数据的分布差异。

动作记录：

- `59-63`：随意走动。
- `64-68`：坐姿，上下半身均动。
- `69-73`：坐姿，仅下半身动。
- `74-78`：跌倒。
- `79-83`：下蹲。
- `84-88`：双手挥动。
- `89-93`：单手挥动。
- `94-98`：摆手再见。
- `99-103`：弯腰，以及弯腰后再起来。
- `104-108`：右手打电话。
- `131-135`：原地走路。
- `136-138`：挥手。
- `139-143`：坐下站起。
- `144-145`：走路。
- `146-147`：蹲起。
- `148` 以后：坐。
- `158-167`：原地走。
- `168-176`：单手挥手。
- `177-180`：双手挥动。
- `181-186`：蹲起。
- `187-190`：坐。
- `191-200`：随机动作。
- `201-204`：手从身体两侧向外平伸。

## 当前约定

- 当前文件名就是唯一主线，不新增版本化脚本。
- 代码备份由用户负责，项目内直接覆盖式迭代。
- 实时/部署阶段只输入雷达，IMU 只用于离线监督和评估。
- 修改主流程或 accepted 数据范围时，同步更新本 README 和 `AGENT.md`。
