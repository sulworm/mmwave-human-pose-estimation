# Project Agent Guide

This file is the current source of truth for this project.
At the start of any new conversation, read this file first.

## Project Goal

This project is a radar-only 3D human pose regression pipeline.

- Radar is the only model input during inference.
- IMU is used only offline as supervision and evaluation label data.
- The current repository state is the initial project line. Existing filenames are the only active baseline.

## Active Pipeline

Current main chain:

1. `Data_Matched`
2. `data_align.py`
3. `Data_Aligned`
4. `process_data.py`
5. `Dataset_Ready`
6. `train_net.py`
7. `Training_Results/best_model.pth`
8. `inference.py`
9. `Data_With_Pred`
10. `visualize.py`

Support tool:

- `viz_check_data.py`: check whether `Data_Aligned` IMU/Radar alignment looks reasonable.

## Files To Keep Active

These are the active root-level code files and should stay easy to find:

- `data_align.py`
- `process_data.py`
- `train_net.py`
- `inference.py`
- `visualize.py`
- `viz_check_data.py`

## Current Decisions

### 1. Use current filenames as the only active line

Do not create versioned filenames such as alternate-numbered or temporary copies.
Update the current root-level scripts directly.

### 2. User owns code backups

Do not create backup copies inside this project unless explicitly requested.
Dataset outputs, training outputs, and inference outputs may be overwritten by the active scripts.

### 3. Data source

`Data_Aligned` is the current training source, but the active scripts now select only `54-108,131-204` by default and exclude `5-44,111-126` at pack/inference time. Groups `5-44` are considered abandoned for the current round and should not be rescued further unless the user explicitly reopens that direction.

`Data_Matched` remains the frame-aligned source used by `data_align.py`.
Do not introduce new references to `Data_Correct`.

### 4. Training representation

The active training path uses local pose coordinates:

- estimate a planar radar center `center=(x,y,0)`
- subtract that center from radar points and IMU pose labels
- train the model to predict local 13-joint pose
- add the center back during inference to write global predictions

This is intended to reduce collapse toward a fixed global point or mean pose.

Current training also uses action-bucket balanced sampling, train-only point-cloud augmentation, joint-weighted pose loss, and extra lower-body temporal losses to reduce the "floating skeleton / static legs" failure mode.

### 5. Extra external dataset outputs are not active

The external batch under `dataset/` was synchronized and aligned experimentally, but it is not part of the active path.

Current decision:

- do not use `dataset/`
- do not use `Data_Extra_Synced`
- do not use `Data_Extra_CA`
- do not use external prediction output experiment directories
- do not mix that batch into training unless the user explicitly reopens that direction

Treat those directories as historical experiment outputs, not active training assets.

### 6. Realtime/hardware path is not active now

`Realtime_3_pm.py` is archived.

- The project currently focuses on offline data, training, inference, and evaluation.
- If hardware/realtime work resumes later, revisit archived code instead of rebuilding assumptions from memory.

## Standard Commands

Preferred Python:

```powershell
C:\Apps\anaconda3\envs\pose\python.exe
```

Default offline workflow:

```powershell
C:\Apps\anaconda3\envs\pose\python.exe process_data.py
C:\Apps\anaconda3\envs\pose\python.exe train_net.py
C:\Apps\anaconda3\envs\pose\python.exe inference.py
C:\Apps\anaconda3\envs\pose\python.exe visualize.py
```

Useful checks:

```powershell
C:\Apps\anaconda3\envs\pose\python.exe viz_check_data.py
```

## Working Rules For Future AI

When continuing this project:

1. Read `AGENT.md` first.
2. Treat current root-level files as the only active line.
3. Do not create versioned scripts unless the user explicitly asks.
4. Keep the root directory clean.
5. Put abandoned or exploratory code in `.else/archive_legacy_code/` only when archiving is explicitly part of the task.
6. If changing the active pipeline, update this file and `README.md` in the same task.

## Current Priority

Main objective:

- train on the currently aligned `Data_Aligned` dataset
- fix or reduce fixed-point / mean-pose prediction collapse
- reduce lower-body mean-pose collapse, especially walking groups such as `59`
- evaluate with group-level train/test split and visual inspection

Non-priority items right now:

- external dataset rescue
- `5-44` rotation/alignment rescue
- realtime hardware integration
- heavy SMPL/FK body-model reconstruction

## Notes

- `README.md` is the user-facing Chinese quick-start document.
- This file should be updated whenever the active pipeline or accepted datasets change.
