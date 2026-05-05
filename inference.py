import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


SOURCE_DIR = "Data_Aligned"
OUTPUT_DIR = "Data_With_Pred"
MODEL_PATH = "Training_Results/best_model.pth"
SEQ_LEN = 10
NUM_POINTS = 128
SEED = 42
DEFAULT_GROUPS = "54-108,131-204"
DEFAULT_EXCLUDE_GROUPS = "5-44,111-126"
LOWER_BODY_JOINTS = [1, 2, 3, 4, 5, 6]
KNEE_FOOT_JOINTS = [2, 3, 5, 6]
FOOT_JOINTS = [3, 6]

LOCAL_CROP_BOX = 1.5
GLOBAL_Z_RANGE = (-0.5, 2.5)
ROUGH_XY_RANGE = (-3.5, 3.5)
ROUGH_Z_RANGE = (-0.5, 2.5)


class PointNetEncoder(nn.Module):
    def __init__(self, emb_dim=256):
        super().__init__()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, emb_dim, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(emb_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        return torch.max(x, 2)[0]


class RadarPoseNet(nn.Module):
    def __init__(self, num_joints=13, seq_len=SEQ_LEN):
        super().__init__()
        self.emb_dim = 256
        self.spatial_encoder = PointNetEncoder(emb_dim=self.emb_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, self.emb_dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.emb_dim,
            nhead=4,
            dim_feedforward=512,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)
        self.regressor = nn.Sequential(
            nn.Linear(self.emb_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_joints * 3),
        )

    def forward(self, x):
        batch_size, seq_len, channels, point_count = x.shape
        x_flat = x.reshape(batch_size * seq_len, channels, point_count)
        feats = self.spatial_encoder(x_flat)
        feats = feats.reshape(batch_size, seq_len, -1) + self.pos_embedding[:, :seq_len, :]
        temp_feats = self.transformer(feats)
        return self.regressor(temp_feats).view(batch_size, seq_len, 13, 3)


def safe_torch_load(path, map_location=None):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def sort_group_ids(group_ids):
    return sorted(group_ids, key=lambda item: int(item) if str(item).isdigit() else str(item))


def parse_group_spec(group_spec):
    if not group_spec:
        return None

    group_spec = str(group_spec).strip()
    if not group_spec or group_spec.lower() == "all":
        return None

    group_ids = []
    for chunk in group_spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue

        if "-" in chunk:
            start_text, end_text = [part.strip() for part in chunk.split("-", 1)]
            if start_text.isdigit() and end_text.isdigit():
                start = int(start_text)
                end = int(end_text)
                step = 1 if start <= end else -1
                group_ids.extend(str(group_id) for group_id in range(start, end + step, step))
                continue

        group_ids.append(chunk)

    return sort_group_ids(dict.fromkeys(group_ids))


def filter_group_ids(group_ids, include_spec=DEFAULT_GROUPS, exclude_spec=DEFAULT_EXCLUDE_GROUPS):
    group_ids = sort_group_ids(str(group_id) for group_id in group_ids)
    include = parse_group_spec(include_spec)
    exclude = set(parse_group_spec(exclude_spec) or [])

    if include is not None:
        include = set(include)
        group_ids = [group_id for group_id in group_ids if group_id in include]

    if exclude:
        group_ids = [group_id for group_id in group_ids if group_id not in exclude]

    return sort_group_ids(group_ids)


def estimate_center(points_xyz, prev_center=None, ema_alpha=0.2):
    if points_xyz.size == 0:
        center = np.zeros(3, dtype=np.float32) if prev_center is None else prev_center.astype(np.float32)
        return center, np.zeros((0, 3), dtype=np.float32)

    rough_mask = (
        (points_xyz[:, 0] >= ROUGH_XY_RANGE[0])
        & (points_xyz[:, 0] <= ROUGH_XY_RANGE[1])
        & (points_xyz[:, 1] >= ROUGH_XY_RANGE[0])
        & (points_xyz[:, 1] <= ROUGH_XY_RANGE[1])
        & (points_xyz[:, 2] >= ROUGH_Z_RANGE[0])
        & (points_xyz[:, 2] <= ROUGH_Z_RANGE[1])
    )
    valid_points = points_xyz[rough_mask]
    if valid_points.shape[0] == 0:
        valid_points = points_xyz

    current_center = valid_points.mean(axis=0).astype(np.float32)
    current_center[2] = 0.0
    if prev_center is None:
        center = current_center
    else:
        center = (ema_alpha * current_center + (1.0 - ema_alpha) * prev_center).astype(np.float32)
        center[2] = 0.0
    return center, valid_points.astype(np.float32)


def crop_local_points(local_points):
    if local_points.size == 0:
        return np.zeros((0, 3), dtype=np.float32)

    mask = (
        (local_points[:, 0] >= -LOCAL_CROP_BOX)
        & (local_points[:, 0] <= LOCAL_CROP_BOX)
        & (local_points[:, 1] >= -LOCAL_CROP_BOX)
        & (local_points[:, 1] <= LOCAL_CROP_BOX)
        & (local_points[:, 2] >= GLOBAL_Z_RANGE[0])
        & (local_points[:, 2] <= GLOBAL_Z_RANGE[1])
    )
    return local_points[mask].astype(np.float32)


def sample_or_pad(points_xyz, rng, num_points=NUM_POINTS):
    point_count = points_xyz.shape[0]
    if point_count >= num_points:
        indices = rng.choice(point_count, num_points, replace=False)
        return points_xyz[indices].astype(np.float32)
    if point_count > 0:
        extra_indices = rng.choice(point_count, num_points - point_count, replace=True)
        sampled = np.vstack([points_xyz, points_xyz[extra_indices]])
        return sampled.astype(np.float32)
    return np.zeros((num_points, 3), dtype=np.float32)


def preprocess_radar_frame(radar_data, prev_center, rng):
    points_xyz = radar_data[:, :3].astype(np.float32)
    center, valid_world = estimate_center(points_xyz, prev_center=prev_center)
    radar_local = crop_local_points(valid_world - center)
    radar_sampled = sample_or_pad(radar_local, rng=rng)
    return radar_sampled, center


def build_padded_windows(frames, seq_len=SEQ_LEN):
    frames = np.asarray(frames, dtype=np.float32)
    if frames.shape[0] == 0:
        return np.zeros((0, seq_len) + frames.shape[1:], dtype=np.float32)
    head = np.repeat(frames[0:1], seq_len - 1, axis=0)
    padded = np.concatenate([head, frames], axis=0)
    return np.asarray([padded[idx : idx + seq_len] for idx in range(frames.shape[0])], dtype=np.float32)


def array_mpjpe(pred, gt):
    return float(np.mean(np.linalg.norm(pred - gt, axis=-1)))


def subset_mpjpe(pred, gt, joints):
    return array_mpjpe(pred[:, joints, :], gt[:, joints, :])


def leg_velocity_error(pred, gt):
    if len(pred) < 2:
        return 0.0
    pred_local = pred[:, LOWER_BODY_JOINTS, :] - pred[:, :1, :]
    gt_local = gt[:, LOWER_BODY_JOINTS, :] - gt[:, :1, :]
    pred_vel = pred_local[1:] - pred_local[:-1]
    gt_vel = gt_local[1:] - gt_local[:-1]
    return array_mpjpe(pred_vel, gt_vel)


def foot_motion_ratio(pred, gt):
    if len(pred) < 2:
        return 1.0
    pred_feet = pred[:, FOOT_JOINTS, :] - pred[:, :1, :]
    gt_feet = gt[:, FOOT_JOINTS, :] - gt[:, :1, :]
    pred_amp = float(np.mean(np.linalg.norm(pred_feet[1:] - pred_feet[:-1], axis=-1)))
    gt_amp = float(np.mean(np.linalg.norm(gt_feet[1:] - gt_feet[:-1], axis=-1)))
    return pred_amp / (gt_amp + 1e-8)


def foot_speed_signal(pose):
    if len(pose) < 2:
        return np.zeros(0, dtype=np.float32)
    feet = pose[:, FOOT_JOINTS, :] - pose[:, :1, :]
    return np.linalg.norm(feet[1:] - feet[:-1], axis=-1).mean(axis=1).astype(np.float32)


def estimate_phase_delay_frames(pred, gt, max_lag=10):
    pred_signal = foot_speed_signal(pred)
    gt_signal = foot_speed_signal(gt)
    if len(pred_signal) < 3 or len(gt_signal) < 3:
        return 0

    max_lag = int(min(max_lag, len(pred_signal) - 2, len(gt_signal) - 2))
    if max_lag <= 0:
        return 0

    best_lag = 0
    best_error = float("inf")
    for lag in range(-max_lag, max_lag + 1):
        if lag > 0:
            pred_part = pred_signal[lag:]
            gt_part = gt_signal[:-lag]
        elif lag < 0:
            pred_part = pred_signal[:lag]
            gt_part = gt_signal[-lag:]
        else:
            pred_part = pred_signal
            gt_part = gt_signal
        if len(pred_part) < 3:
            continue
        error = float(np.mean((pred_part - gt_part) ** 2))
        if error < best_error:
            best_error = error
            best_lag = lag
    return int(best_lag)


def compute_group_diagnostics(prediction, gt, phase_max_lag):
    return {
        "mpjpe": array_mpjpe(prediction, gt),
        "lower_body_mpjpe": subset_mpjpe(prediction, gt, LOWER_BODY_JOINTS),
        "knee_foot_mpjpe": subset_mpjpe(prediction, gt, KNEE_FOOT_JOINTS),
        "leg_velocity_error": leg_velocity_error(prediction, gt),
        "foot_motion_ratio": foot_motion_ratio(prediction, gt),
        "phase_delay_frames": estimate_phase_delay_frames(prediction, gt, max_lag=phase_max_lag),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dir", default=SOURCE_DIR)
    parser.add_argument("--output_dir", default=OUTPUT_DIR)
    parser.add_argument("--model_path", default=MODEL_PATH)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--groups", default=DEFAULT_GROUPS, help="Included group ids/ranges, or 'all'.")
    parser.add_argument("--exclude_groups", default=DEFAULT_EXCLUDE_GROUPS, help="Excluded group ids/ranges.")
    parser.add_argument("--phase_max_lag", type=int, default=10)
    return parser.parse_args()


def run_inference():
    args = parse_args()
    if not os.path.exists(args.model_path):
        print(f"错误：找不到模型权重文件 {args.model_path}。请先运行 train_net.py。")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RadarPoseNet(seq_len=SEQ_LEN).to(device)
    model.load_state_dict(safe_torch_load(args.model_path, map_location=device))
    model.eval()

    source_groups = sort_group_ids(
        group for group in os.listdir(args.source_dir) if os.path.isdir(os.path.join(args.source_dir, group))
    )
    groups = filter_group_ids(source_groups, include_spec=args.groups, exclude_spec=args.exclude_groups)
    os.makedirs(args.output_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    diagnostics = {
        "groups": groups,
        "source_groups": source_groups,
        "groups_spec": args.groups,
        "exclude_groups": args.exclude_groups,
        "group_metrics": {},
    }

    print(f"开始推理 {len(groups)} / {len(source_groups)} 个组的数据，device={device}")
    print(f"使用组范围: {args.groups} | 排除组范围: {args.exclude_groups}")

    for group in tqdm(groups, disable=not sys.stderr.isatty()):
        source_group = os.path.join(args.source_dir, group)
        radar_dir = os.path.join(source_group, "Radar")
        imu_dir = os.path.join(source_group, "IMU")
        output_group = os.path.join(args.output_dir, group)
        os.makedirs(output_group, exist_ok=True)

        if not os.path.isdir(radar_dir) or not os.path.isdir(imu_dir):
            print(f"跳过缺少 IMU/Radar 的数据组: {source_group}")
            continue

        radar_files = sorted(name for name in os.listdir(radar_dir) if name.endswith(".npy"))
        imu_files = sorted(name for name in os.listdir(imu_dir) if name.endswith(".npy"))
        frame_count = min(len(radar_files), len(imu_files))
        if frame_count < 1:
            continue

        radar_local_frames = []
        centers = []
        gt_global = []
        gt_local = []
        prev_center = None

        for frame_idx in range(frame_count):
            radar_data = np.load(os.path.join(radar_dir, radar_files[frame_idx]))
            imu_data = np.load(os.path.join(imu_dir, imu_files[frame_idx]))[:, :3].astype(np.float32)
            radar_local, center = preprocess_radar_frame(radar_data, prev_center=prev_center, rng=rng)
            radar_local_frames.append(radar_local)
            centers.append(center)
            gt_global.append(imu_data)
            gt_local.append((imu_data - center).astype(np.float32))
            prev_center = center

        radar_local_frames = np.asarray(radar_local_frames, dtype=np.float32)
        centers = np.asarray(centers, dtype=np.float32)
        gt_global = np.asarray(gt_global, dtype=np.float32)
        gt_local = np.asarray(gt_local, dtype=np.float32)
        input_windows = build_padded_windows(radar_local_frames)
        input_tensor = torch.from_numpy(input_windows).float().permute(0, 1, 3, 2)

        predictions_local = []
        with torch.no_grad():
            for start in range(0, len(input_tensor), args.batch_size):
                batch = input_tensor[start : start + args.batch_size].to(device)
                out = model(batch)
                predictions_local.append(out[:, -1, :, :].cpu().numpy())

        prediction_local = np.concatenate(predictions_local, axis=0).astype(np.float32)
        prediction_global = (prediction_local + centers[:, None, :]).astype(np.float32)
        radar_global_frames = (radar_local_frames + centers[:, None, :]).astype(np.float32)

        np.save(os.path.join(output_group, "prediction.npy"), prediction_global)
        np.save(os.path.join(output_group, "prediction_local.npy"), prediction_local)
        np.save(os.path.join(output_group, "processed_radar.npy"), radar_global_frames)
        np.save(os.path.join(output_group, "processed_radar_local.npy"), radar_local_frames)
        np.save(os.path.join(output_group, "center.npy"), centers)
        np.save(os.path.join(output_group, "gt.npy"), gt_global)
        np.save(os.path.join(output_group, "gt_local.npy"), gt_local)

        group_diag = compute_group_diagnostics(prediction_global, gt_global, args.phase_max_lag)
        diagnostics["group_metrics"][group] = group_diag
        with open(os.path.join(output_group, "diagnostics.json"), "w", encoding="utf-8") as handle:
            json.dump(group_diag, handle, ensure_ascii=False, indent=2)

    if diagnostics["group_metrics"]:
        metric_names = [
            "mpjpe",
            "lower_body_mpjpe",
            "knee_foot_mpjpe",
            "leg_velocity_error",
            "foot_motion_ratio",
        ]
        diagnostics["summary"] = {
            name: float(np.mean([row[name] for row in diagnostics["group_metrics"].values()]))
            for name in metric_names
        }
        diagnostics["summary"]["phase_delay_frames_mean"] = float(
            np.mean([row["phase_delay_frames"] for row in diagnostics["group_metrics"].values()])
        )

    with open(os.path.join(args.output_dir, "diagnostics.json"), "w", encoding="utf-8") as handle:
        json.dump(diagnostics, handle, ensure_ascii=False, indent=2)
    print(f"推理完成，结果已覆盖保存至 {args.output_dir}")


if __name__ == "__main__":
    run_inference()
