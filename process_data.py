import argparse
from collections import Counter
import json
import os
import random
import sys

import numpy as np
import torch
from tqdm import tqdm


SOURCE_DIR = "Data_Aligned"
SAVE_DIR = "Dataset_Ready"
SEQ_LEN = 10
NUM_POINTS = 128
SEED = 42
TEST_RATIO = 0.2
DEFAULT_GROUPS = "54-108,131-204"
DEFAULT_EXCLUDE_GROUPS = "5-44,111-126"

LOCAL_CROP_BOX = 1.5
GLOBAL_Z_RANGE = (-0.5, 2.5)
ROUGH_XY_RANGE = (-3.5, 3.5)
ROUGH_Z_RANGE = (-0.5, 2.5)


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


def list_group_ids(source_dir):
    return sort_group_ids(
        item for item in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, item))
    )


def action_bucket(group_id):
    try:
        gid = int(group_id)
    except ValueError:
        return "other"

    if 54 <= gid <= 63:
        return "walk_54_63"
    if 64 <= gid <= 73:
        return "seated_lower_motion_64_73"
    if 74 <= gid <= 83:
        return "fall_squat_74_83"
    if 84 <= gid <= 98:
        return "arm_motion_84_98"
    if 99 <= gid <= 108:
        return "bend_phone_99_108"
    if 131 <= gid <= 135:
        return "inplace_walk_131_135"
    if 136 <= gid <= 138:
        return "arm_motion_136_138"
    if 139 <= gid <= 143:
        return "sit_to_stand_139_143"
    if 144 <= gid <= 145:
        return "walk_144_145"
    if 146 <= gid <= 147:
        return "standup_146_147"
    if 148 <= gid <= 157:
        return "sit_148_157"
    if 158 <= gid <= 167:
        return "inplace_walk_158_167"
    if 168 <= gid <= 180:
        return "arm_motion_168_180"
    if 181 <= gid <= 186:
        return "standup_181_186"
    if 187 <= gid <= 190:
        return "sit_187_190"
    if 191 <= gid <= 200:
        return "random_motion_191_200"
    if 201 <= gid <= 204:
        return "arm_raise_201_204"
    return "other"


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


def split_groups(group_ids, test_ratio=TEST_RATIO, seed=SEED):
    buckets = {}
    for group_id in sort_group_ids(group_ids):
        buckets.setdefault(action_bucket(group_id), []).append(group_id)

    rng = random.Random(seed)
    train_groups = []
    test_groups = []

    for bucket_name in sorted(buckets):
        groups = list(buckets[bucket_name])
        rng.shuffle(groups)

        if len(groups) < 2:
            train_groups.extend(groups)
            continue

        test_count = max(1, int(round(len(groups) * test_ratio)))
        test_count = min(test_count, len(groups) - 1)
        test_groups.extend(groups[:test_count])
        train_groups.extend(groups[test_count:])

    return sort_group_ids(train_groups), sort_group_ids(test_groups)


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


def preprocess_frame(radar_data, imu_data, prev_center, rng):
    points_xyz = radar_data[:, :3].astype(np.float32)
    center, valid_world = estimate_center(points_xyz, prev_center=prev_center)
    radar_local = crop_local_points(valid_world - center)
    radar_sampled = sample_or_pad(radar_local, rng=rng)

    pose_global = imu_data[:, :3].astype(np.float32)
    if pose_global.shape != (13, 3):
        raise ValueError(f"Expected IMU shape (13, 3), got {pose_global.shape}")
    pose_local = (pose_global - center).astype(np.float32)
    return radar_sampled, pose_local, pose_global, center


def build_windows(items, seq_len=SEQ_LEN):
    items = np.asarray(items, dtype=np.float32)
    if items.shape[0] < seq_len:
        return np.zeros((0, seq_len) + items.shape[1:], dtype=np.float32)
    return np.asarray([items[idx : idx + seq_len] for idx in range(items.shape[0] - seq_len + 1)], dtype=np.float32)


def safe_concat(blocks, tail_shape):
    if not blocks:
        return np.zeros((0,) + tail_shape, dtype=np.float32)
    return np.concatenate(blocks, axis=0).astype(np.float32, copy=False)


def build_dataset_for_groups(source_dir, group_ids, rng):
    radar_window_blocks = []
    pose_local_window_blocks = []
    pose_global_window_blocks = []
    center_window_blocks = []
    sample_groups = []
    sample_buckets = []
    sample_start_frames = []
    group_summaries = {}

    for group_id in tqdm(group_ids, desc="Packing groups", leave=False, disable=not sys.stderr.isatty()):
        group_path = os.path.join(source_dir, group_id)
        radar_dir = os.path.join(group_path, "Radar")
        imu_dir = os.path.join(group_path, "IMU")
        if not os.path.isdir(radar_dir) or not os.path.isdir(imu_dir):
            continue

        radar_files = sorted(name for name in os.listdir(radar_dir) if name.endswith(".npy"))
        imu_files = sorted(name for name in os.listdir(imu_dir) if name.endswith(".npy"))
        frame_count = min(len(radar_files), len(imu_files))
        if frame_count < SEQ_LEN:
            continue

        frame_radars = []
        frame_pose_local = []
        frame_pose_global = []
        frame_centers = []
        prev_center = None

        for frame_idx in range(frame_count):
            radar_data = np.load(os.path.join(radar_dir, radar_files[frame_idx]))
            imu_data = np.load(os.path.join(imu_dir, imu_files[frame_idx]))
            radar_local, pose_local, pose_global, center = preprocess_frame(
                radar_data, imu_data, prev_center=prev_center, rng=rng
            )
            frame_radars.append(radar_local)
            frame_pose_local.append(pose_local)
            frame_pose_global.append(pose_global)
            frame_centers.append(center)
            prev_center = center

        radar_group_windows = build_windows(frame_radars)
        pose_local_group_windows = build_windows(frame_pose_local)
        pose_global_group_windows = build_windows(frame_pose_global)
        center_group_windows = build_windows(frame_centers)
        sample_count = radar_group_windows.shape[0]
        bucket_name = action_bucket(group_id)

        if sample_count == 0:
            continue

        radar_window_blocks.append(radar_group_windows)
        pose_local_window_blocks.append(pose_local_group_windows)
        pose_global_window_blocks.append(pose_global_group_windows)
        center_window_blocks.append(center_group_windows)
        sample_groups.extend([group_id] * sample_count)
        sample_buckets.extend([bucket_name] * sample_count)
        sample_start_frames.extend(list(range(sample_count)))
        group_summaries[group_id] = {
            "frames": int(frame_count),
            "samples": int(sample_count),
            "bucket": bucket_name,
        }

    return {
        "radar": torch.from_numpy(safe_concat(radar_window_blocks, (SEQ_LEN, NUM_POINTS, 3))),
        "pose_local": torch.from_numpy(safe_concat(pose_local_window_blocks, (SEQ_LEN, 13, 3))),
        "pose_global": torch.from_numpy(safe_concat(pose_global_window_blocks, (SEQ_LEN, 13, 3))),
        "center": torch.from_numpy(safe_concat(center_window_blocks, (SEQ_LEN, 3))),
        "groups": sample_groups,
        "sample_buckets": sample_buckets,
        "start_frames": sample_start_frames,
        "group_summaries": group_summaries,
        "config": {
            "source_dir": source_dir,
            "seq_len": SEQ_LEN,
            "num_points": NUM_POINTS,
            "seed": SEED,
            "groups": DEFAULT_GROUPS,
            "exclude_groups": DEFAULT_EXCLUDE_GROUPS,
            "label_mode": "pose_local_equals_global_pose_minus_planar_radar_center",
            "sample_mode": "repeat_points_when_sparse_zero_only_when_empty",
        },
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dir", default=SOURCE_DIR)
    parser.add_argument("--save_dir", default=SAVE_DIR)
    parser.add_argument("--test_ratio", type=float, default=TEST_RATIO)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--groups", default=DEFAULT_GROUPS, help="Included group ids/ranges, or 'all'.")
    parser.add_argument("--exclude_groups", default=DEFAULT_EXCLUDE_GROUPS, help="Excluded group ids/ranges.")
    return parser.parse_args()


def main():
    args = parse_args()
    if not os.path.exists(args.source_dir):
        print(f"错误: 找不到源文件夹 {args.source_dir}，请先运行 data_align.py。")
        return

    os.makedirs(args.save_dir, exist_ok=True)
    all_source_groups = list_group_ids(args.source_dir)
    group_ids = filter_group_ids(all_source_groups, include_spec=args.groups, exclude_spec=args.exclude_groups)
    train_groups, test_groups = split_groups(group_ids, test_ratio=args.test_ratio, seed=args.seed)
    overlap = sorted(set(train_groups) & set(test_groups))
    if overlap:
        raise RuntimeError(f"Train/test group overlap detected: {overlap}")

    print(f"数据组总数: {len(group_ids)} / 源目录组数: {len(all_source_groups)}")
    print(f"使用组范围: {args.groups} | 排除组范围: {args.exclude_groups}")
    print(f"训练组: {len(train_groups)} | 测试组: {len(test_groups)}")
    print(f"测试组列表: {test_groups}")

    rng = np.random.default_rng(args.seed)
    train_data = build_dataset_for_groups(args.source_dir, train_groups, rng)
    test_data = build_dataset_for_groups(args.source_dir, test_groups, rng)
    train_data["config"]["groups"] = args.groups
    train_data["config"]["exclude_groups"] = args.exclude_groups
    test_data["config"]["groups"] = args.groups
    test_data["config"]["exclude_groups"] = args.exclude_groups

    if train_data["radar"].shape[0] == 0:
        print("错误：没有生成训练样本，请检查 Data_Aligned。")
        return
    if test_data["radar"].shape[0] == 0:
        print("错误：没有生成测试样本，请检查分组或 SEQ_LEN。")
        return

    split_info = {
        "source_dir": args.source_dir,
        "save_dir": args.save_dir,
        "seed": args.seed,
        "test_ratio": args.test_ratio,
        "groups": args.groups,
        "exclude_groups": args.exclude_groups,
        "source_groups": all_source_groups,
        "all_groups": group_ids,
        "train_groups": train_groups,
        "test_groups": test_groups,
        "train_samples": int(train_data["radar"].shape[0]),
        "test_samples": int(test_data["radar"].shape[0]),
        "train_sample_buckets": train_data["sample_buckets"],
        "test_sample_buckets": test_data["sample_buckets"],
        "train_bucket_counts": dict(sorted(Counter(train_data["sample_buckets"]).items())),
        "test_bucket_counts": dict(sorted(Counter(test_data["sample_buckets"]).items())),
        "train_group_summaries": train_data["group_summaries"],
        "test_group_summaries": test_data["group_summaries"],
    }
    train_data["config"]["split"] = split_info
    test_data["config"]["split"] = split_info

    torch.save(train_data, os.path.join(args.save_dir, "train_data.pt"))
    torch.save(test_data, os.path.join(args.save_dir, "test_data.pt"))
    with open(os.path.join(args.save_dir, "split_info.json"), "w", encoding="utf-8") as handle:
        json.dump(split_info, handle, ensure_ascii=False, indent=2)

    print(f"训练样本: {split_info['train_samples']}")
    print(f"测试样本: {split_info['test_samples']}")
    print(f"处理完成，数据已覆盖保存至 {args.save_dir}")


if __name__ == "__main__":
    main()
