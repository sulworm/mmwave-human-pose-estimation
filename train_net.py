import argparse
from collections import Counter
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

from model import MODEL_TYPES, build_model


DATA_DIR = "Dataset_Ready"
SAVE_DIR = "Training_Results"
SEQ_LEN = 5
BATCH_SIZE = 128
EPOCHS = 8
LEARNING_RATE = 0.001
BONE_WEIGHT = 0.2
VELOCITY_WEIGHT = 0.05
LEG_VELOCITY_WEIGHT = 0.1
LEG_ACCEL_WEIGHT = 0.02
ROOT_RELATIVE_WEIGHT = 0.05
YAW_AUG_DEG = 8.0
POINT_JITTER_STD = 0.01
POINT_DROPOUT = 0.08

LOWER_BODY_JOINTS = [1, 2, 3, 4, 5, 6]
KNEE_FOOT_JOINTS = [2, 3, 5, 6]
FOOT_JOINTS = [3, 6]
JOINT_WEIGHTS = torch.tensor(
    [1.0, 1.3, 2.2, 2.4, 1.3, 2.2, 2.4, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    dtype=torch.float32,
)

SKELETON_BONES = [
    (0, 7),
    (7, 8),
    (0, 1),
    (1, 2),
    (2, 3),
    (0, 4),
    (4, 5),
    (5, 6),
    (7, 9),
    (9, 10),
    (7, 11),
    (11, 12),
]


def safe_torch_load(path, map_location=None):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def rotate_xy_tensor(values, angle_rad):
    c = torch.cos(angle_rad)
    s = torch.sin(angle_rad)
    out = values.clone()
    x = values[..., 0].clone()
    y = values[..., 1].clone()
    out[..., 0] = x * c - y * s
    out[..., 1] = x * s + y * c
    return out


class ReadyDataset(Dataset):
    def __init__(
        self,
        pt_path,
        augment=False,
        yaw_aug_deg=YAW_AUG_DEG,
        jitter_std=POINT_JITTER_STD,
        point_dropout=POINT_DROPOUT,
    ):
        data = safe_torch_load(pt_path, map_location="cpu")
        self.radar = data["radar"]  # (N, 10, 128, 3)
        self.pose_local = data["pose_local"]  # (N, 10, 13, 3)
        self.pose_global = data["pose_global"]
        self.center = data["center"]  # (N, 10, 3)
        self.groups = data.get("groups", [])
        self.sample_buckets = data.get("sample_buckets", ["unknown"] * len(self.radar))
        self.config = data.get("config", {})
        self.input_channels = int(self.radar.shape[-1])
        self.seq_len = int(self.radar.shape[1])
        self.radar_channels = self.config.get("radar_channels", "xyz")
        self.augment = augment
        self.yaw_aug_deg = float(yaw_aug_deg)
        self.jitter_std = float(jitter_std)
        self.point_dropout = float(point_dropout)

    def __len__(self):
        return len(self.radar)

    def __getitem__(self, idx):
        radar = self.radar[idx].clone()  # (Seq, Points, 3)
        pose_local = self.pose_local[idx].clone()
        pose_global = self.pose_global[idx].clone()
        center = self.center[idx].clone()

        if self.augment:
            if self.yaw_aug_deg > 0:
                angle = torch.empty((), dtype=radar.dtype).uniform_(
                    -self.yaw_aug_deg, self.yaw_aug_deg
                ) * (torch.pi / 180.0)
                radar = rotate_xy_tensor(radar, angle)
                pose_local = rotate_xy_tensor(pose_local, angle)
                pose_global = pose_local + center.unsqueeze(1)

            valid = torch.any(torch.abs(radar[..., :3]) > 1e-6, dim=-1, keepdim=True)
            if self.jitter_std > 0:
                radar[..., :3] = radar[..., :3] + torch.randn_like(radar[..., :3]) * self.jitter_std * valid
            if self.point_dropout > 0:
                keep = (torch.rand(radar.shape[:-1], dtype=radar.dtype) > self.point_dropout).unsqueeze(-1)
                radar = radar * keep

        return radar.permute(0, 2, 1), pose_local, pose_global, center


def build_bucket_sampler(dataset):
    buckets = list(dataset.sample_buckets)
    if not buckets:
        return None
    counts = Counter(buckets)
    if len(counts) <= 1:
        return None
    weights = torch.tensor([1.0 / counts[bucket] for bucket in buckets], dtype=torch.double)
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def mpjpe(pred, gt):
    return torch.sqrt(torch.sum((pred - gt) ** 2, dim=-1) + 1e-8).mean()


def weighted_mpjpe(pred, gt, joint_weights):
    distances = torch.sqrt(torch.sum((pred - gt) ** 2, dim=-1) + 1e-8)
    weights = joint_weights.to(pred.device).view(1, 1, -1)
    return torch.sum(distances * weights) / (distances.shape[0] * distances.shape[1] * torch.sum(weights))


def subset_mpjpe(pred, gt, joint_indices):
    return mpjpe(pred[:, :, joint_indices, :], gt[:, :, joint_indices, :])


def bone_lengths(points):
    lengths = []
    for start, end in SKELETON_BONES:
        lengths.append(torch.norm(points[:, :, start, :] - points[:, :, end, :], dim=-1))
    return torch.stack(lengths, dim=-1)


def bone_length_loss(pred, gt):
    return torch.mean(torch.abs(bone_lengths(pred) - bone_lengths(gt)))


def temporal_velocity_loss(pred, gt, joint_indices=None):
    if pred.shape[1] < 2:
        return pred.new_tensor(0.0)
    if joint_indices is not None:
        pred = pred[:, :, joint_indices, :]
        gt = gt[:, :, joint_indices, :]
    pred_vel = pred[:, 1:] - pred[:, :-1]
    gt_vel = gt[:, 1:] - gt[:, :-1]
    return mpjpe(pred_vel, gt_vel)


def temporal_acceleration_loss(pred, gt, joint_indices=None):
    if pred.shape[1] < 3:
        return pred.new_tensor(0.0)
    if joint_indices is not None:
        pred = pred[:, :, joint_indices, :]
        gt = gt[:, :, joint_indices, :]
    pred_vel = pred[:, 1:] - pred[:, :-1]
    gt_vel = gt[:, 1:] - gt[:, :-1]
    pred_acc = pred_vel[:, 1:] - pred_vel[:, :-1]
    gt_acc = gt_vel[:, 1:] - gt_vel[:, :-1]
    return mpjpe(pred_acc, gt_acc)


def foot_motion_ratio(pred, gt):
    if pred.shape[1] < 2:
        return pred.new_tensor(1.0)
    pred_feet = pred[:, :, FOOT_JOINTS, :] - pred[:, :, :1, :]
    gt_feet = gt[:, :, FOOT_JOINTS, :] - gt[:, :, :1, :]
    pred_amp = torch.norm(pred_feet[:, 1:] - pred_feet[:, :-1], dim=-1).mean()
    gt_amp = torch.norm(gt_feet[:, 1:] - gt_feet[:, :-1], dim=-1).mean()
    return pred_amp / (gt_amp + 1e-8)


def root_relative_mpjpe(pred, gt):
    return mpjpe(pred - pred[:, :, :1, :], gt - gt[:, :, :1, :])


def total_loss(
    pred,
    gt,
    bone_weight=BONE_WEIGHT,
    velocity_weight=VELOCITY_WEIGHT,
    leg_velocity_weight=LEG_VELOCITY_WEIGHT,
    leg_accel_weight=LEG_ACCEL_WEIGHT,
    root_relative_weight=ROOT_RELATIVE_WEIGHT,
):
    pose_loss = weighted_mpjpe(pred, gt, JOINT_WEIGHTS)
    plain_pose_loss = mpjpe(pred, gt)
    root_loss = root_relative_mpjpe(pred, gt)
    bone_loss = bone_length_loss(pred, gt)
    vel_loss = temporal_velocity_loss(pred, gt)
    leg_vel_loss = temporal_velocity_loss(pred, gt, LOWER_BODY_JOINTS)
    leg_acc_loss = temporal_acceleration_loss(pred, gt, LOWER_BODY_JOINTS)
    loss = (
        pose_loss
        + bone_weight * bone_loss
        + velocity_weight * vel_loss
        + leg_velocity_weight * leg_vel_loss
        + leg_accel_weight * leg_acc_loss
        + root_relative_weight * root_loss
    )
    return loss, {
        "mpjpe": plain_pose_loss,
        "weighted_mpjpe": pose_loss,
        "root_relative": root_loss,
        "bone": bone_loss,
        "velocity": vel_loss,
        "leg_velocity": leg_vel_loss,
        "leg_acceleration": leg_acc_loss,
    }


def mean_pose_baseline(train_set, test_set):
    mean_pose = train_set.pose_local.float().mean(dim=(0, 1), keepdim=True)
    pred = mean_pose.expand_as(test_set.pose_local.float())
    return {
        "mean_pose_local_mpjpe": mpjpe(pred, test_set.pose_local.float()).item(),
        "mean_pose_root_relative_mpjpe": root_relative_mpjpe(pred, test_set.pose_local.float()).item(),
    }


def save_model_config(save_dir, args, train_set):
    config = {
        "model_type": args.model_type,
        "input_channels": train_set.input_channels,
        "radar_channels": train_set.radar_channels,
        "seq_len": train_set.seq_len,
        "num_joints": 13,
    }
    with open(os.path.join(save_dir, "best_model_config.json"), "w", encoding="utf-8") as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2)


def evaluate(
    model,
    loader,
    device,
    bone_weight,
    velocity_weight,
    leg_velocity_weight,
    leg_accel_weight,
    root_relative_weight,
):
    model.eval()
    values = {
        "loss": [],
        "local_mpjpe": [],
        "global_mpjpe": [],
        "root_relative_mpjpe": [],
        "root_translation_error": [],
        "lower_body_mpjpe": [],
        "knee_foot_mpjpe": [],
        "root_relative_loss": [],
        "bone_loss": [],
        "velocity_loss": [],
        "leg_velocity_loss": [],
        "leg_acceleration_loss": [],
        "foot_motion_ratio": [],
    }
    pred_roots = []
    gt_roots = []

    with torch.no_grad():
        for radar, pose_local, pose_global, center in loader:
            radar = radar.to(device)
            pose_local = pose_local.to(device)
            pose_global = pose_global.to(device)
            center = center.to(device)

            pred_local = model(radar)
            pred_global = pred_local + center.unsqueeze(2)
            loss, parts = total_loss(
                pred_local,
                pose_local,
                bone_weight,
                velocity_weight,
                leg_velocity_weight,
                leg_accel_weight,
                root_relative_weight,
            )

            values["loss"].append(loss.item())
            values["local_mpjpe"].append(mpjpe(pred_local, pose_local).item())
            values["global_mpjpe"].append(mpjpe(pred_global, pose_global).item())
            values["root_relative_mpjpe"].append(root_relative_mpjpe(pred_global, pose_global).item())
            values["root_translation_error"].append(mpjpe(pred_global[:, :, :1, :], pose_global[:, :, :1, :]).item())
            values["lower_body_mpjpe"].append(subset_mpjpe(pred_global, pose_global, LOWER_BODY_JOINTS).item())
            values["knee_foot_mpjpe"].append(subset_mpjpe(pred_global, pose_global, KNEE_FOOT_JOINTS).item())
            values["root_relative_loss"].append(parts["root_relative"].item())
            values["bone_loss"].append(parts["bone"].item())
            values["velocity_loss"].append(parts["velocity"].item())
            values["leg_velocity_loss"].append(parts["leg_velocity"].item())
            values["leg_acceleration_loss"].append(parts["leg_acceleration"].item())
            values["foot_motion_ratio"].append(foot_motion_ratio(pred_global, pose_global).item())
            pred_roots.append(pred_global[:, :, 0, :].detach().cpu())
            gt_roots.append(pose_global[:, :, 0, :].detach().cpu())

    metrics = {key: float(np.mean(vals)) for key, vals in values.items()}
    if pred_roots:
        pred_root = torch.cat(pred_roots, dim=0).reshape(-1, 3)
        gt_root = torch.cat(gt_roots, dim=0).reshape(-1, 3)
        metrics["pred_root_std_mean"] = float(pred_root.std(dim=0).mean().item())
        metrics["gt_root_std_mean"] = float(gt_root.std(dim=0).mean().item())
    return metrics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=DATA_DIR)
    parser.add_argument("--save_dir", default=SAVE_DIR)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--bone_weight", type=float, default=BONE_WEIGHT)
    parser.add_argument("--velocity_weight", type=float, default=VELOCITY_WEIGHT)
    parser.add_argument("--leg_velocity_weight", type=float, default=LEG_VELOCITY_WEIGHT)
    parser.add_argument("--leg_accel_weight", type=float, default=LEG_ACCEL_WEIGHT)
    parser.add_argument("--root_relative_weight", type=float, default=ROOT_RELATIVE_WEIGHT)
    parser.add_argument("--model_type", choices=MODEL_TYPES, default="baseline")
    parser.add_argument("--no_balanced_sampling", action="store_true")
    parser.add_argument("--no_augment", action="store_true")
    parser.add_argument("--yaw_aug_deg", type=float, default=YAW_AUG_DEG)
    parser.add_argument("--point_jitter_std", type=float, default=POINT_JITTER_STD)
    parser.add_argument("--point_dropout", type=float, default=POINT_DROPOUT)
    return parser.parse_args()


def train():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    train_path = os.path.join(args.data_dir, "train_data.pt")
    test_path = os.path.join(args.data_dir, "test_data.pt")
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print(f"错误：找不到数据集，请先运行 process_data.py。缺少目录: {args.data_dir}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_set = ReadyDataset(
        train_path,
        augment=not args.no_augment,
        yaw_aug_deg=args.yaw_aug_deg,
        jitter_std=args.point_jitter_std,
        point_dropout=args.point_dropout,
    )
    test_set = ReadyDataset(test_path, augment=False)
    sampler = None if args.no_balanced_sampling else build_bucket_sampler(train_set)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=sampler is None, sampler=sampler)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    bucket_counts = Counter(train_set.sample_buckets)
    if bucket_counts:
        print(f"Train action buckets: {dict(sorted(bucket_counts.items()))}")
    print(f"Balanced sampling: {'off' if sampler is None else 'on'} | Augment: {'off' if args.no_augment else 'on'}")
    print(
        f"Model: {args.model_type} | "
        f"Radar channels: {train_set.radar_channels} ({train_set.input_channels}) | "
        f"seq_len={train_set.seq_len}"
    )

    baseline = mean_pose_baseline(train_set, test_set)
    print(f"Mean-pose baseline local MPJPE: {baseline['mean_pose_local_mpjpe']:.4f}m")
    print(f"Mean-pose baseline root-relative MPJPE: {baseline['mean_pose_root_relative_mpjpe']:.4f}m")

    model = build_model(
        model_type=args.model_type,
        input_channels=train_set.input_channels,
        seq_len=train_set.seq_len,
        num_joints=13,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    history = {
        "baseline": baseline,
        "epochs": [],
        "config": vars(args)
        | {
            "device": str(device),
            "input_channels": train_set.input_channels,
            "radar_channels": train_set.radar_channels,
            "seq_len": train_set.seq_len,
        },
    }
    best_local_mpjpe = float("inf")

    print(f"开始训练: device={device}, epochs={args.epochs}, train={len(train_set)}, test={len(test_set)}")
    for epoch in range(args.epochs):
        model.train()
        train_losses = []
        train_mpjpe = []
        loop = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{args.epochs}",
            leave=False,
            disable=not sys.stderr.isatty(),
        )

        for radar, pose_local, _pose_global, _center in loop:
            radar = radar.to(device)
            pose_local = pose_local.to(device)
            optimizer.zero_grad()
            pred_local = model(radar)
            loss, parts = total_loss(
                pred_local,
                pose_local,
                args.bone_weight,
                args.velocity_weight,
                args.leg_velocity_weight,
                args.leg_accel_weight,
                args.root_relative_weight,
            )
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            train_mpjpe.append(parts["mpjpe"].item())
            loop.set_postfix(loss=f"{loss.item():.4f}", mpjpe=f"{parts['mpjpe'].item():.4f}")

        scheduler.step()
        metrics = evaluate(
            model,
            test_loader,
            device,
            args.bone_weight,
            args.velocity_weight,
            args.leg_velocity_weight,
            args.leg_accel_weight,
            args.root_relative_weight,
        )
        train_loss = float(np.mean(train_losses))
        train_local_mpjpe = float(np.mean(train_mpjpe))
        epoch_record = {"epoch": epoch + 1, "train_loss": train_loss, "train_local_mpjpe": train_local_mpjpe, **metrics}
        history["epochs"].append(epoch_record)

        print(
            f"Epoch {epoch + 1} | "
            f"train_loss={train_loss:.4f} | train_local={train_local_mpjpe:.4f}m | "
            f"test_local={metrics['local_mpjpe']:.4f}m | "
            f"test_root_rel={metrics['root_relative_mpjpe']:.4f}m | "
            f"lower={metrics['lower_body_mpjpe']:.4f}m | "
            f"knee_foot={metrics['knee_foot_mpjpe']:.4f}m | "
            f"foot_ratio={metrics['foot_motion_ratio']:.3f} | "
            f"pred_root_std={metrics.get('pred_root_std_mean', 0.0):.4f} | "
            f"gt_root_std={metrics.get('gt_root_std_mean', 0.0):.4f}"
        )

        if metrics["local_mpjpe"] < best_local_mpjpe:
            best_local_mpjpe = metrics["local_mpjpe"]
            torch.save(model.state_dict(), os.path.join(args.save_dir, "best_model.pth"))
            save_model_config(args.save_dir, args, train_set)
            print("  >>> Best model saved")

        with open(os.path.join(args.save_dir, "training_log.json"), "w", encoding="utf-8") as handle:
            json.dump(history, handle, ensure_ascii=False, indent=2)

    epochs = history["epochs"]
    plt.figure(figsize=(10, 6))
    plt.plot([row["train_local_mpjpe"] for row in epochs], label="Train Local MPJPE")
    plt.plot([row["local_mpjpe"] for row in epochs], label="Test Local MPJPE")
    plt.axhline(baseline["mean_pose_local_mpjpe"], color="gray", linestyle="--", label="Mean Pose Baseline")
    plt.xlabel("Epoch")
    plt.ylabel("Error (meters)")
    plt.title("Training Loss Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, "loss_curve.png"))
    plt.close()
    print(f"训练结束，结果已覆盖保存至 {args.save_dir}")


if __name__ == "__main__":
    train()
