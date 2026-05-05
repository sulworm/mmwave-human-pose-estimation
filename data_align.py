import argparse
import math
import os

import numpy as np
from tqdm import tqdm


def _normalize_degrees(angle):
    return (float(angle) + 180.0) % 360.0 - 180.0


def _rotation_matrix_x(degrees):
    radians = math.radians(float(degrees))
    c = math.cos(radians)
    s = math.sin(radians)
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, c, -s],
            [0.0, s, c],
        ],
        dtype=np.float32,
    )


def _rotation_matrix_y(degrees):
    radians = math.radians(float(degrees))
    c = math.cos(radians)
    s = math.sin(radians)
    return np.array(
        [
            [c, 0.0, s],
            [0.0, 1.0, 0.0],
            [-s, 0.0, c],
        ],
        dtype=np.float32,
    )


def _rotation_matrix_z(degrees):
    radians = math.radians(float(degrees))
    c = math.cos(radians)
    s = math.sin(radians)
    return np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def euler_rotation_matrix(rot_x=0.0, rot_y=0.0, rot_z=0.0):
    # Apply X pitch first, then Y roll, then Z yaw.
    return _rotation_matrix_z(rot_z) @ _rotation_matrix_y(rot_y) @ _rotation_matrix_x(rot_x)


def parse_group_spec(group_spec):
    if not group_spec:
        return None

    group_ids = []
    for chunk in str(group_spec).split(","):
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

    return group_ids


class DataCenterAlignerSmoothNoZ:
    IMU_INDICES = [0, 1, 2, 3, 5, 6, 7, 13, 14, 16, 18, 20, 22]
    DEFAULT_RADAR_ROTATION_GROUPS = "5-44"

    def __init__(
        self,
        source_dir=None,
        target_dir=None,
        radar_rotation_mode="none",
        radar_rotation_groups=None,
        fixed_rotation=(0.0, 0.0, 0.0),
        rotation_pivot="auto",
        rotation_translation="auto",
        auto_yaw=True,
        rotation_sample_count=180,
        pitch_min=-89.0,
        pitch_max=89.0,
        pitch_step=1.0,
        yaw_min_gain=0.20,
        yaw_min_spread=0.08,
        centroid_filter_radius=1.5,
        smooth_window=12,
    ):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.source_dir = source_dir if source_dir else os.path.join(base_dir, "Data_Matched")
        self.target_dir = target_dir if target_dir else os.path.join(base_dir, "Data_Aligned")

        self.radar_rotation_mode = radar_rotation_mode
        self.radar_rotation_groups = set(
            parse_group_spec(radar_rotation_groups or self.DEFAULT_RADAR_ROTATION_GROUPS) or []
        )
        self.fixed_rotation = tuple(float(value) for value in fixed_rotation)
        self.rotation_pivot = str(rotation_pivot)
        self.rotation_translation = str(rotation_translation)
        self.auto_yaw = bool(auto_yaw)
        self.rotation_sample_count = max(5, int(rotation_sample_count))
        self.pitch_min = float(pitch_min)
        self.pitch_max = float(pitch_max)
        self.pitch_step = max(0.1, float(pitch_step))
        self.yaw_min_gain = float(yaw_min_gain)
        self.yaw_min_spread = float(yaw_min_spread)
        self.centroid_filter_radius = float(centroid_filter_radius)
        self.smooth_window = max(1, int(smooth_window))

        self.available_groups = []
        self._find_available_groups()

    def _sort_group_ids(self, group_ids):
        try:
            return sorted(group_ids, key=lambda x: int(x) if str(x).isdigit() else str(x))
        except TypeError:
            return sorted(group_ids)

    def select_groups(self, group_ids):
        wanted = {str(group_id) for group_id in group_ids}
        available = set(self.available_groups)
        missing = self._sort_group_ids(wanted - available)
        if missing:
            print(f"Warning: requested groups not found and will be skipped: {missing}")
        self.available_groups = [group for group in self.available_groups if group in wanted]

    def prepare_imu_core(self, imu_raw, group_id, frame_name):
        if imu_raw.ndim != 2 or imu_raw.shape[1] < 3:
            raise ValueError(f"Invalid IMU shape in group {group_id}, frame {frame_name}: {imu_raw.shape}")

        if imu_raw.shape[0] == len(self.IMU_INDICES):
            return imu_raw[:, :3].copy()

        if imu_raw.shape[0] > max(self.IMU_INDICES):
            return imu_raw[self.IMU_INDICES, :3].copy()

        raise ValueError(
            f"IMU frame has {imu_raw.shape[0]} joints, expected 13 pre-cut joints or at least "
            f"{max(self.IMU_INDICES) + 1} raw joints: group {group_id}, frame {frame_name}"
        )

    def _find_available_groups(self):
        if not os.path.exists(self.source_dir):
            print(f"Error: source directory does not exist: {self.source_dir}")
            return

        for item in os.listdir(self.source_dir):
            group_path = os.path.join(self.source_dir, item)
            imu_dir = os.path.join(group_path, "IMU")
            radar_dir = os.path.join(group_path, "Radar")
            if os.path.isdir(group_path) and os.path.isdir(imu_dir) and os.path.isdir(radar_dir):
                self.available_groups.append(item)
            elif os.path.isdir(group_path):
                print(f"Skip non-standard data group: {group_path}")

        self.available_groups = self._sort_group_ids(self.available_groups)
        print(f"Found {len(self.available_groups)} groups to process")

    def _needs_legacy_imu_rotation(self, group_id):
        try:
            return int(group_id) >= 158
        except ValueError:
            return False

    def _needs_radar_rotation(self, group_id):
        if self.radar_rotation_mode == "none":
            return False
        return str(group_id) in self.radar_rotation_groups

    def _resolve_rotation_pivot_mode(self):
        if self.rotation_pivot != "auto":
            return self.rotation_pivot
        if self.radar_rotation_mode == "fixed":
            return "frame_centroid"
        return "origin"

    def _wants_rotation_translation(self, pivot_mode):
        if self.rotation_translation == "none":
            return False
        if self.rotation_translation == "xy":
            return True
        return pivot_mode in {"origin", "group_centroid", "imu_group_centroid"}

    def _apply_legacy_imu_rotation(self, imu_raw):
        imu_rotated = imu_raw.copy()
        old_x = imu_rotated[:, 0].copy()
        old_y = imu_rotated[:, 1].copy()
        imu_rotated[:, 0] = -old_y
        imu_rotated[:, 1] = old_x
        return imu_rotated

    def get_centroid(self, points):
        if points.shape[0] == 0:
            return np.zeros(3, dtype=np.float32)
        return np.mean(points, axis=0)

    def get_robust_centroid(self, points):
        if points.shape[0] == 0:
            return np.zeros(3, dtype=np.float32)

        finite_mask = np.isfinite(points).all(axis=1)
        finite_points = points[finite_mask]
        if finite_points.shape[0] == 0:
            return np.zeros(3, dtype=np.float32)

        if finite_points.shape[0] < 8:
            return np.mean(finite_points, axis=0).astype(np.float32)

        median = np.median(finite_points, axis=0)
        distances = np.linalg.norm(finite_points[:, :2] - median[:2], axis=1)
        keep = distances <= np.percentile(distances, 85)
        if not np.any(keep):
            return np.mean(finite_points, axis=0).astype(np.float32)
        return np.mean(finite_points[keep], axis=0).astype(np.float32)

    def get_radar_alignment_centroid(self, radar_xyz, imu_centroid):
        if radar_xyz.shape[0] == 0:
            return np.zeros(3, dtype=np.float32)

        radius = self.centroid_filter_radius
        mask = (
            (np.abs(radar_xyz[:, 0] - imu_centroid[0]) <= radius)
            & (np.abs(radar_xyz[:, 1] - imu_centroid[1]) <= radius)
        )
        filtered = radar_xyz[mask]
        if filtered.shape[0] > 0:
            return self.get_centroid(filtered).astype(np.float32)

        return self.get_robust_centroid(radar_xyz)

    def smooth_offsets(self, offsets):
        if len(offsets) < self.smooth_window or self.smooth_window <= 1:
            return offsets

        smoothed = np.zeros_like(offsets)
        kernel = np.ones(self.smooth_window, dtype=np.float32) / self.smooth_window
        for axis in range(3):
            smoothed[:, axis] = np.convolve(offsets[:, axis], kernel, mode="same")

        half_win = self.smooth_window // 2
        smoothed[:half_win] = smoothed[half_win]
        smoothed[-half_win:] = smoothed[-half_win - 1]
        return smoothed

    def _estimate_group_pivot(self, samples, pivot_mode):
        if pivot_mode == "group_centroid":
            centers = [self.get_robust_centroid(radar_xyz) for radar_xyz in samples["radar_frames"]]
            return np.median(np.asarray(centers, dtype=np.float32), axis=0).astype(np.float32)
        if pivot_mode == "imu_group_centroid":
            return np.median(samples["imu_centers_3d"], axis=0).astype(np.float32)
        return None

    def _get_sample_rotation_pivot(self, radar_xyz, imu_center_3d, pivot_mode, group_pivot=None):
        if pivot_mode == "origin":
            return None
        if pivot_mode == "frame_centroid":
            return self.get_robust_centroid(radar_xyz)
        if pivot_mode == "imu_frame_centroid":
            return imu_center_3d.astype(np.float32)
        if pivot_mode in {"group_centroid", "imu_group_centroid"}:
            return group_pivot
        return None

    def _get_frame_rotation_pivot(self, radar_xyz, imu_13, pivot_mode, group_pivot=None):
        imu_center_3d = self.get_centroid(imu_13[:, :3]).astype(np.float32)
        return self._get_sample_rotation_pivot(radar_xyz, imu_center_3d, pivot_mode, group_pivot)

    def _transform_xyz(self, radar_xyz, rotation_matrix, pivot=None):
        if pivot is None:
            return radar_xyz @ rotation_matrix.T
        return (radar_xyz - pivot) @ rotation_matrix.T + pivot

    def transform_radar(self, radar_raw, rotation_matrix, translation=None, pivot=None):
        if rotation_matrix is None or radar_raw.shape[0] == 0:
            return radar_raw

        transformed = radar_raw.copy()
        transformed_xyz = self._transform_xyz(radar_raw[:, :3].astype(np.float32), rotation_matrix, pivot)
        if translation is not None:
            transformed_xyz = transformed_xyz + translation
        transformed[:, :3] = transformed_xyz.astype(radar_raw.dtype, copy=False)
        return transformed

    def _sample_frame_indices(self, count):
        sample_count = min(count, self.rotation_sample_count)
        if sample_count <= 0:
            return np.array([], dtype=int)
        return np.linspace(0, count - 1, sample_count, dtype=int)

    def _load_rotation_samples(self, group_id, imu_dir, radar_dir, imu_files, radar_files, count):
        imu_centers = []
        imu_centers_3d = []
        radar_frames = []
        imu_z_values = []
        radar_z_values = []

        for frame_idx in self._sample_frame_indices(count):
            imu_raw = np.load(os.path.join(imu_dir, imu_files[frame_idx]))
            radar_raw = np.load(os.path.join(radar_dir, radar_files[frame_idx]))

            if self._needs_legacy_imu_rotation(group_id):
                imu_raw = self._apply_legacy_imu_rotation(imu_raw)

            try:
                imu_13 = self.prepare_imu_core(imu_raw, group_id, imu_files[frame_idx])
            except ValueError as exc:
                print(exc)
                return None

            radar_xyz = radar_raw[:, :3].astype(np.float32)
            if radar_xyz.shape[0] == 0:
                continue

            imu_centers.append(np.mean(imu_13[:, :2], axis=0))
            imu_centers_3d.append(np.mean(imu_13[:, :3], axis=0))
            radar_frames.append(radar_xyz)
            imu_z_values.extend(imu_13[:, 2].tolist())
            radar_z_values.extend(radar_xyz[:, 2].tolist())

        if not radar_frames or not imu_centers:
            return None

        return {
            "imu_centers": np.asarray(imu_centers, dtype=np.float32),
            "imu_centers_3d": np.asarray(imu_centers_3d, dtype=np.float32),
            "radar_frames": radar_frames,
            "imu_z": np.asarray(imu_z_values, dtype=np.float32),
            "radar_z": np.asarray(radar_z_values, dtype=np.float32),
        }

    def _score_pitch(self, samples, pitch_degrees, pivot_mode, group_pivot=None):
        pitch_matrix = _rotation_matrix_x(pitch_degrees)
        rotated_z = []
        for frame_idx, radar_xyz in enumerate(samples["radar_frames"]):
            pivot = self._get_sample_rotation_pivot(
                radar_xyz,
                samples["imu_centers_3d"][frame_idx],
                pivot_mode,
                group_pivot,
            )
            rotated = self._transform_xyz(radar_xyz, pitch_matrix, pivot)
            rotated_z.extend(rotated[:, 2].tolist())

        rotated_z = np.asarray(rotated_z, dtype=np.float32)
        imu_z = samples["imu_z"]
        if rotated_z.shape[0] < 5 or imu_z.shape[0] < 5:
            return np.inf, 0.0

        radar_quantiles = np.percentile(rotated_z, [10, 50, 90])
        imu_quantiles = np.percentile(imu_z, [10, 50, 90])

        radar_height = radar_quantiles[2] - radar_quantiles[0]
        imu_height = imu_quantiles[2] - imu_quantiles[0]
        valid_z_ratio = float(np.mean((rotated_z >= -0.5) & (rotated_z <= 2.5)))
        score = abs(radar_quantiles[1] - imu_quantiles[1])
        score += 0.35 * abs(radar_height - imu_height)
        if valid_z_ratio < 0.85:
            score += (0.85 - valid_z_ratio) * 2.0
        return float(score), valid_z_ratio

    def _estimate_pitch(self, samples, pivot_mode, group_pivot=None):
        best_pitch = 0.0
        best_score = np.inf
        best_valid_z = 0.0

        pitch = self.pitch_min
        while pitch <= self.pitch_max + 1e-6:
            score, valid_z_ratio = self._score_pitch(samples, pitch, pivot_mode, group_pivot)
            if score < best_score:
                best_score = score
                best_pitch = pitch
                best_valid_z = valid_z_ratio
            pitch += self.pitch_step

        return float(best_pitch), float(best_score), float(best_valid_z)

    def _fit_planar_yaw(self, radar_centers, imu_centers):
        finite = np.isfinite(radar_centers).all(axis=1) & np.isfinite(imu_centers).all(axis=1)
        radar_centers = radar_centers[finite]
        imu_centers = imu_centers[finite]
        if radar_centers.shape[0] < 5:
            return 0.0, {
                "enabled": False,
                "reason": "too_few_frames",
                "gain": 0.0,
                "error": np.inf,
                "base_error": np.inf,
                "radar_spread": 0.0,
                "imu_spread": 0.0,
            }

        radar_centered = radar_centers - np.mean(radar_centers, axis=0)
        imu_centered = imu_centers - np.mean(imu_centers, axis=0)
        radar_spread = float(np.sqrt(np.mean(np.sum(radar_centered**2, axis=1))))
        imu_spread = float(np.sqrt(np.mean(np.sum(imu_centered**2, axis=1))))

        covariance = radar_centered.T @ imu_centered
        try:
            u, _, vt = np.linalg.svd(covariance)
        except np.linalg.LinAlgError:
            return 0.0, {
                "enabled": False,
                "reason": "svd_failed",
                "gain": 0.0,
                "error": np.inf,
                "base_error": np.inf,
                "radar_spread": radar_spread,
                "imu_spread": imu_spread,
            }

        rotation_2d = vt.T @ u.T
        if np.linalg.det(rotation_2d) < 0:
            vt[-1, :] *= -1
            rotation_2d = vt.T @ u.T

        rotated = radar_centered @ rotation_2d.T + np.mean(imu_centers, axis=0)
        translated_only = radar_centers + (np.mean(imu_centers, axis=0) - np.mean(radar_centers, axis=0))
        error = float(np.median(np.linalg.norm(rotated - imu_centers, axis=1)))
        base_error = float(np.median(np.linalg.norm(translated_only - imu_centers, axis=1)))
        gain = (base_error - error) / (base_error + 1e-9)
        yaw = _normalize_degrees(math.degrees(math.atan2(rotation_2d[1, 0], rotation_2d[0, 0])))

        enabled = (
            self.auto_yaw
            and gain >= self.yaw_min_gain
            and radar_spread >= self.yaw_min_spread
            and imu_spread >= self.yaw_min_spread
        )
        reason = "ok" if enabled else "low_confidence"
        return (yaw if enabled else 0.0), {
            "enabled": enabled,
            "reason": reason,
            "gain": float(gain),
            "error": error,
            "base_error": base_error,
            "radar_spread": radar_spread,
            "imu_spread": imu_spread,
            "raw_yaw": yaw,
        }

    def _estimate_radar_translation(self, samples, rotation_matrix, pivot_mode, group_pivot=None):
        if not self._wants_rotation_translation(pivot_mode):
            return np.zeros(3, dtype=np.float32)

        radar_centers = []
        for frame_idx, radar_xyz in enumerate(samples["radar_frames"]):
            pivot = self._get_sample_rotation_pivot(
                radar_xyz,
                samples["imu_centers_3d"][frame_idx],
                pivot_mode,
                group_pivot,
            )
            rotated_xyz = self._transform_xyz(radar_xyz, rotation_matrix, pivot)
            radar_centers.append(self.get_robust_centroid(rotated_xyz)[:2])

        radar_centers = np.asarray(radar_centers, dtype=np.float32)
        imu_centers = samples["imu_centers"]
        finite = np.isfinite(radar_centers).all(axis=1) & np.isfinite(imu_centers).all(axis=1)
        if not np.any(finite):
            return np.zeros(3, dtype=np.float32)

        xy_translation = np.median(imu_centers[finite] - radar_centers[finite], axis=0)
        return np.array([xy_translation[0], xy_translation[1], 0.0], dtype=np.float32)

    def _estimate_auto_radar_rotation(self, group_id, imu_dir, radar_dir, imu_files, radar_files, count):
        samples = self._load_rotation_samples(group_id, imu_dir, radar_dir, imu_files, radar_files, count)
        if not samples:
            return None, None, {"mode": "auto", "enabled": False, "reason": "no_samples"}

        pivot_mode = self._resolve_rotation_pivot_mode()
        group_pivot = self._estimate_group_pivot(samples, pivot_mode)

        pitch, pitch_score, valid_z_ratio = self._estimate_pitch(samples, pivot_mode, group_pivot)
        pitch_matrix = _rotation_matrix_x(pitch)

        rotated_radar_centers = []
        for frame_idx, radar_xyz in enumerate(samples["radar_frames"]):
            pivot = self._get_sample_rotation_pivot(
                radar_xyz,
                samples["imu_centers_3d"][frame_idx],
                pivot_mode,
                group_pivot,
            )
            rotated_xyz = self._transform_xyz(radar_xyz, pitch_matrix, pivot)
            rotated_radar_centers.append(self.get_robust_centroid(rotated_xyz)[:2])
        rotated_radar_centers = np.asarray(rotated_radar_centers, dtype=np.float32)

        yaw, yaw_report = self._fit_planar_yaw(rotated_radar_centers, samples["imu_centers"])
        rotation_matrix = euler_rotation_matrix(rot_x=pitch, rot_y=0.0, rot_z=yaw)
        translation = self._estimate_radar_translation(samples, rotation_matrix, pivot_mode, group_pivot)
        report = {
            "mode": "auto",
            "enabled": True,
            "pitch": float(pitch),
            "yaw": float(yaw),
            "pivot_mode": pivot_mode,
            "group_pivot": group_pivot.tolist() if group_pivot is not None else None,
            "translation": translation.tolist(),
            "pitch_score": float(pitch_score),
            "valid_z_ratio": float(valid_z_ratio),
            "yaw_report": yaw_report,
        }
        return rotation_matrix, translation, report

    def _get_radar_rotation(self, group_id, imu_dir, radar_dir, imu_files, radar_files, count):
        if not self._needs_radar_rotation(group_id):
            return None, None, {"mode": self.radar_rotation_mode, "enabled": False, "reason": "group_not_selected"}

        if self.radar_rotation_mode == "fixed":
            rot_x, rot_y, rot_z = self.fixed_rotation
            rotation_matrix = euler_rotation_matrix(rot_x=rot_x, rot_y=rot_y, rot_z=rot_z)
            samples = self._load_rotation_samples(group_id, imu_dir, radar_dir, imu_files, radar_files, count)
            pivot_mode = self._resolve_rotation_pivot_mode()
            group_pivot = self._estimate_group_pivot(samples, pivot_mode) if samples else None
            translation = (
                self._estimate_radar_translation(samples, rotation_matrix, pivot_mode, group_pivot)
                if samples
                else np.zeros(3, dtype=np.float32)
            )
            return rotation_matrix, translation, {
                "mode": "fixed",
                "enabled": True,
                "rot_x": float(rot_x),
                "rot_y": float(rot_y),
                "rot_z": float(rot_z),
                "pivot_mode": pivot_mode,
                "group_pivot": group_pivot.tolist() if group_pivot is not None else None,
                "translation": translation.tolist(),
            }

        if self.radar_rotation_mode == "auto":
            return self._estimate_auto_radar_rotation(group_id, imu_dir, radar_dir, imu_files, radar_files, count)

        return None, None, {"mode": self.radar_rotation_mode, "enabled": False, "reason": "disabled"}

    def process_group(self, group_id):
        source_group = os.path.join(self.source_dir, str(group_id))
        target_group = os.path.join(self.target_dir, str(group_id))
        source_imu_dir = os.path.join(source_group, "IMU")
        source_radar_dir = os.path.join(source_group, "Radar")
        if not os.path.isdir(source_imu_dir) or not os.path.isdir(source_radar_dir):
            print(f"Skip group missing IMU/Radar: {source_group}")
            return

        os.makedirs(os.path.join(target_group, "IMU"), exist_ok=True)
        os.makedirs(os.path.join(target_group, "Radar"), exist_ok=True)

        imu_files = sorted([f for f in os.listdir(source_imu_dir) if f.endswith(".npy")])
        radar_files = sorted([f for f in os.listdir(source_radar_dir) if f.endswith(".npy")])
        count = min(len(imu_files), len(radar_files))
        if count == 0:
            print(f"Skip empty group: {source_group}")
            return

        radar_rotation, radar_translation, rotation_report = self._get_radar_rotation(
            group_id, source_imu_dir, source_radar_dir, imu_files, radar_files, count
        )
        if rotation_report.get("enabled"):
            if rotation_report["mode"] == "auto":
                yaw_info = rotation_report["yaw_report"]
                print(
                    f"Group {group_id}: radar auto rotation "
                    f"pitch={rotation_report['pitch']:.1f}, yaw={rotation_report['yaw']:.1f}, "
                    f"pivot={rotation_report['pivot_mode']}, "
                    f"shift=({rotation_report['translation'][0]:.2f}, {rotation_report['translation'][1]:.2f}), "
                    f"z_valid={rotation_report['valid_z_ratio']:.2f}, "
                    f"yaw_gain={yaw_info.get('gain', 0.0):.2f}, yaw={yaw_info.get('reason')}"
                )
            else:
                print(
                    f"Group {group_id}: radar fixed rotation "
                    f"x={rotation_report['rot_x']:.1f}, y={rotation_report['rot_y']:.1f}, "
                    f"z={rotation_report['rot_z']:.1f}, pivot={rotation_report['pivot_mode']}, "
                    f"shift=({rotation_report['translation'][0]:.2f}, {rotation_report['translation'][1]:.2f})"
                )

        need_legacy_imu_rotation = self._needs_legacy_imu_rotation(group_id)
        pivot_mode = rotation_report.get("pivot_mode", "origin")
        group_pivot = rotation_report.get("group_pivot")
        group_pivot = np.asarray(group_pivot, dtype=np.float32) if group_pivot is not None else None
        raw_offsets = []
        imu_data_cache = []
        radar_data_cache = []

        for frame_idx in range(count):
            imu_raw = np.load(os.path.join(source_imu_dir, imu_files[frame_idx]))
            radar_raw = np.load(os.path.join(source_radar_dir, radar_files[frame_idx]))

            if need_legacy_imu_rotation:
                imu_raw = self._apply_legacy_imu_rotation(imu_raw)

            try:
                imu_13 = self.prepare_imu_core(imu_raw, group_id, imu_files[frame_idx])
            except ValueError as exc:
                print(exc)
                return

            radar_pivot = None
            if radar_rotation is not None:
                radar_pivot = self._get_frame_rotation_pivot(
                    radar_raw[:, :3].astype(np.float32),
                    imu_13,
                    pivot_mode,
                    group_pivot,
                )
            radar_aligned = self.transform_radar(radar_raw, radar_rotation, radar_translation, radar_pivot)
            imu_data_cache.append(imu_13)
            radar_data_cache.append(radar_aligned)

            if radar_aligned.shape[0] > 0:
                imu_xyz = imu_13[:, :3]
                radar_xyz = radar_aligned[:, :3]
                c_imu = self.get_centroid(imu_xyz)
                c_radar = self.get_radar_alignment_centroid(radar_xyz, c_imu)
                diff = c_radar - c_imu
                diff[2] = 0.0
                raw_offsets.append(diff.astype(np.float32))
            else:
                raw_offsets.append(np.zeros(3, dtype=np.float32))

        raw_offsets = np.asarray(raw_offsets, dtype=np.float32)
        smoothed_offsets = self.smooth_offsets(raw_offsets)

        for frame_idx in range(count):
            offset = smoothed_offsets[frame_idx]
            imu_final = imu_data_cache[frame_idx].copy()
            imu_final[:, 0] += offset[0]
            imu_final[:, 1] += offset[1]

            np.save(os.path.join(target_group, "IMU", imu_files[frame_idx]), imu_final)
            np.save(os.path.join(target_group, "Radar", radar_files[frame_idx]), radar_data_cache[frame_idx])

    def run(self):
        print("=== Start data alignment ===")
        print(
            "Logic: read -> optional radar rotation -> legacy >=158 IMU xy rotation -> "
            "13-joint trim -> XY centroid alignment, keep IMU Z -> smooth -> save"
        )
        if self.radar_rotation_mode != "none":
            print(
                f"Radar rotation mode={self.radar_rotation_mode}, "
                f"groups={self._sort_group_ids(self.radar_rotation_groups)}, "
                f"pivot={self.rotation_pivot}, translation={self.rotation_translation}"
            )

        for group in tqdm(self.available_groups):
            self.process_group(group)

        print(f"Done. Aligned data saved to: {self.target_dir}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dir", default=None)
    parser.add_argument("--target_dir", default=None)
    parser.add_argument("--groups", default=None, help="Group ids or ranges, for example: 59 or 5-44")
    parser.add_argument(
        "--radar_rotation",
        default="none",
        choices=["auto", "fixed", "none"],
        help="Radar point-cloud rotation correction mode. Auto only affects --rotation_groups.",
    )
    parser.add_argument(
        "--rotation_groups",
        default=DataCenterAlignerSmoothNoZ.DEFAULT_RADAR_ROTATION_GROUPS,
        help="Groups that should receive radar rotation correction.",
    )
    parser.add_argument("--rot_x", type=float, default=0.0, help="Fixed radar rotation around X in degrees.")
    parser.add_argument("--rot_y", type=float, default=0.0, help="Fixed radar rotation around Y in degrees.")
    parser.add_argument("--rot_z", type=float, default=0.0, help="Fixed radar rotation around Z in degrees.")
    parser.add_argument(
        "--rotation_pivot",
        default="auto",
        choices=[
            "auto",
            "origin",
            "frame_centroid",
            "group_centroid",
            "imu_frame_centroid",
            "imu_group_centroid",
        ],
        help="Pivot used for radar rotation. Auto uses frame_centroid for fixed mode and origin for auto mode.",
    )
    parser.add_argument(
        "--rotation_translation",
        default="auto",
        choices=["auto", "xy", "none"],
        help="Optional group-level XY translation after rotation. Auto skips it for frame pivots.",
    )
    parser.add_argument("--no_auto_yaw", action="store_true", help="Disable auto yaw after auto pitch estimation.")
    parser.add_argument("--rotation_samples", type=int, default=180, help="Max sampled frames for auto rotation.")
    parser.add_argument("--pitch_min", type=float, default=-89.0)
    parser.add_argument("--pitch_max", type=float, default=89.0)
    parser.add_argument("--pitch_step", type=float, default=1.0)
    parser.add_argument("--yaw_min_gain", type=float, default=0.20)
    parser.add_argument("--yaw_min_spread", type=float, default=0.08)
    parser.add_argument("--centroid_filter_radius", type=float, default=1.5)
    parser.add_argument("--smooth_window", type=int, default=12)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    aligner = DataCenterAlignerSmoothNoZ(
        source_dir=args.source_dir,
        target_dir=args.target_dir,
        radar_rotation_mode=args.radar_rotation,
        radar_rotation_groups=args.rotation_groups,
        fixed_rotation=(args.rot_x, args.rot_y, args.rot_z),
        rotation_pivot=args.rotation_pivot,
        rotation_translation=args.rotation_translation,
        auto_yaw=not args.no_auto_yaw,
        rotation_sample_count=args.rotation_samples,
        pitch_min=args.pitch_min,
        pitch_max=args.pitch_max,
        pitch_step=args.pitch_step,
        yaw_min_gain=args.yaw_min_gain,
        yaw_min_spread=args.yaw_min_spread,
        centroid_filter_radius=args.centroid_filter_radius,
        smooth_window=args.smooth_window,
    )
    selected_groups = parse_group_spec(args.groups)
    if selected_groups:
        aligner.select_groups(selected_groups)
    aligner.run()
