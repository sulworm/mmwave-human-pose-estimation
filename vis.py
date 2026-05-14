import argparse
import os

import matplotlib

try:
    matplotlib.use("TkAgg")
except Exception:
    pass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d.art3d import Line3DCollection


try:
    matplotlib.use("TkAgg")
except Exception:
    pass


class FinalVisualizer:
    def __init__(self, data_root="Data_With_Pred_mmchain_lite_st"):
        self.data_root = data_root
        self.radar_data = None
        self.pred_data = None
        self.gt_data = None
        self.center_data = None
        self.group_id = None

        self.is_playing = True
        self.current_idx = 0
        self.total_frames = 0
        self.show_layers = [True, True, True, True]  # Radar, Pred, GT, Center

        self.skeleton_bones = [
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

    def list_groups(self):
        groups = [name for name in os.listdir(self.data_root) if os.path.isdir(os.path.join(self.data_root, name))]
        return sorted(groups, key=lambda item: int(item) if item.isdigit() else item)

    def select_group(self):
        if not os.path.exists(self.data_root):
            print(f"找不到文件夹 {self.data_root}，请先运行 inference.py")
            return False

        groups = self.list_groups()
        print("可用数据组:")
        print(groups)

        gid = input("请输入要可视化的组名 (例如 59): ").strip()
        group_path = os.path.join(self.data_root, gid)
        if not os.path.exists(group_path):
            print("组不存在。")
            return False

        try:
            self.radar_data = np.load(os.path.join(group_path, "processed_radar.npy"))
            self.pred_data = np.load(os.path.join(group_path, "prediction.npy"))
            self.gt_data = np.load(os.path.join(group_path, "gt.npy"))
            center_path = os.path.join(group_path, "center.npy")
            self.center_data = np.load(center_path) if os.path.exists(center_path) else None
            self.total_frames = min(len(self.radar_data), len(self.pred_data), len(self.gt_data))
            self.radar_data = self.radar_data[: self.total_frames]
            self.pred_data = self.pred_data[: self.total_frames]
            self.gt_data = self.gt_data[: self.total_frames]
            if self.center_data is not None:
                self.center_data = self.center_data[: self.total_frames]
            self.group_id = gid
            print(f"成功加载组 {gid}，共 {self.total_frames} 帧。")
            self.print_collapse_summary()
            return True
        except Exception as exc:
            print(f"加载数据失败: {exc}")
            return False

    def print_collapse_summary(self):
        pred_root_std = np.std(self.pred_data[:, 0, :], axis=0)
        gt_root_std = np.std(self.gt_data[:, 0, :], axis=0)
        pred_center = np.mean(self.pred_data[:, 0, :], axis=0)
        gt_center = np.mean(self.gt_data[:, 0, :], axis=0)
        print(f"Pred root std xyz: {pred_root_std}")
        print(f"GT root std xyz:   {gt_root_std}")
        print(f"Pred root mean xyz: {pred_center}")
        print(f"GT root mean xyz:   {gt_center}")

    def compute_axes(self):
        arrays = []
        for data in (self.radar_data, self.pred_data, self.gt_data):
            arr = np.asarray(data).reshape(-1, 3)
            arr = arr[np.isfinite(arr).all(axis=1)]
            arr = arr[np.any(np.abs(arr) > 1e-6, axis=1)]
            if arr.size > 0:
                arrays.append(arr)

        if not arrays:
            return (-2, 2), (-2, 2), (-0.5, 2.5)

        pts = np.vstack(arrays)
        mins = pts.min(axis=0)
        maxs = pts.max(axis=0)
        spans = np.maximum(maxs - mins, 0.5)
        margins = spans * 0.12
        return (
            (mins[0] - margins[0], maxs[0] + margins[0]),
            (mins[1] - margins[1], maxs[1] + margins[1]),
            (max(-0.8, mins[2] - margins[2]), maxs[2] + margins[2]),
        )

    def get_lines(self, points):
        return [[points[start], points[end]] for start, end in self.skeleton_bones]

    def run(self):
        if not self.select_group():
            return

        fig = plt.figure(figsize=(12, 9))
        plt.subplots_adjust(bottom=0.15)
        ax = fig.add_subplot(111, projection="3d")

        ax_slider = plt.axes([0.20, 0.05, 0.60, 0.03])
        self.slider = Slider(ax_slider, "Frame", 0, self.total_frames - 1, valinit=0, valfmt="%d")
        self.slider.on_changed(lambda val: setattr(self, "current_idx", int(val)))

        scat_radar = ax.scatter([], [], [], c="gray", s=12, alpha=0.45, label="Radar (1)")
        scat_pred = ax.scatter([], [], [], c="red", s=25, label="Pred (2)")
        line_pred = Line3DCollection([], colors="red", linewidths=2)
        ax.add_collection(line_pred)
        scat_gt = ax.scatter([], [], [], c="blue", s=25, label="GT (3)")
        line_gt = Line3DCollection([], colors="blue", linewidths=2, alpha=0.55)
        ax.add_collection(line_gt)
        scat_center = ax.scatter([], [], [], c="gold", s=65, marker="x", label="Center (4)")

        txt_info = fig.text(0.05, 0.95, "", fontsize=11, fontfamily="monospace")
        fig.text(
            0.05,
            0.02,
            "Controls: [Space] Pause  [Left/Right] Step  [J] Jump  [1/2/3/4] Toggle",
            fontsize=10,
        )

        xlim, ylim, zlim = self.compute_axes()
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_zlim(*zlim)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(f"Group {self.group_id}: Prediction vs GT")
        ax.legend()

        pred_root_std = np.std(self.pred_data[:, 0, :], axis=0).mean()
        gt_root_std = np.std(self.gt_data[:, 0, :], axis=0).mean()

        def update(_frame):
            if self.is_playing:
                self.current_idx = (self.current_idx + 1) % self.total_frames
                self.slider.eventson = False
                self.slider.set_val(self.current_idx)
                self.slider.eventson = True

            idx = self.current_idx

            if self.show_layers[0]:
                radar = self.radar_data[idx]
                mask = np.any(np.abs(radar) > 1e-6, axis=1)
                radar = radar[mask]
                scat_radar._offsets3d = (radar[:, 0], radar[:, 1], radar[:, 2]) if len(radar) else ([], [], [])
            else:
                scat_radar._offsets3d = ([], [], [])

            if self.show_layers[1]:
                pred = self.pred_data[idx]
                scat_pred._offsets3d = (pred[:, 0], pred[:, 1], pred[:, 2])
                line_pred.set_segments(self.get_lines(pred))
            else:
                scat_pred._offsets3d = ([], [], [])
                line_pred.set_segments([])

            if self.show_layers[2]:
                gt = self.gt_data[idx]
                scat_gt._offsets3d = (gt[:, 0], gt[:, 1], gt[:, 2])
                line_gt.set_segments(self.get_lines(gt))
            else:
                scat_gt._offsets3d = ([], [], [])
                line_gt.set_segments([])

            if self.show_layers[3] and self.center_data is not None:
                center = self.center_data[idx]
                scat_center._offsets3d = ([center[0]], [center[1]], [center[2]])
            else:
                scat_center._offsets3d = ([], [], [])

            pred = self.pred_data[idx]
            gt = self.gt_data[idx]
            mpjpe = np.mean(np.linalg.norm(pred - gt, axis=1))
            root_err = np.linalg.norm(pred[0] - gt[0])
            state = "Play" if self.is_playing else "Pause"
            txt_info.set_text(
                f"Frame: {idx}/{self.total_frames - 1}\n"
                f"State: {state}\n"
                f"MPJPE: {mpjpe:.3f}m\n"
                f"Root Err: {root_err:.3f}m\n"
                f"Root Std P/G: {pred_root_std:.3f}/{gt_root_std:.3f}"
            )

        def on_key(event):
            if event.key == " ":
                self.is_playing = not self.is_playing
            elif event.key == "right":
                self.current_idx = (self.current_idx + 1) % self.total_frames
                self.slider.set_val(self.current_idx)
            elif event.key == "left":
                self.current_idx = (self.current_idx - 1) % self.total_frames
                self.slider.set_val(self.current_idx)
            elif event.key in {"1", "2", "3", "4"}:
                layer_idx = int(event.key) - 1
                self.show_layers[layer_idx] = not self.show_layers[layer_idx]
            elif event.key == "j":
                self.is_playing = False
                try:
                    frame_idx = int(input("Jump to frame: "))
                    if 0 <= frame_idx < self.total_frames:
                        self.current_idx = frame_idx
                        self.slider.set_val(frame_idx)
                except ValueError:
                    pass

        fig.canvas.mpl_connect("key_press_event", on_key)
        self.anim = FuncAnimation(fig, update, interval=50, cache_frame_data=False)
        plt.show()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root",
        "--pred_dir",
        dest="data_root",
        default="Data_With_Pred",
        help="Prediction output directory produced by inference.py.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    FinalVisualizer(data_root=args.data_root).run()
