import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.widgets import Slider
import matplotlib
# 尝试设置后端以支持弹窗
try:
    matplotlib.use('TkAgg')
except:
    pass

#检查各种文件夹中的数据是否对齐，是否正确加载。存几个名字在这方便复制 Data_Aligned Data_Matched
class DataAlignedVisualizer:
    def __init__(self, data_root='Data_Aligned'):
        self.data_root = data_root
        self.imu_data = []
        self.radar_data = []

        self.is_playing = True
        self.current_idx = 0
        self.total_frames = 0

        # 图层显示状态：[Radar, Skeleton]
        # 默认都显示
        self.show_layers = [True, True]

        # 13点骨骼连接 (基于削减后的索引 0-12)
        # 你的数据对齐代码已经把数据削减成了13个点，所以索引是连续的 0-12
        # 映射关系:
        # 0:Hip, 1:R_Hip, 2:R_Knee, 3:R_Foot
        # 4:L_Hip, 5:L_Knee, 6:L_Foot
        # 7:Neck, 8:Head
        # 9:R_Shoulder, 10:R_Elbow
        # 11:L_Shoulder, 12:L_Elbow
        self.skeleton_bones = [
            (0, 7), (7, 8),  # 躯干 (Hip -> Neck -> Head)
            (0, 1), (1, 2), (2, 3),  # 右腿
            (0, 4), (4, 5), (5, 6),  # 左腿
            (7, 9), (9, 10),  # 右臂 (Neck -> Shoulder -> Elbow)
            (7, 11), (11, 12)  # 左臂
        ]

    def select_group(self):
        if not os.path.exists(self.data_root):
            print(f"错误: 找不到文件夹 {self.data_root}")
            return False

        # 列出所有组
        groups = sorted([d for d in os.listdir(self.data_root) if os.path.isdir(os.path.join(self.data_root, d))])
        # 简单的自然排序
        try:
            groups.sort(key=lambda x: int(x) if x.isdigit() else x)
        except:
            pass

        print(f"可用数据组 ({len(groups)}):")
        print(groups[:10], "...", groups[-5:])  # 只打印部分

        gid = input("请输入组名 (例如 131): ").strip()
        g_path = os.path.join(self.data_root, gid)

        if not os.path.exists(g_path):
            print("错误: 组不存在")
            return False

        try:
            r_dir = os.path.join(g_path, 'Radar')
            i_dir = os.path.join(g_path, 'IMU')

            # === 修改核心: 分别读取两个文件夹的文件列表 ===
            r_files = sorted([f for f in os.listdir(r_dir) if f.endswith('.npy')])
            i_files = sorted([f for f in os.listdir(i_dir) if f.endswith('.npy')])

            # 取最小长度，确保一一对应
            count = min(len(r_files), len(i_files))

            if count == 0:
                print("错误: 数据为空")
                return False

            print(f"正在加载组 {gid}...")
            print(f"Radar文件数: {len(r_files)}, IMU文件数: {len(i_files)} -> 取前 {count} 帧")

            # 按索引加载
            self.radar_data = [np.load(os.path.join(r_dir, r_files[i])) for i in range(count)]
            self.imu_data = [np.load(os.path.join(i_dir, i_files[i])) for i in range(count)]

            self.total_frames = count
            print("加载成功！")
            return True
        except Exception as e:
            print(f"加载过程中发生异常: {e}")
            return False

    def run(self):
        if not self.select_group(): return

        # 初始化绘图
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        plt.subplots_adjust(bottom=0.15)

        # 进度条
        ax_slider = plt.axes([0.20, 0.05, 0.60, 0.03])
        self.slider = Slider(ax_slider, 'Frame', 0, self.total_frames - 1, valinit=0, valfmt='%d')

        def on_slide(val):
            self.current_idx = int(val)

        self.slider.on_changed(on_slide)

        # 绘图对象初始化
        # 1. Radar 点云 (灰色)
        scat_radar = ax.scatter([], [], [], c='gray', s=10, alpha=0.5, label='Radar (Toggle: 1)')
        # 2. Skeleton 关节点 (红色)
        scat_imu = ax.scatter([], [], [], c='red', s=20, label='Skeleton (Toggle: 2)')
        # 3. Skeleton 连线 (红色)
        line_imu = Line3DCollection([], colors='red', linewidths=2)
        ax.add_collection(line_imu)

        # 文本信息
        txt_info = fig.text(0.05, 0.95, '', fontsize=12, fontfamily='monospace')
        txt_help = fig.text(0.05, 0.02,
                            "Controls: [Space] Pause/Play  [1] Radar On/Off  [2] Skeleton On/Off  [Left/Right] Step",
                            fontsize=10)

        # 设置坐标轴范围 (雷达坐标系)
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_zlim(-1, 3)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title("Check Alignment Result")
        ax.legend()

        def get_lines(pts):
            # pts shape: (13, 3)
            return [[pts[s], pts[e]] for s, e in self.skeleton_bones]

        def update(frame):
            if self.is_playing:
                self.current_idx = (self.current_idx + 1) % self.total_frames
                # 更新滑块但不触发回调
                self.slider.eventson = False
                self.slider.set_val(self.current_idx)
                self.slider.eventson = True

            idx = self.current_idx

            # --- 更新 Radar (Layer 1) ---
            if self.show_layers[0]:
                r = self.radar_data[idx]
                if r.shape[0] > 0:
                    scat_radar._offsets3d = (r[:, 0], r[:, 1], r[:, 2])
                else:
                    scat_radar._offsets3d = ([], [], [])
            else:
                scat_radar._offsets3d = ([], [], [])

            # --- 更新 Skeleton (Layer 2) ---
            if self.show_layers[1]:
                imu = self.imu_data[idx]  # (13, 3)
                scat_imu._offsets3d = (imu[:, 0], imu[:, 1], imu[:, 2])
                line_imu.set_segments(get_lines(imu))
            else:
                scat_imu._offsets3d = ([], [], [])
                line_imu.set_segments([])

            status = "Play" if self.is_playing else "Pause"
            layer_status = f"Radar:{'ON' if self.show_layers[0] else 'OFF'} | Skel:{'ON' if self.show_layers[1] else 'OFF'}"
            txt_info.set_text(f"Frame: {idx}/{self.total_frames}\nStatus: {status}\n{layer_status}")

        def on_key(event):
            if event.key == ' ':
                self.is_playing = not self.is_playing
            elif event.key == 'right':
                self.current_idx = (self.current_idx + 1) % self.total_frames
                self.slider.set_val(self.current_idx)
            elif event.key == 'left':
                self.current_idx = (self.current_idx - 1) % self.total_frames
                self.slider.set_val(self.current_idx)
            elif event.key == '1':
                self.show_layers[0] = not self.show_layers[0]  # 切换雷达
            elif event.key == '2':
                self.show_layers[1] = not self.show_layers[1]  # 切换骨骼

        fig.canvas.mpl_connect('key_press_event', on_key)
        ani = FuncAnimation(fig, update, interval=50, cache_frame_data=False)
        plt.show()


if __name__ == "__main__":
    viz = DataAlignedVisualizer()
    viz.run()
