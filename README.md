# Stereo Visual Odometry with Pre-computed Depth Maps

基于 [Stereo-Visual-SLAM-Odometry](https://github.com/sakshamjindal/Stereo-Visual-SLAM-Odometry) 修改，支持使用预计算深度图的立体视觉里程计。

## 功能特点

- 使用预计算深度图，跳过传统的立体匹配和三角测量步骤
- 支持 SIFT 和 ArUco 特征检测
- 基于 P3P + RANSAC 的位姿估计
- 支持点云拼接功能
- 优化的特征检测（仅处理左图，速度提升约60%）

## 安装

```bash
# 创建环境
conda env create -f setup/environment.yml
conda activate stereovo

# 安装包
pip install -e .
```

## 快速开始

```bash
# 运行 VO 管道
python main_depth.py --config_path configs/params_entry.yaml --no-vis

# 带点云拼接
python main_depth.py --config_path configs/params_entry.yaml --concatenate-pcd --pcd-voxel-size 0.001
```

## 为新项目创建配置

### 1. 准备数据集

数据集需要按以下结构组织：

```
your_dataset/
├── {timestamp_1}/
│   ├── rect_left.jpg      # 校正后的左图
│   ├── rect_right.jpg     # 校正后的右图
│   ├── depth_meter.npy    # 深度图 (米为单位, float32)
│   ├── cloud.ply          # 点云文件 (用于拼接)
│   └── K.txt              # 相机参数
├── {timestamp_2}/
│   └── ...
└── ...
```

**K.txt 格式：**
```
fx 0 cx 0 fy cy 0 0 1
baseline
```
示例：
```
3370.2686745 0.0 2016.095913 0.0 3369.0298780000003 1553.7248455 0.0 0.0 1.0
0.20116963017260622
```

### 2. 创建配置文件

复制现有配置文件并修改：

```bash
cp configs/params_entry.yaml configs/params_your_project.yaml
```

编辑配置文件，主要修改以下内容：

```yaml
# 输出路径
output:
    path: './saved_data/your_project'

# 数据集路径
dataset:
    name: 'Project'
    path: 'C:\path\to\your_dataset'  # 修改为你的数据集路径

# 相机内参 (从 K.txt 获取)
initial:
    intrinsic_left: [[fx, 0.0, cx],
                     [0.0, fy, cy],
                     [0.0, 0.0, 1.0]]
    intrinsic_right: [[fx, 0.0, cx],
                      [0.0, fy, cy],
                      [0.0, 0.0, 1.0]]

    # 外参：基线 (从 K.txt 第二行获取)
    extrinsic: [[1.0, 0.0, 0.0, -baseline],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0]]

# 根据场景调整深度范围
geometry:
    depth_min: 0.3       # 最小有效深度 (米)
    depth_max: 50.0      # 最大有效深度 (米)

    detection:
        method: 'SIFT'   # 或 'ARUCO'
        nfeatures: 5000  # 限制特征数量 (4K图像建议5000)
```

### 3. 运行

```bash
# 基本运行（无可视化）
python main_depth.py --config_path configs/params_your_project.yaml --no-vis

# 带点云拼接
python main_depth.py --config_path configs/params_your_project.yaml --concatenate-pcd --no-vis --pcd-voxel-size 0.001

# 自定义点云输出路径
python main_depth.py --config_path configs/params_your_project.yaml --concatenate-pcd --pcd-output output.ply
```

## 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--config_path` | 配置文件路径 | `configs/params.yaml` |
| `--no-vis` | 禁用可视化窗口 | `False` |
| `--concatenate-pcd` | 拼接点云 | `False` |
| `--pcd-voxel-size` | 点云下采样体素大小 (米) | `0.01` |
| `--pcd-output` | 点云输出路径 | `{output.path}/combined_pointcloud.ply` |

## 输出文件

运行完成后，输出文件保存在配置的 `output.path` 目录：

```
saved_data/your_project/
├── svo_frames_poses.pkl     # 位姿字典 (pickle格式)
├── svo_frames_poses.txt     # 位姿 (可读文本格式)
└── combined_pointcloud.ply  # 拼接的点云 (如果启用)
```

## 配置参数说明

### 关键参数

| 参数 | 路径 | 说明 |
|------|------|------|
| `depth_min` | `geometry.depth_min` | 最小有效深度，过滤噪声 |
| `depth_max` | `geometry.depth_max` | 最大有效深度，过滤远处不可靠点 |
| `nfeatures` | `geometry.detection.nfeatures` | SIFT特征数量限制，0=无限制 |
| `deltaT` | `geometry.pnpSolver.deltaT` | 单帧最大平移量，用于异常检测 |
| `minRatio` | `geometry.pnpSolver.minRatio` | PnP最小内点比例 |
| `maxRatio` | `geometry.featureMatcher.configs.maxRatio` | 特征匹配Lowe比率测试阈值 |

### 调试选项

```yaml
debug:
    my_draw_matches: True   # 显示特征匹配可视化 (会暂停程序)
```

## 常见问题

### 1. 位姿估计失败 (INVALID)

可能原因：
- 帧间运动过大 → 减小采样间隔或增加 `deltaT`
- 特征匹配不足 → 降低 `maxRatio` (如 0.7) 或增加 `nfeatures`
- 深度无效点过多 → 检查深度图质量

### 2. 点云拼接效果差

可能原因：
- 位姿漂移累积 → 这是纯 VO 的固有问题，无回环检测
- 无效帧过多 → 检查并移除问题帧

### 3. 处理速度慢

优化建议：
- 设置 `nfeatures: 5000` 限制特征数量
- 使用 `--no-vis` 禁用可视化
- 确保使用优化后的深度模式（自动跳过右图处理）

## 项目结构

```
stereoVO/
├── model/
│   └── stereoVO_depth.py    # 主 VO 类
├── geometry/
│   ├── depth_utils.py       # 深度查询和优化的特征检测
│   ├── features.py          # SIFT 特征检测
│   ├── features_aruco.py    # ArUco 特征检测
│   └── tracking_by_matching.py  # 帧间特征跟踪
├── datasets/
│   └── Project_Dataset.py   # 数据集加载器
└── configs/
    └── loader.py            # 配置解析器
```

## 许可证

基于原始项目 [Stereo-Visual-SLAM-Odometry](https://github.com/sakshamjindal/Stereo-Visual-SLAM-Odometry) 修改。
