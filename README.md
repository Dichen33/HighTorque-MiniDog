<p align="center">
  <img src="5.Images/V3.png" width="780" alt="HighTorque-MiniDog CAD render" />
</p>

<h1 align="center">HighTorque-MiniDog</h1>

<p align="center">
  基于 HighTorque-MiniDog 的 URDF / MJCF 机器人模型，进行 MuJoCo 仿真、MJLab 强化学习环境搭建，以及四足机器人奖惩设置与步态学习的实验项目。
</p>

<p align="center">
  <b>中文</b> | <a href="README_EN.md">English</a>
</p>

---

## 项目简介

本仓库以 HighTorque-MiniDog 的 12 自由度点足四足机器人为基础，围绕以下目标进行整理和开发：

- 将原始 URDF / 机器人资源导入 MuJoCo，维护可仿真的 MJCF 模型。
- 搭建 MJLab + RSL-RL 风格的强化学习训练框架。
- 学习和调试四足机器人强化学习中的观测维度、奖励项、惩罚项、地形课程和 play 可视化流程。
- 通过 Zh-db 提供本地训练曲线查看、历史训练记录管理和训练结果分析。

当前项目重点不是直接给出一个已经完全收敛的最终策略，而是作为 HighTorque-MiniDog 在 MuJoCo / MJLab 中进行强化学习实验、奖惩调参和训练流程管理的项目基础。

## 目录结构

```text
HighTorque-MiniDog/
├─ 1.Hardware/
│  ├─ htdw_4438_urdf/          # 原始机器人 URDF 与 mesh 资源
│  └─ htdw_4438_mjcf/          # 从 URDF 转换整理得到的 MJCF 资源
├─ mujoco/
│  ├─ htdw_4438/               # MuJoCo 单机仿真、IK 控制、pygame 控制面板
│  └─ mjlab/                   # MJLab 强化学习训练与 play 框架
├─ Zh-db/                      # 本地训练曲线网页，可查看历史训练数据
├─ scripts/                    # 根目录快捷启动脚本
├─ 2.Software/                 # 原项目软件与 SDK 资料
├─ 3.Document/                 # 参考文档
├─ 4.Paper/                    # 相关论文资料
└─ 5.Images/                   # 图片资源
```

## MuJoCo 仿真

MuJoCo 相关文件位于：

```text
mujoco/htdw_4438/
```

其中包含：

- `htdw_4438.xml`：机器人 MJCF 模型。
- `ik_control.py`：基于足端轨迹和 IK 的运动控制实验。
- `pygame_control_panel.py`：stand / trot / lay / zero 以及速度滑杆 UI。
- `stand_check.py`：站立姿态检查脚本。
- `terrain_*.xml` / `*.hfield`：地形实验资源。

## MJLab 强化学习

MJLab 强化学习框架位于：

```text
mujoco/mjlab/
```

主要内容：

- `minidog_native/`：原生 MJLab 风格任务、观测、奖励、机器人配置。
- `legged_gym/`：兼容 legged_gym 风格的 train / play 入口。
- `rsl_rl/`：RSL-RL 训练接口。
- `TRAINING_TASKS.md`：训练命令、play 命令、奖励惩罚说明。
- `OBSERVATION_GAIT_ANALYSIS.md`：观测维度和步态相关分析。
- `NATIVE_MJLAB_RSL_RL.md`：原生 MJLab + RSL-RL 训练说明。

任务配置采用类似下面的划分：

```text
Robot-Flat-v0
Robot-Rough-v0
Robot-Crawl-v0
```

地形、奖励项和课程学习尽量放在任务级配置中，而不是写死在单个 XML 文件里，方便对比平地、崎岖地形和低姿态通行等实验。

## Zh-db 训练曲线查看

Zh-db 位于：

```text
Zh-db/
```

它用于本地记录和查看训练曲线，目标是提供一个简化版的 wandb 风格界面：

- 查看历史训练曲线。
- 实时接收训练指标。
- 点击曲线查看具体坐标数值。
- 删除旧训练记录。
- 导出曲线数据为 CSV。
- 可接入 DeepSeek API 对训练结果做中文总结分析。

本地训练数据默认保存在 `Zh-db/db-data/`，导出的 CSV 默认保存在 `Zh-db/db-download/`。这些目录属于本地运行数据，默认不会提交到 Git。

## 快速开始

进入 MJLab 目录：

```bat
cd /d D:\mujoco_test\HighTorque-MiniDog\mujoco\mjlab
```

使用 uv 初始化环境：

```bat
scripts\uv_setup_native.bat
```

训练示例：

```bat
scripts\train_native.bat Robot-Rough-v0 --num_envs 2048 --max_iterations 2000 --run_name rough_v2
```

play 可视化示例：

```bat
scripts\play_native.bat Robot-Rough-v0 --run-name rough_v2 --checkpoint latest
```

更完整的训练、play、奖励项和参数说明见：

```text
mujoco/mjlab/TRAINING_TASKS.md
```

## 当前研究重点

本项目当前主要关注：

- 12 自由度串联点足机器人的默认站姿和关节零位定义。
- 四足 trot 步态学习中的相位观测、足端摆动高度、足端打滑、接触节奏和腿部镜像约束。
- rough 地形下的奖励权重、碰撞惩罚、非法接触、课程难度和训练稳定性。
- MuJoCo 后端与 Isaac Gym / Isaac Lab 风格环境之间的差异。
- 训练曲线的本地可视化、导出和 AI 总结分析。

## Git 管理说明

仓库只提交源码、模型描述文件、配置、文档和必要资源。

以下内容默认不会提交：

- Python 虚拟环境 `.venv/`
- 训练日志 `logs/`、`runs/`、`wandb/`
- 模型权重 `model/`、`*.pt`、`*.pth`
- Zh-db 本地数据 `db-data/`、`db-download/`
- 视频、缓存、临时文件和压缩备份

这样可以让 GitHub 仓库保持适合项目管理和代码协作的状态。
