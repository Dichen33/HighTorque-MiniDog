# Zh-db 训练可视化面板

这是一个简化版 W&B 风格的本地训练面板，用 WebSocket 通信展示不同训练次数的曲线，并支持一键生成中文训练分析。

## 启动

```powershell
cd D:\mujoco_test\HighTorque-MiniDog\Zh-db
python server.py --host 127.0.0.1 --port 8765
```

浏览器打开：

```text
http://127.0.0.1:8765
```

页面默认以“训练轮次 iteration”为横轴显示曲线。底层 JSON 仍保留
`step/sample_step`，用于和 SB3、TensorBoard 或历史数据兼容；前端右上角可以切换为“样本”横轴。

也可以直接运行：

```powershell
D:\mujoco_test\HighTorque-MiniDog\Zh-db\start_viewer.ps1
```

这个面板不要求训练正在运行。训练结束后，历史数据仍然保存在
`Zh-db\db-data\runs`，平时只要启动 `server.py` 或 `start_viewer.ps1`，
就能在网页上查看以前的训练曲线和总结。

## 对接 MJLab 强化学习训练

先启动 Zh-db：

```powershell
cd D:\mujoco_test\HighTorque-MiniDog\Zh-db
python server.py --host 127.0.0.1 --port 8765
```

再启动 MJLab 训练：

```powershell
cd D:\mujoco_test\HighTorque-MiniDog\mujoco\mjlab
python -m minidog_rl.scripts.train_sb3 --total-steps 1000000 --num-envs 4 --run-name residual_v1
```

训练脚本会自动把 PPO 和环境指标实时写入：

```text
Zh-db\db-data\runs\residual_v1.json
```

MJLab 训练端默认每个 PPO rollout 写入一个点：

```text
iteration = sample_step / (num_envs * steps_per_env)
```

网页会通过 WebSocket 收到 `run_updated` 消息并实时刷新曲线。训练结束后，
同一个 JSON 文件仍然留在本地，所以平时不训练也能打开网页看历史曲线和总结。

## 数据格式

每一次训练对应一个 JSON 文件，放在：

```text
Zh-db/db-data/runs/
```

示例：

```json
{
  "id": "run_003",
  "name": "第三次训练：横移奖励增强",
  "created_at": "2026-08-18 00:20",
  "config": {"算法": "PPO", "环境": "MiniDog residual IK"},
  "metrics": [
    {"step": 0, "mean_reward": 10.0, "error_vel_xy": 1.8},
    {"step": 100, "mean_reward": 42.0, "error_vel_xy": 1.2}
  ]
}
```

支持的常用指标：

- Train：`mean_reward`、`mean_episode_length`
- Policy：`policy_mean_std`
- Perf：`total_fps`、`learning_time`、`collection_time`
- Metrics：`wheel_roll_error_mean`、`error_vel_yaw`、`error_vel_xy`
- Loss：`loss_value`、`loss_surrogate`、`loss_learning_rate`、`loss_entropy`
- Episode Termination：`termination_time_out`、`termination_nan_detection`、`termination_base_ground_contact`、`termination_bad_orientation`
- Episode Reward：`reward_wheel_roll_tracking`、`reward_track_lin_vel`、`reward_wheel_contact_bonus`、`reward_leg_motion_penalty`、`reward_joint_torques`、`reward_action_rate`、`reward_joint_pos_limits`
- Curriculum：`curriculum_max`

网页不会只显示上面这些固定指标。它会像 W&B 一样扫描当前 run 的
`metrics` 数组，只要某个字段是数值并且不是 `step`，就会自动生成曲线。
未知字段会按名字自动分组，例如：

- `reward_*` 会进入奖励分项
- `termination_*` 会进入终止原因
- `loss_*` 会进入损失
- `curriculum_*` 或包含 `terrain/difficulty` 的字段会进入课程学习
- 包含 `error/tracking/vel` 的字段会进入控制指标

## DeepSeek 分析

如果设置了环境变量，会优先调用 DeepSeek：

```powershell
$env:DEEPSEEK_API_KEY="你的 key"
python server.py
```

不设置 key 时，会使用内置规则分析，仍然能指出奖励、速度跟踪、姿态、终止原因、探索度等问题。

总结功能分两层：

1. 本地规则分析：直接读取当前 run 的尾段均值、首尾变化、奖励/惩罚分项、
   termination 分项、loss、policy std、课程难度等，生成健康分、优势、风险和建议。
2. DeepSeek 增强：如果配置了 `DEEPSEEK_API_KEY`，服务端会把尾段曲线和本地规则摘要发给 DeepSeek，
   让它生成更自然的中文工程诊断；如果调用失败，会自动回退到本地规则分析。

## 导出 CSV 与数据真实性

网页右上角点击“导出 CSV”后，Zh-db 服务端会从本地原始记录重新读取当前 run 的曲线数据：

```text
Zh-db\db-data\runs\<run_id>.json
```

并导出到：

```text
D:\mujoco_test\HighTorque-MiniDog\Zh-db\db-download
```

每次导出会生成两个文件：

- `<run_id>_<filter>_<time>.csv`：曲线图使用的原始数值表
- `<run_id>_<filter>_<time>.meta.json`：源 JSON 路径、源 JSON SHA256、CSV SHA256、行列数、导出时间

从新版训练开始，每条实时写入的 metric 还会带：

```text
prev_record_hash
record_hash
```

这形成一条简单哈希链。导出的 meta 中如果 `hash_chain_status` 为 `valid`，说明本地 JSON 中每条记录之间的链式校验能对上；如果旧数据没有这些字段，则会显示 `not_available`。

需要注意：本地文件系统上的数据不能在密码学意义上证明“绝对没有被人伪造”，除非训练进程、日志服务和文件系统都是可信并且使用外部签名/只追加存储。但当前链路可以证明页面曲线、导出 CSV、保存的 JSON 三者一致，并可通过训练终端输出、模型文件时间戳、checkpoint 回放结果互相交叉验证。
