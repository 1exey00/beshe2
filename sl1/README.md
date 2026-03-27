# 德州扑克AI训练系统

一个用于德州扑克单挑（Heads-Up）训练、评估与分析的项目。当前训练主线为 **CFR + 监督学习（SL）离线流程**：先用 CFR 生成离线数据集，再用 SL 训练策略网络。

## 项目简介

当前版本采用两阶段训练：

1. **CFR 数据生成阶段**：通过自博弈收集状态-策略监督样本（离线数据集）
2. **SL 训练阶段**：仅基于离线数据集训练 `PolicyNetwork`

这条流程保留了 CFR 的策略改进能力，同时避免了在线“边采样边更新”带来的流程耦合。

### 当前训练架构（CFR+SL）

- **CFRTrainer**：维护信息集、累积遗憾与平均策略
- **PolicyNetwork**：学习 CFR 生成的目标策略分布
- **TrainingEngine**：编排“先生成数据，再离线训练”两阶段流程

训练流程：
1. 自博弈采样一局，记录轨迹
2. 用终局收益更新 CFR 遗憾与策略统计
3. 生成 CFR 引导目标并写入离线数据集
4. 完成数据集后，批量训练策略网络

---

## 主要功能

- **离线训练**：CFR 生成数据，SL 基于数据集训练
- **完整规则环境**：合法行动校验、阶段推进、摊牌结算
- **模型评估**：对抗多种基准策略
- **策略分析**：分析模型在给定状态下的行为
- **命令行工具**：训练、评估、分析、检查点管理、抽象生成

---

## 项目结构

```text
.
├── models/          # 核心数据模型和神经网络
│   ├── core.py      # Card, Action, GameState, TrainingConfig 等
│   └── networks.py  # PolicyNetwork / RegretNetwork / ValueNetwork
├── environment/     # 游戏环境模块
│   ├── poker_environment.py # 对局推进环境（step/reset）
│   ├── cfr_environment.py   # CFR 友好环境接口（无状态转移）
│   ├── rule_engine.py       # 规则引擎
│   ├── hand_evaluator.py    # 手牌评估
│   └── state_encoder.py     # 状态编码
├── training/
│   ├── training_engine.py   # CFR+SL 离线训练引擎
│   ├── cfr_trainer.py       # CFR 信息集/遗憾/平均策略
│   └── ...
├── analysis/        # 评估与策略分析
├── utils/           # 配置、日志、检查点等
├── tests/           # 测试
└── cli.py           # 命令行入口
```

---

## 安装说明

### 环境要求

- Python 3.9+
- PyTorch 2.0+

### 安装步骤

```bash
git clone <repository-url>
cd texas-holdem-ai
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## 快速开始

### 1) 命令行训练

```bash
# 查看帮助
python3 cli.py --help

# 使用配置文件训练
python3 cli.py train --config configs/default_config.json --episodes 1000
```

### 2) 命令行评估

```bash
python3 cli.py evaluate --model checkpoints/checkpoint.pt --games 100
```

### 3) 命令行分析

```bash
python3 cli.py analyze --model checkpoints/checkpoint.pt
```

### 4) 代码调用示例

```python
from models.core import TrainingConfig
from training.training_engine import TrainingEngine

config = TrainingConfig(
    learning_rate=1e-3,
    batch_size=2048,
    num_episodes=10000,   # 数据生成回合数
    network_train_steps=4000,
    initial_stack=1000,
    small_blind=5,
    big_blind=10,
)

engine = TrainingEngine(config, checkpoint_dir="checkpoints")
result = engine.train()

print("训练完成")
print("总回合:", result["total_episodes"])
print("数据集大小:", result.get("dataset_size"))
```

---

## 命令行使用说明

### train

```bash
python3 cli.py train [选项]
```

常用选项：
- `--config, -c` 配置文件路径
- `--episodes, -e` 训练回合数（数据生成回合数）
- `--checkpoint-dir, -d` 检查点目录
- `--resume, -r` 从检查点恢复
- `--learning-rate, -lr` 学习率覆盖
- `--batch-size, -b` 批次大小覆盖

### evaluate

```bash
python3 cli.py evaluate --model <checkpoint_path> --games 100
```

### analyze

```bash
python3 cli.py analyze --model <checkpoint_path>
```

---

## 配置参数（训练相关）

| 参数 | 说明 |
|---|---|
| `learning_rate` | 策略网络学习率 |
| `batch_size` | SL 训练批次大小 |
| `num_episodes` | CFR 数据生成回合数 |
| `network_train_steps` | 离线 SL 训练步数 |
| `checkpoint_interval` | 检查点保存间隔 |
| `initial_stack/small_blind/big_blind` | 牌局盲注与筹码配置 |

> 说明：当前主流程是“离线两阶段”，即先 CFR 数据生成，再 SL 训练。

---

## 算法说明（当前实现）

### 阶段1：CFR 数据生成

- 每局根据当前 CFR 策略采样动作
- 终局后更新信息集遗憾和策略累积
- 为每个状态生成 CFR 引导目标策略分布
- 写入离线数据集（`supervised_buffer`）

### 阶段2：SL 离线训练

- 从离线数据集随机采样 mini-batch
- 用交叉熵形式（soft target）训练 `PolicyNetwork`
- 训练完成后保存 `cfr_sl_v1` 检查点

---

## 测试

```bash
# 全量
python3 -m pytest tests -v

# 关键子集
python3 -m pytest tests/test_cli.py tests/test_cfr_environment.py -v
```

---

## 技术栈

- PyTorch
- NumPy
- pytest / Hypothesis
- matplotlib（可选）

