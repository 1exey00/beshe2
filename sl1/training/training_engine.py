"""训练引擎模块 - CFR+SL 训练流程。

该版本移除了 Deep CFR 依赖，保留 CFR（遗憾最小化）+
监督学习（SL）策略网络拟合流程：
1. 用 CFRTrainer 在自博弈轨迹上累计遗憾与平均策略
2. 用平均策略作为监督信号训练 PolicyNetwork
"""

import signal
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from models.core import TrainingConfig, Action, ActionType, GameState
from models.networks import PolicyNetwork
from environment.poker_environment import PokerEnvironment
from environment.state_encoder import StateEncoder
from training.cfr_trainer import CFRTrainer
from utils.checkpoint_manager import CheckpointManager


class TrainingEngine:
    """CFR+SL 训练引擎。"""

    def __init__(
        self,
        config: TrainingConfig,
        checkpoint_dir: str = "checkpoints",
        tensorboard_dir: str = "runs",
        enable_tensorboard: bool = True,
        experiment_name: Optional[str] = None,
    ):
        self.config = config

        self.env = PokerEnvironment(
            initial_stack=config.initial_stack,
            small_blind=config.small_blind,
            big_blind=config.big_blind,
            max_raises_per_street=config.max_raises_per_street,
        )
        self.state_encoder = StateEncoder()

        input_dim = self.state_encoder.encoding_dim
        self.action_dim = 6

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")

        # 将网络移到GPU
        self.policy_network = PolicyNetwork(
            input_dim=input_dim,
            hidden_dims=config.network_architecture,
            action_dim=self.action_dim,
        ).to(self.device)  # ← 关键：移到GPU

        self.policy_optimizer = optim.Adam(
            self.policy_network.parameters(),
            lr=config.learning_rate,
        )

        # CFR：累计 regret_sum / strategy_sum
        self.cfr_trainer = CFRTrainer(
            num_actions=self.action_dim,
            initial_stack=config.initial_stack,
        )

        self.supervised_buffer: List[Tuple[np.ndarray, np.ndarray]] = []

        self.checkpoint_manager = CheckpointManager(checkpoint_dir)

        self.current_episode = 0
        self.total_rewards: List[float] = []
        self.win_count = 0
        self.should_stop = False

        self._setup_signal_handlers()

    def _setup_signal_handlers(self) -> None:
        def signal_handler(signum, frame):
            print("\n收到中断信号，正在保存检查点...")
            self.should_stop = True

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    def _action_to_index(self, action: Action) -> int:
        mapping = {
            ActionType.FOLD: 0,
            ActionType.CHECK: 1,
            ActionType.CALL: 2,
            ActionType.RAISE_SMALL: 3,
            ActionType.RAISE_BIG: 4,
            ActionType.RAISE: 3,
            ActionType.ALL_IN: 5,
        }
        return mapping.get(action.action_type, 0)

    def _select_action_from_cfr(self, state: GameState, player_id: int, legal_actions: List[Action]) -> Action:
        info_set = self.cfr_trainer.get_info_set(state, player_id)
        strategy = self.cfr_trainer.get_strategy(info_set)  #strategy是一个使用cfr产生的策略分布
        return self.cfr_trainer.get_action_from_strategy(strategy, legal_actions)

    def _train_policy_once(self) -> float:
        if len(self.supervised_buffer) < self.config.batch_size:
            return 0.0

        batch_size = min(self.config.batch_size, len(self.supervised_buffer))
        idx = np.random.choice(len(self.supervised_buffer), size=batch_size, replace=False)
        states = np.array([self.supervised_buffer[i][0] for i in idx], dtype=np.float32)
        targets = np.array([self.supervised_buffer[i][1] for i in idx], dtype=np.float32)

        # ========== 修改：将数据移到GPU ==========
        states_t = torch.FloatTensor(states).to(self.device)  # ← 移到GPU
        targets_t = torch.FloatTensor(targets).to(self.device)  # ← 移到GPU

        self.policy_optimizer.zero_grad()
        logits = self.policy_network(states_t)  # 网络已在GPU，自动计算
        log_probs = torch.log_softmax(logits, dim=-1)
        loss = -torch.mean(torch.sum(targets_t * log_probs, dim=-1))
        loss.backward()
        self.policy_optimizer.step()

        return float(loss.item())

    def _run_single_episode(self) -> Dict[str, float]:
        state = self.env.reset()  #重置对局
        done = False
        max_steps = 10
        step_count = 0

        transitions: List[Dict[str, Any]] = []

        while not done and step_count < max_steps:
            current_player = state.current_player
            legal_actions = self.env.get_legal_actions(state)
            if not legal_actions:
                break

            #找到合适的动作——逻辑是什么？
            action = self._select_action_from_cfr(state, current_player, legal_actions)
            action_idx = self._action_to_index(action)
            legal_indices = [self._action_to_index(a) for a in legal_actions]

            transitions.append(
                {
                    "state": state,
                    "player_id": current_player,
                    "action_idx": action_idx,
                    "legal_indices": legal_indices,
                }
            )

            state, _, done = self.env.step(action)
            step_count += 1

        # 终局收益（以初始筹码为基准）
        rewards = [
            float(state.player_stacks[0] - self.config.initial_stack),
            float(state.player_stacks[1] - self.config.initial_stack),
        ]

        for t in transitions:
            player_id = t["player_id"]
            legal_indices = t["legal_indices"]

            self.cfr_trainer.compute_and_update_regrets(
                state=t["state"],
                player_id=player_id,
                action_taken=t["action_idx"],
                reward=rewards[player_id],
                legal_action_indices=legal_indices,
            )

            target = self.cfr_trainer.get_cfr_guided_target(
                state=t["state"],
                player_id=player_id,
                legal_action_indices=legal_indices,
            )
            encoding = self.state_encoder.encode(t["state"], player_id)
            #生成数据并且加入缓冲
            self.supervised_buffer.append((encoding, target))

        utility_p0 = rewards[0]
        return {
            "utility_p0": utility_p0,
            "utility_p1": rewards[1],
            "num_info_sets": float(self.cfr_trainer.get_num_info_sets()),
        }

    def _generate_cfr_dataset(self, num_episodes: int) -> None:
        """阶段1：仅用 CFR 生成离线数据集，不进行网络训练。"""
        target_episode = self.current_episode + num_episodes
        print(f"阶段1/2: CFR 生成离线数据集，目标回合数: {num_episodes}")

        while self.current_episode < target_episode and not self.should_stop:
            metrics = self._run_single_episode()
            self.current_episode += 1

            utility = metrics.get("utility_p0", 0.0)
            self.total_rewards.append(utility)
            if utility > 0:
                self.win_count += 1

            if self.current_episode % 100 == 0:
                win_rate = self.win_count / self.current_episode
                avg_reward = float(np.mean(self.total_rewards[-100:])) if self.total_rewards else 0.0
                print(
                    f"数据生成 {self.current_episode}/{target_episode} | "
                    f"胜率: {win_rate:.2%} | 平均收益: {avg_reward:.2f} | "
                    f"info_sets: {int(metrics.get('num_info_sets', 0))} | "
                    f"dataset_size: {len(self.supervised_buffer)}"
                )

            if self.current_episode % self.config.checkpoint_interval == 0:
                self._save_checkpoint()

    def _train_policy_offline(self) -> float:
        """阶段2：仅基于离线数据集训练策略网络。"""
        if len(self.supervised_buffer) < self.config.batch_size:
            print(
                f"离线训练跳过: dataset_size={len(self.supervised_buffer)} < batch_size={self.config.batch_size}"
            )
            return 0.0

        print(
            f"阶段2/2: 基于离线数据集训练策略网络，steps={self.config.network_train_steps}, "
            f"dataset_size={len(self.supervised_buffer)}"
        )
        loss_total = 0.0
        steps = 0
        for step in range(self.config.network_train_steps):
            if self.should_stop:
                break
            loss = self._train_policy_once()
            loss_total += loss
            steps += 1
            if (step + 1) % 500 == 0:
                print(f"  SL训练步数: {step + 1}/{self.config.network_train_steps}")

        avg_loss = loss_total / max(steps, 1)
        print(f"离线训练完成，平均SL损失: {avg_loss:.6f}")
        return avg_loss

    def train(self, num_episodes: Optional[int] = None) -> Dict[str, Any]:
        if num_episodes is None:
            num_episodes = self.config.num_episodes

        print(f"开始 CFR+SL 离线训练流程，数据回合数: {num_episodes}")
        self._generate_cfr_dataset(num_episodes)
        self._train_policy_offline()
        return self._finalize_training()

    def _finalize_training(self) -> Dict[str, Any]:
        checkpoint_path = self._save_checkpoint()
        win_rate = self.win_count / self.current_episode if self.current_episode > 0 else 0.0
        avg_reward = float(np.mean(self.total_rewards)) if self.total_rewards else 0.0

        print("-" * 50)
        print("CFR+SL 训练完成！")
        print(f"总回合数: {self.current_episode}")
        print(f"最终胜率: {win_rate:.2%}")
        print(f"平均收益: {avg_reward:.2f}")
        print(f"信息集数量: {self.cfr_trainer.get_num_info_sets()}")
        print(f"离线数据集大小: {len(self.supervised_buffer)}")
        print(f"检查点: {checkpoint_path}")

        return {
            "total_episodes": self.current_episode,
            "win_rate": win_rate,
            "avg_reward": avg_reward,
            "total_rewards": self.total_rewards,
            "num_info_sets": self.cfr_trainer.get_num_info_sets(),
            "dataset_size": len(self.supervised_buffer),
        }

    def _save_checkpoint(self) -> str:
        import time
        from datetime import datetime

        timestamp = int(time.time() * 1000000)
        filename = f"checkpoint_{timestamp}_{self.current_episode}.pt"
        path = Path(self.checkpoint_manager.checkpoint_dir) / filename

        data = {
            "checkpoint_format": "cfr_sl_v1",
            "episode_number": self.current_episode,
            "timestamp": datetime.now().isoformat(),
            "win_count": self.win_count,
            "total_rewards": self.total_rewards,
            "policy_network_state_dict": self.policy_network.state_dict(),
            "policy_optimizer_state_dict": self.policy_optimizer.state_dict(),
            "cfr_regret_sum": self.cfr_trainer.regret_sum,
            "cfr_strategy_sum": self.cfr_trainer.strategy_sum,
            "cfr_iterations": self.cfr_trainer.iterations,
            "action_dim": self.action_dim,
        }
        torch.save(data, path)
        print(f"检查点已保存: {path}")
        return str(path)

    def save_checkpoint(self, path: Optional[str] = None) -> str:
        return self._save_checkpoint()

    def load_checkpoint(self, checkpoint_path: str) -> None:
        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"检查点文件不存在: {checkpoint_path}")

        data = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        fmt = data.get("checkpoint_format", "legacy")

        if fmt != "cfr_sl_v1":
            raise ValueError(f"不支持的检查点格式: {fmt}。当前仅支持 cfr_sl_v1")

        self.policy_network.load_state_dict(data["policy_network_state_dict"])
        self.policy_optimizer.load_state_dict(data["policy_optimizer_state_dict"])

        self.cfr_trainer.regret_sum = data.get("cfr_regret_sum", {})
        self.cfr_trainer.strategy_sum = data.get("cfr_strategy_sum", {})
        self.cfr_trainer.iterations = data.get("cfr_iterations", 0)

        self.current_episode = data.get("episode_number", 0)
        self.win_count = data.get("win_count", 0)
        self.total_rewards = data.get("total_rewards", [])

        print(f"已加载 CFR+SL 检查点: {checkpoint_path}")
        print(f"  当前回合: {self.current_episode}")
        print(f"  信息集数量: {self.cfr_trainer.get_num_info_sets()}")
        self.policy_network.to(self.device)
