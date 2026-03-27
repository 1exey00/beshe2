"""数据日志器模块 - 用于持久化训练数据和结果。

该模块提供训练回合数据的写入、读取、查询和导出功能。
使用JSON Lines格式存储数据，每行一个JSON对象。
"""

import json
import csv
import os
from datetime import datetime
from typing import List, Dict, Any, Optional, Callable
from dataclasses import asdict

from models.core import Episode, GameState, Action, ActionType, GameStage, Card


class DataLogger:
    """数据日志器 - 管理训练数据的持久化存储。
    
    Attributes:
        log_dir: 日志文件存储目录
        log_file: 当前日志文件路径
    """
    
    def __init__(self, log_dir: str = "logs"):
        """初始化数据日志器。
        
        Args:
            log_dir: 日志文件存储目录，默认为"logs"
        """
        self.log_dir = log_dir
        self.log_file = os.path.join(log_dir, "training_log.jsonl")
        
        # 确保日志目录存在
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
    
    def _serialize_card(self, card: Card) -> Dict[str, Any]:
        """序列化Card对象为字典。"""
        return {"rank": card.rank, "suit": card.suit}
    
    def _serialize_action(self, action: Action) -> Dict[str, Any]:
        """序列化Action对象为字典。"""
        return {
            "action_type": action.action_type.value,
            "amount": action.amount
        }
    
    def _serialize_game_state(self, state: GameState) -> Dict[str, Any]:
        """序列化GameState对象为字典。"""
        return {
            "player_hands": [
                [self._serialize_card(card) for card in hand]
                for hand in state.player_hands
            ],
            "community_cards": [
                self._serialize_card(card) for card in state.community_cards
            ],
            "pot": state.pot,
            "player_stacks": state.player_stacks,
            "current_bets": state.current_bets,
            "button_position": state.button_position,
            "stage": state.stage.value,
            "action_history": [
                self._serialize_action(action) for action in state.action_history
            ],
            "current_player": state.current_player
        }
    
    def _serialize_episode(self, episode: Episode) -> Dict[str, Any]:
        """序列化Episode对象为字典。"""
        return {
            "states": [self._serialize_game_state(s) for s in episode.states],
            "actions": [self._serialize_action(a) for a in episode.actions],
            "rewards": episode.rewards,
            "player_id": episode.player_id,
            "final_reward": episode.final_reward
        }

    def _deserialize_card(self, data: Dict[str, Any]) -> Card:
        """反序列化字典为Card对象。"""
        return Card(rank=data["rank"], suit=data["suit"])
    
    def _deserialize_action(self, data: Dict[str, Any]) -> Action:
        """反序列化字典为Action对象。"""
        return Action(
            action_type=ActionType(data["action_type"]),
            amount=data["amount"]
        )
    
    def _deserialize_game_state(self, data: Dict[str, Any]) -> GameState:
        """反序列化字典为GameState对象。"""
        player_hands = [
            tuple(self._deserialize_card(card) for card in hand)
            for hand in data["player_hands"]
        ]
        community_cards = [
            self._deserialize_card(card) for card in data["community_cards"]
        ]
        action_history = [
            self._deserialize_action(action) for action in data["action_history"]
        ]
        
        return GameState(
            player_hands=player_hands,
            community_cards=community_cards,
            pot=data["pot"],
            player_stacks=data["player_stacks"],
            current_bets=data["current_bets"],
            button_position=data["button_position"],
            stage=GameStage(data["stage"]),
            action_history=action_history,
            current_player=data["current_player"]
        )
    
    def _deserialize_episode(self, data: Dict[str, Any]) -> Episode:
        """反序列化字典为Episode对象。"""
        states = [self._deserialize_game_state(s) for s in data["states"]]
        actions = [self._deserialize_action(a) for a in data["actions"]]
        
        return Episode(
            states=states,
            actions=actions,
            rewards=data["rewards"],
            player_id=data["player_id"],
            final_reward=data["final_reward"]
        )
    
    def write_episode(self, episode: Episode, episode_number: int) -> None:
        """写入回合数据到日志文件。
        
        使用JSON Lines格式，每条记录包含timestamp、episode_number和数据。
        
        Args:
            episode: 要写入的Episode对象
            episode_number: 回合编号
            
        Raises:
            IOError: 当磁盘空间不足或写入失败时
        """
        record = {
            "timestamp": datetime.now().isoformat(),
            "episode_number": episode_number,
            "data": self._serialize_episode(episode)
        }
        
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        except OSError as e:
            raise IOError(f"写入日志文件失败: {e}")
    
    def read_episodes(self) -> List[Dict[str, Any]]:
        """读取所有历史回合数据。
        
        Returns:
            包含所有记录的列表，每条记录包含timestamp、episode_number和Episode对象
            
        Raises:
            FileNotFoundError: 当日志文件不存在时
        """
        if not os.path.exists(self.log_file):
            return []
        
        records = []
        with open(self.log_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    record = json.loads(line)
                    records.append({
                        "timestamp": record["timestamp"],
                        "episode_number": record["episode_number"],
                        "episode": self._deserialize_episode(record["data"])
                    })
        
        return records
    
    def query_episodes(
        self,
        filter_func: Optional[Callable[[Dict[str, Any]], bool]] = None,
        min_episode: Optional[int] = None,
        max_episode: Optional[int] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """按条件查询和过滤回合数据。
        
        Args:
            filter_func: 自定义过滤函数，接收记录字典，返回是否保留
            min_episode: 最小回合编号（包含）
            max_episode: 最大回合编号（包含）
            start_time: 开始时间（ISO格式字符串）
            end_time: 结束时间（ISO格式字符串）
            
        Returns:
            满足条件的记录列表
        """
        records = self.read_episodes()
        filtered = []
        
        for record in records:
            # 检查回合编号范围
            if min_episode is not None and record["episode_number"] < min_episode:
                continue
            if max_episode is not None and record["episode_number"] > max_episode:
                continue
            
            # 检查时间范围
            if start_time is not None and record["timestamp"] < start_time:
                continue
            if end_time is not None and record["timestamp"] > end_time:
                continue
            
            # 应用自定义过滤函数
            if filter_func is not None and not filter_func(record):
                continue
            
            filtered.append(record)
        
        return filtered

    def export_to_csv(self, output_path: str) -> None:
        """导出训练数据为CSV格式。
        
        CSV文件包含以下列：
        - timestamp: 记录时间戳
        - episode_number: 回合编号
        - player_id: 玩家ID
        - final_reward: 最终奖励
        - num_states: 状态数量
        - num_actions: 行动数量
        - total_rewards: 累计奖励
        
        Args:
            output_path: 输出CSV文件路径
            
        Raises:
            IOError: 当写入失败时
        """
        records = self.read_episodes()
        
        if not records:
            # 创建空的CSV文件，只有表头
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'episode_number', 'player_id', 
                    'final_reward', 'num_states', 'num_actions', 'total_rewards'
                ])
            return
        
        try:
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                # 写入表头
                writer.writerow([
                    'timestamp', 'episode_number', 'player_id', 
                    'final_reward', 'num_states', 'num_actions', 'total_rewards'
                ])
                
                # 写入数据行
                for record in records:
                    episode = record["episode"]
                    writer.writerow([
                        record["timestamp"],
                        record["episode_number"],
                        episode.player_id,
                        episode.final_reward,
                        len(episode.states),
                        len(episode.actions),
                        sum(episode.rewards)
                    ])
        except OSError as e:
            raise IOError(f"导出CSV文件失败: {e}")
    
    def export_to_json(self, output_path: str, pretty: bool = True) -> None:
        """导出训练数据为JSON格式。
        
        导出完整的训练数据，包含所有状态、行动和奖励信息。
        
        Args:
            output_path: 输出JSON文件路径
            pretty: 是否格式化输出（默认True）
            
        Raises:
            IOError: 当写入失败时
        """
        if not os.path.exists(self.log_file):
            # 创建空的JSON文件
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump([], f)
            return
        
        # 读取原始JSON Lines数据
        records = []
        with open(self.log_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                if pretty:
                    json.dump(records, f, ensure_ascii=False, indent=2)
                else:
                    json.dump(records, f, ensure_ascii=False)
        except OSError as e:
            raise IOError(f"导出JSON文件失败: {e}")
    
    def clear_log(self) -> None:
        """清空日志文件。
        
        用于测试或重置日志。
        """
        if os.path.exists(self.log_file):
            os.remove(self.log_file)
    
    def get_log_file_path(self) -> str:
        """获取当前日志文件路径。
        
        Returns:
            日志文件的完整路径
        """
        return self.log_file
    
    def get_episode_count(self) -> int:
        """获取已记录的回合数量。
        
        Returns:
            回合数量
        """
        if not os.path.exists(self.log_file):
            return 0
        
        count = 0
        with open(self.log_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    count += 1
        return count
