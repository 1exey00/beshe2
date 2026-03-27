"""CFR 训练专用环境（无状态接口）。

该环境提供 CFR 树遍历常用能力：
- 纯函数式状态转移（不依赖内部 current_state）
- 机会节点发牌（可随机，也可外部指定）
- 终局判断与效用计算
"""

from __future__ import annotations

import random
from copy import deepcopy
from typing import List, Tuple

from models.core import Action, ActionType, Card, GameStage, GameState
from environment.rule_engine import RuleEngine
from environment.poker_environment import PokerEnvironment


class CFREnvironment:
    """面向 CFR/CFR+SL 的德扑环境包装器。"""

    def __init__(
        self,
        initial_stack: int = 1000,
        small_blind: int = 5,
        big_blind: int = 10,
        max_raises_per_street: int = 4,
    ):
        self.initial_stack = initial_stack
        self.small_blind = small_blind
        self.big_blind = big_blind
        self.max_raises_per_street = max_raises_per_street

        self.rule_engine = RuleEngine()
        # 复用已有合法动作逻辑
        self._poker_env = PokerEnvironment(
            initial_stack=initial_stack,
            small_blind=small_blind,
            big_blind=big_blind,
            max_raises_per_street=max_raises_per_street,
        )

    def reset(self) -> GameState:
        """重置并返回初始状态。"""
        return self._poker_env.reset()

    def get_legal_actions(self, state: GameState) -> List[Action]:
        """获取合法动作。"""
        return self._poker_env.get_legal_actions(state)

    def apply_action(self, state: GameState, action: Action) -> GameState:
        """纯函数式应用动作。"""
        return self.rule_engine.apply_action(state, action)

    def apply_action_and_advance_chance(
        self,
        state: GameState,
        action: Action,
    ) -> GameState:
        """应用动作并在需要时推进机会节点（发公共牌）。"""
        next_state = self.apply_action(state, action)

        # 仅在非终局时处理公共牌发放
        if not self.is_terminal(next_state):
            next_state = self._advance_chance_if_needed(next_state)

        return next_state

    def is_terminal(self, state: GameState) -> bool:
        """判断是否终局（弃牌或河牌摊牌完成）。"""
        if state.action_history and state.action_history[-1].action_type == ActionType.FOLD:
            return True

        if state.stage != GameStage.RIVER:
            return False

        if len(state.community_cards) != 5:
            return False

        if state.current_bets[0] != state.current_bets[1]:
            return False

        # 河牌回合结束：CALL 或 连续 CHECK
        if state.action_history:
            last = state.action_history[-1]
            if last.action_type == ActionType.CALL:
                return True
            if len(state.action_history) >= 2 and last.action_type == ActionType.CHECK:
                prev = state.action_history[-2]
                if prev.action_type == ActionType.CHECK:
                    return True

        return False

    def get_terminal_utility(self, state: GameState, player_id: int) -> float:
        """返回终局收益（相对初始筹码变化）。"""
        if not self.is_terminal(state):
            return 0.0

        winner = self.rule_engine.determine_winner(state)
        final_state = self.rule_engine.distribute_pot(state, winner)
        return float(final_state.player_stacks[player_id] - self.initial_stack)

    def enumerate_chance_cards(self, state: GameState, count: int) -> List[Tuple[Card, ...]]:
        """枚举在当前状态下可发放的机会牌组合。"""
        if count != 1:
            raise ValueError(f"当前仅支持 count=1 的机会牌枚举，收到 count={count}")

        deck = self._remaining_deck(state)
        return [(c,) for c in deck]

    def _advance_chance_if_needed(
        self,
        state: GameState,
    ) -> GameState:
        """在阶段推进后补发公共牌。"""
        next_state = deepcopy(state)

        expected = {
            GameStage.FLOP: 3,
            GameStage.TURN: 4,
            GameStage.RIVER: 5,
        }

        if next_state.stage not in expected:
            return next_state

        target_count = expected[next_state.stage]
        need = target_count - len(next_state.community_cards)
        if need <= 0:
            return next_state

        remaining = self._remaining_deck(next_state)
        cards_to_deal = random.sample(remaining, need)

        next_state.community_cards.extend(cards_to_deal)
        return next_state

    def _remaining_deck(self, state: GameState) -> List[Card]:
        used = {(c.rank, c.suit) for hand in state.player_hands for c in hand}
        used.update((c.rank, c.suit) for c in state.community_cards)

        deck: List[Card] = []
        for suit in ["h", "d", "c", "s"]:
            for rank in range(2, 15):
                if (rank, suit) not in used:
                    deck.append(Card(rank, suit))
        return deck
