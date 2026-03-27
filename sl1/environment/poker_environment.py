"""Poker game environment for Texas Hold'em."""

import random
from typing import List, Tuple
from copy import deepcopy
from models.core import GameState, Action, ActionType, GameStage, Card
from environment.rule_engine import RuleEngine
from environment.hand_evaluator import compare_hands


class PokerEnvironment:
    """德州扑克游戏环境，用于AI训练。
    
    支持配置每条街的最大加注次数，以减少游戏树深度并加速收敛。
    """
    
    def __init__(
        self, 
        initial_stack: int = 1000, 
        small_blind: int = 5, 
        big_blind: int = 10,
        max_raises_per_street: int = 4
    ):
        """初始化扑克环境。
        
        Args:
            initial_stack: 每个玩家的初始筹码
            small_blind: 小盲注金额
            big_blind: 大盲注金额
            max_raises_per_street: 每条街最大加注次数（默认4次，设为0表示无限制）
        """
        self.initial_stack = initial_stack
        self.small_blind = small_blind
        self.big_blind = big_blind
        self.max_raises_per_street = max_raises_per_street
        self.rule_engine = RuleEngine()
        self.current_state = None
        self.deck = []
        
    def reset(self) -> GameState:
        """Start a new hand and return the initial game state.
        
        Returns:
            Initial game state with cards dealt and blinds posted
        """
        # Create and shuffle deck
        self.deck = self._create_deck()
        random.shuffle(self.deck)
        
        # Deal hole cards (2 cards per player)
        player_hands = [
            (self.deck.pop(), self.deck.pop()),
            (self.deck.pop(), self.deck.pop())
        ]
        
        # Initialize game state
        # Button position alternates (we'll start with player 0 as button)
        button_position = 0
        
        # In heads-up, button posts small blind and acts first preflop
        # Player after button (1) posts big blind
        player_stacks = [self.initial_stack, self.initial_stack]
        
        # Post blinds
        player_stacks[button_position] -= self.small_blind
        player_stacks[1 - button_position] -= self.big_blind
        
        current_bets = [0, 0]
        current_bets[button_position] = self.small_blind
        current_bets[1 - button_position] = self.big_blind
        
        pot = self.small_blind + self.big_blind
        
        # In heads-up preflop, button (small blind) acts first
        current_player = button_position
        
        self.current_state = GameState(
            player_hands=player_hands,
            community_cards=[],
            pot=pot,
            player_stacks=player_stacks,
            current_bets=current_bets,
            button_position=button_position,
            stage=GameStage.PREFLOP,
            action_history=[],
            current_player=current_player
        )
        
        return deepcopy(self.current_state)
    
    def step(self, action: Action) -> Tuple[GameState, float, bool]:
        """Execute an action and return the new state, reward, and done flag.
        
        Args:
            action: Action to execute
            
        Returns:
            Tuple of (new_state, reward, done) where:
                - new_state: Updated game state
                - reward: Reward for the current player (0 during hand, final stack change at end)
                - done: True if the hand is complete
        """
        if self.current_state is None:
            raise ValueError("Must call reset() before step()")
        
        # Store initial stack for reward calculation
        current_player = self.current_state.current_player
        initial_stack = self.current_state.player_stacks[current_player]
        
        # Apply the action
        self.current_state = self.rule_engine.apply_action(self.current_state, action)
        
        # Check if hand is over
        done = False
        reward = 0.0
        
        # Check for fold
        if action.action_type == ActionType.FOLD:
            # Hand is over, opponent wins
            winner = 1 - current_player
            self.current_state = self.rule_engine.distribute_pot(self.current_state, winner)
            done = True
            # Reward is the change in stack
            final_stack = self.current_state.player_stacks[current_player]
            reward = float(final_stack - initial_stack)
        else:
            # Check if we need to deal community cards
            if self._should_deal_community_cards():
                self._deal_community_cards()
            
            # Check if we've reached showdown
            if self._is_showdown():
                winner = self.rule_engine.determine_winner(self.current_state)
                self.current_state = self.rule_engine.distribute_pot(self.current_state, winner)
                done = True
                # Reward is the change in stack
                final_stack = self.current_state.player_stacks[current_player]
                reward = float(final_stack - initial_stack)
        
        return deepcopy(self.current_state), reward, done
    
    def get_legal_actions(self, state: GameState = None) -> List[Action]:
        """获取当前状态的合法行动列表。
        
        德州扑克规则：
        - 当可以过牌（CHECK）时，不应该有弃牌（FOLD）选项
        - 当需要跟注时，有FOLD和CALL选项
        - 每条街最多加注 max_raises_per_street 次
        - 加注分为两种：RAISE_SMALL（半底池）和 RAISE_BIG（全底池）
        - 当筹码不足以进行标准加注时，只能选择 ALL_IN
        
        底池计算规则：
        - 下注相等时：底池 = 当前底池
        - 下注不等时：先加上跟注金额，然后计算底池大小
        
        Args:
            state: 要检查的游戏状态（如果为None则使用当前状态）
            
        Returns:
            合法行动列表
        """
        if state is None:
            state = self.current_state
        
        if state is None:
            raise ValueError("没有可用的游戏状态")
        
        legal_actions = []
        current_player = state.current_player
        
        # 检查下注是否相等
        bets_equal = state.current_bets[0] == state.current_bets[1]
        
        # 计算基本信息
        player_stack = state.player_stacks[current_player]
        current_bet = state.current_bets[current_player]
        opponent_bet = state.current_bets[1 - current_player]
        bet_to_call = opponent_bet - current_bet
        
        # 如果玩家已经全下（筹码为0），没有合法行动
        if player_stack <= 0:
            return legal_actions
        
        if bets_equal:
            # 下注相等时可以过牌，不需要弃牌选项
            legal_actions.append(Action(ActionType.CHECK))
        else:
            # 下注不等时需要跟注或弃牌
            legal_actions.append(Action(ActionType.FOLD))
            # 只有当玩家有足够筹码跟注时才添加 CALL
            # 如果筹码不足以跟注，则只能 ALL_IN 或 FOLD
            if player_stack >= bet_to_call:
                legal_actions.append(Action(ActionType.CALL))
            # 如果筹码不足以跟注，ALL_IN 会在下面处理
        
        # 检查当前街的加注次数是否已达上限
        if self.max_raises_per_street > 0:
            raise_count = self._count_raises_in_current_street(state)
            if raise_count >= self.max_raises_per_street:
                # 已达加注上限，但如果筹码不足以跟注，仍可以 ALL_IN
                if not bets_equal and player_stack < bet_to_call:
                    legal_actions.append(Action(ActionType.ALL_IN, player_stack))
                return legal_actions
        
        # 计算有效底池（用于计算加注大小）
        # 规则：下注相等时为当前底池；下注不等时先加上跟注金额
        if bets_equal:
            effective_pot = state.pot
        else:
            effective_pot = state.pot + bet_to_call
        
        # 计算最小加注金额
        min_raise_increment = max(bet_to_call, self.big_blind) if bet_to_call > 0 else self.big_blind
        min_raise_amount = bet_to_call + min_raise_increment
        
        # 计算半底池和全底池加注金额
        half_pot_raise = bet_to_call + effective_pot // 2
        full_pot_raise = bet_to_call + effective_pot
        
        # 确保半底池加注至少是最小加注
        half_pot_raise = max(half_pot_raise, min_raise_amount)
        # 确保全底池加注大于半底池加注
        full_pot_raise = max(full_pot_raise, half_pot_raise + 1)
        
        # 判断筹码是否足够进行各种加注
        can_raise_small = player_stack >= half_pot_raise
        can_raise_big = player_stack >= full_pot_raise
        can_min_raise = player_stack >= min_raise_amount
        
        if can_raise_small:
            # 筹码足够进行半底池加注
            legal_actions.append(Action(ActionType.RAISE_SMALL, half_pot_raise))
        
        if can_raise_big:
            # 筹码足够进行全底池加注
            legal_actions.append(Action(ActionType.RAISE_BIG, full_pot_raise))
        elif can_raise_small and player_stack > half_pot_raise:
            # 筹码不足以全底池加注，但超过半底池加注，可以 ALL_IN
            legal_actions.append(Action(ActionType.ALL_IN, player_stack))
        elif not can_raise_small and can_min_raise:
            # 筹码不足以半底池加注，但足够最小加注，可以 ALL_IN
            legal_actions.append(Action(ActionType.ALL_IN, player_stack))
        elif not can_min_raise and player_stack > 0:
            # 筹码不足以最小加注，只能 ALL_IN（作为不足额加注）
            # 或者筹码不足以跟注时的 ALL_IN
            if not bets_equal and player_stack < bet_to_call:
                # 筹码不足以跟注，ALL_IN
                legal_actions.append(Action(ActionType.ALL_IN, player_stack))
            elif player_stack > bet_to_call:
                # 筹码超过跟注金额但不足以最小加注，可以 ALL_IN
                legal_actions.append(Action(ActionType.ALL_IN, player_stack))
        
        return legal_actions
    
    def _count_raises_in_current_street(self, state: GameState) -> int:
        """统计当前街的加注次数。
        
        使用简化的方法：从行动历史末尾向前查找，
        直到遇到阶段结束标志（CALL 或连续两个 CHECK）。
        
        注意：ALL_IN 也算作加注（如果金额超过跟注金额）。
        
        Args:
            state: 游戏状态
            
        Returns:
            当前街的加注次数
        """
        if not state.action_history:
            return 0
        
        raise_count = 0
        
        # 所有加注类型（包括 ALL_IN）
        raise_types = {ActionType.RAISE, ActionType.RAISE_SMALL, ActionType.RAISE_BIG, ActionType.ALL_IN}
        
        # 从后向前遍历，统计加注次数，直到遇到阶段结束标志
        for i in range(len(state.action_history) - 1, -1, -1):
            action = state.action_history[i]
            
            if action.action_type in raise_types:
                raise_count += 1
            elif action.action_type == ActionType.CALL:
                # CALL 是阶段结束标志，停止计数
                break
            elif action.action_type == ActionType.CHECK:
                # 检查是否是连续两个 CHECK（阶段结束标志）
                if i > 0 and state.action_history[i - 1].action_type == ActionType.CHECK:
                    break
        
        return raise_count
    
    def _create_deck(self) -> List[Card]:
        """Create a standard 52-card deck.
        
        Returns:
            List of all 52 cards
        """
        deck = []
        for suit in ['h', 'd', 'c', 's']:
            for rank in range(2, 15):  # 2-14 (A)
                deck.append(Card(rank, suit))
        return deck
    
    def _deal_community_cards(self) -> None:
        """Deal community cards based on current stage."""
        stage = self.current_state.stage
        num_community = len(self.current_state.community_cards)
        
        if stage == GameStage.FLOP and num_community == 0:
            # Deal 3 cards for flop
            for _ in range(3):
                self.current_state.community_cards.append(self.deck.pop())
        elif stage == GameStage.TURN and num_community == 3:
            # Deal 1 card for turn
            self.current_state.community_cards.append(self.deck.pop())
        elif stage == GameStage.RIVER and num_community == 4:
            # Deal 1 card for river
            self.current_state.community_cards.append(self.deck.pop())
    
    def _should_deal_community_cards(self) -> bool:
        """Check if we should deal community cards.
        
        Returns:
            True if community cards should be dealt
        """
        stage = self.current_state.stage
        num_community = len(self.current_state.community_cards)
        
        # Deal flop (3 cards) when entering flop stage
        if stage == GameStage.FLOP and num_community == 0:
            return True
        
        # Deal turn (1 card) when entering turn stage
        if stage == GameStage.TURN and num_community == 3:
            return True
        
        # Deal river (1 card) when entering river stage
        if stage == GameStage.RIVER and num_community == 4:
            return True
        
        return False
    
    def _is_showdown(self) -> bool:
        """Check if we've reached showdown (all betting complete).
        
        Returns:
            True if hand should go to showdown
        """
        # Showdown happens when:
        # 1. We're at river stage
        # 2. Both players have acted
        # 3. Bets are equal (both checked or one called)
        # 4. All 5 community cards are dealt
        
        if self.current_state.stage != GameStage.RIVER:
            return False
        
        # Check if all community cards are dealt
        if len(self.current_state.community_cards) != 5:
            return False
        
        # Check if both players have equal bets
        if self.current_state.current_bets[0] != self.current_state.current_bets[1]:
            return False
        
        # Check if last action was CALL (someone called a bet on river)
        if len(self.current_state.action_history) > 0:
            last_action = self.current_state.action_history[-1]
            if last_action.action_type == ActionType.CALL:
                return True
        
        # Check if both players have checked on the river
        # Simply check if the last two actions were both CHECK
        if len(self.current_state.action_history) >= 2:
            last_action = self.current_state.action_history[-1]
            prev_action = self.current_state.action_history[-2]
            
            if last_action.action_type == ActionType.CHECK and prev_action.action_type == ActionType.CHECK:
                return True
        
        return False
