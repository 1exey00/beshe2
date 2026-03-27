"""Rule engine for Texas Hold'em poker game logic."""

from typing import List, Tuple
from copy import deepcopy
from models.core import GameState, Action, ActionType, GameStage, Card
from environment.hand_evaluator import compare_hands


class RuleEngine:
    """Validates and applies poker game rules."""
    
    @staticmethod
    def is_action_legal(state: GameState, action: Action) -> bool:
        """Verify if an action is legal in the current game state.
        
        Args:
            state: Current game state
            action: Action to validate
            
        Returns:
            True if action is legal, False otherwise
        """
        current_player = state.current_player
        player_stack = state.player_stacks[current_player]
        current_bet = state.current_bets[current_player]
        opponent_bet = state.current_bets[1 - current_player]
        
        # FOLD is always legal
        if action.action_type == ActionType.FOLD:
            return True
        
        # CHECK is only legal when bets are equal
        if action.action_type == ActionType.CHECK:
            return current_bet == opponent_bet
        
        # CALL is only legal when opponent has bet more and player has enough chips
        if action.action_type == ActionType.CALL:
            if current_bet >= opponent_bet:
                return False  # Nothing to call
            # Must have enough chips to call (otherwise should use ALL_IN)
            bet_to_call = opponent_bet - current_bet
            return player_stack >= bet_to_call
        
        # ALL_IN validation
        if action.action_type == ActionType.ALL_IN:
            # ALL_IN must use exactly all remaining chips
            if action.amount != player_stack:
                return False
            # Must have positive chips
            if action.amount <= 0:
                return False
            return True
        
        # RAISE validation（支持 RAISE, RAISE_SMALL, RAISE_BIG）
        if action.action_type in {ActionType.RAISE, ActionType.RAISE_SMALL, ActionType.RAISE_BIG}:
            # Must have positive raise amount
            if action.amount <= 0:
                return False
            
            # Total bet after raise
            total_bet = current_bet + action.amount
            
            # Must raise at least to match opponent's bet
            if total_bet <= opponent_bet:
                return False
            
            # Cannot raise more than stack
            if action.amount > player_stack:
                return False
            
            # 计算最小加注增量
            bet_to_call = max(0, opponent_bet - current_bet)
            
            # 如果是全下（all-in），允许任何金额
            # 只要加注后的总下注大于对手下注即可
            if action.amount == player_stack:
                return True
            
            # 非全下的加注需要满足最小加注要求
            # 最小加注 = 跟注金额 + 最小加注增量
            # 最小加注增量通常是上一次加注的大小或大盲注
            # 简化处理：最小加注增量 = max(bet_to_call, 1)
            # 使用1作为最小增量，允许更灵活的加注
            min_raise_increment = max(bet_to_call, 1)
            min_raise_amount = bet_to_call + min_raise_increment
            
            if action.amount < min_raise_amount:
                return False
            
            return True
        
        return False
    
    @staticmethod
    def apply_action(state: GameState, action: Action) -> GameState:
        """Apply an action and return the new game state.
        
        Args:
            state: Current game state
            action: Action to apply
            
        Returns:
            New game state after applying the action
            
        Raises:
            ValueError: If action is not legal
        """
        if not RuleEngine.is_action_legal(state, action):
            raise ValueError(f"Illegal action: {action}")
        
        # Create a deep copy of the state
        new_state = deepcopy(state)
        current_player = new_state.current_player
        
        # Add action to history
        new_state.action_history.append(action)
        
        # Handle FOLD
        if action.action_type == ActionType.FOLD:
            # Game ends, opponent wins
            # We'll mark this by setting a flag or just return the state
            # The caller will need to check if someone folded
            return new_state
        
        # Handle CHECK
        if action.action_type == ActionType.CHECK:
            # Check if bets are equal and both players have acted
            if new_state.current_bets[0] == new_state.current_bets[1]:
                # Bets are equal
                # Check if both players have acted in this betting round
                # This can be:
                # 1. Both players checked (last two actions are CHECK)
                # 2. One player called, other player checked (last action is CHECK, previous is CALL)
                
                if len(new_state.action_history) >= 2:
                    prev_action = new_state.action_history[-2]
                    # If previous action was CHECK or CALL, both players have acted
                    if prev_action.action_type in [ActionType.CHECK, ActionType.CALL]:
                        # Both players have acted with equal bets
                        # Don't advance stage if we're already at river (showdown)
                        if new_state.stage != GameStage.RIVER:
                            new_state = RuleEngine._advance_stage(new_state)
                        else:
                            # At river, just switch player (showdown will be handled by environment)
                            new_state.current_player = 1 - current_player
                        return new_state
                
                # If we get here, only one player has acted, switch to other player
                new_state.current_player = 1 - current_player
            else:
                # Bets are not equal, switch to other player
                new_state.current_player = 1 - current_player
            return new_state
        
        # Handle CALL
        if action.action_type == ActionType.CALL:
            opponent_bet = new_state.current_bets[1 - current_player]
            bet_to_call = opponent_bet - new_state.current_bets[current_player]
            
            # Handle all-in
            actual_call = min(bet_to_call, new_state.player_stacks[current_player])
            
            new_state.player_stacks[current_player] -= actual_call
            new_state.current_bets[current_player] += actual_call
            new_state.pot += actual_call
            
            # After call, switch to other player
            # They can check (advancing stage) or raise
            new_state.current_player = 1 - current_player
            
            return new_state
        
        # Handle RAISE（支持 RAISE, RAISE_SMALL, RAISE_BIG）
        if action.action_type in {ActionType.RAISE, ActionType.RAISE_SMALL, ActionType.RAISE_BIG}:
            # Deduct from stack
            new_state.player_stacks[current_player] -= action.amount
            new_state.current_bets[current_player] += action.amount
            new_state.pot += action.amount
            
            # Switch to other player (they need to respond to the raise)
            new_state.current_player = 1 - current_player
            
            return new_state
        
        # Handle ALL_IN
        if action.action_type == ActionType.ALL_IN:
            # Deduct all remaining chips from stack
            new_state.player_stacks[current_player] -= action.amount
            new_state.current_bets[current_player] += action.amount
            new_state.pot += action.amount
            
            # Switch to other player (they need to respond)
            new_state.current_player = 1 - current_player
            
            return new_state
        
        return new_state
    
    @staticmethod
    def _advance_stage(state: GameState) -> GameState:
        """Advance to the next game stage.
        
        Args:
            state: Current game state
            
        Returns:
            New game state in the next stage
        """
        new_state = deepcopy(state)
        
        # Reset current bets (they go into the pot)
        new_state.current_bets = [0, 0]
        
        # Clear action history for the new stage
        # This prevents checking actions from previous stages
        new_state.action_history = []
        
        # Advance stage
        if new_state.stage == GameStage.PREFLOP:
            new_state.stage = GameStage.FLOP
            # Flop cards should already be dealt by environment
        elif new_state.stage == GameStage.FLOP:
            new_state.stage = GameStage.TURN
            # Turn card should already be dealt by environment
        elif new_state.stage == GameStage.TURN:
            new_state.stage = GameStage.RIVER
            # River card should already be dealt by environment
        elif new_state.stage == GameStage.RIVER:
            # Game ends after river, no more stages
            pass
        
        # First to act after flop/turn/river is the player after button
        # In heads-up, button acts first preflop but last postflop
        if new_state.stage != GameStage.PREFLOP:
            new_state.current_player = 1 - new_state.button_position
        
        return new_state
    
    @staticmethod
    def determine_winner(state: GameState) -> int:
        """Determine the winner of the hand.
        
        Args:
            state: Final game state
            
        Returns:
            Index of winning player (0 or 1), or -1 for tie
        """
        # Check if someone folded
        if len(state.action_history) > 0:
            last_action = state.action_history[-1]
            if last_action.action_type == ActionType.FOLD:
                # The player who didn't fold wins
                # Find who folded by checking who acted last
                # In our apply_action, we don't switch player on fold
                # So current_player is the one who folded
                return 1 - state.current_player
        
        # Showdown - compare hands
        hand1 = list(state.player_hands[0])
        hand2 = list(state.player_hands[1])
        community = state.community_cards
        
        result = compare_hands(hand1, hand2, community)
        return result
    
    @staticmethod
    def distribute_pot(state: GameState, winner: int) -> GameState:
        """Distribute the pot to the winner(s).
        
        Args:
            state: Current game state
            winner: Index of winner (0, 1, or -1 for tie)
            
        Returns:
            New game state with pot distributed
        """
        new_state = deepcopy(state)
        
        if winner == -1:
            # Tie - split pot
            split_amount = new_state.pot // 2
            new_state.player_stacks[0] += split_amount
            new_state.player_stacks[1] += split_amount
            
            # Handle odd chip (goes to player after button)
            if new_state.pot % 2 == 1:
                new_state.player_stacks[1 - new_state.button_position] += 1
        else:
            # Winner takes all
            new_state.player_stacks[winner] += new_state.pot
        
        new_state.pot = 0
        return new_state
