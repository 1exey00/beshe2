"""Hand evaluation for Texas Hold'em poker."""

from typing import List, Tuple
from collections import Counter
from models.core import Card, HandRank


class HandEvaluator:
    """Evaluates poker hands and compares them."""
    
    @staticmethod
    def evaluate_hand(cards: List[Card]) -> Tuple[HandRank, List[int]]:
        """Evaluate a poker hand and return its rank and kickers.
        
        Args:
            cards: List of 5-7 cards to evaluate
            
        Returns:
            Tuple of (HandRank, kickers) where kickers is a list of ranks
            in descending order used for tie-breaking
        """
        if len(cards) < 5:
            raise ValueError(f"Need at least 5 cards to evaluate, got {len(cards)}")
        
        # If more than 5 cards, find the best 5-card combination
        if len(cards) > 5:
            return HandEvaluator._find_best_hand(cards)
        
        # Exactly 5 cards
        return HandEvaluator._evaluate_five_cards(cards)
    
    @staticmethod
    def _find_best_hand(cards: List[Card]) -> Tuple[HandRank, List[int]]:
        """Find the best 5-card hand from 6 or 7 cards."""
        from itertools import combinations
        
        best_rank = HandRank.HIGH_CARD
        best_kickers = []
        
        # Try all 5-card combinations
        for combo in combinations(cards, 5):
            rank, kickers = HandEvaluator._evaluate_five_cards(list(combo))
            if (rank.value > best_rank.value or 
                (rank.value == best_rank.value and kickers > best_kickers)):
                best_rank = rank
                best_kickers = kickers
        
        return best_rank, best_kickers
    
    @staticmethod
    def _evaluate_five_cards(cards: List[Card]) -> Tuple[HandRank, List[int]]:
        """Evaluate exactly 5 cards."""
        ranks = sorted([c.rank for c in cards], reverse=True)
        suits = [c.suit for c in cards]
        
        is_flush_hand = is_flush(suits)
        is_straight_hand, straight_high = is_straight(ranks)
        
        # Straight flush
        if is_flush_hand and is_straight_hand:
            return HandRank.STRAIGHT_FLUSH, [straight_high]
        
        # Four of a kind
        rank_counts = Counter(ranks)
        counts = sorted(rank_counts.values(), reverse=True)
        
        if counts[0] == 4:
            quad_rank = [r for r, c in rank_counts.items() if c == 4][0]
            kicker = [r for r, c in rank_counts.items() if c == 1][0]
            return HandRank.FOUR_OF_A_KIND, [quad_rank, kicker]
        
        # Full house
        if counts[0] == 3 and counts[1] == 2:
            trip_rank = [r for r, c in rank_counts.items() if c == 3][0]
            pair_rank = [r for r, c in rank_counts.items() if c == 2][0]
            return HandRank.FULL_HOUSE, [trip_rank, pair_rank]
        
        # Flush
        if is_flush_hand:
            return HandRank.FLUSH, ranks
        
        # Straight
        if is_straight_hand:
            return HandRank.STRAIGHT, [straight_high]
        
        # Three of a kind
        if counts[0] == 3:
            trip_rank = [r for r, c in rank_counts.items() if c == 3][0]
            kickers = sorted([r for r, c in rank_counts.items() if c == 1], reverse=True)
            return HandRank.THREE_OF_A_KIND, [trip_rank] + kickers
        
        # Two pair
        if counts[0] == 2 and counts[1] == 2:
            pairs = sorted([r for r, c in rank_counts.items() if c == 2], reverse=True)
            kicker = [r for r, c in rank_counts.items() if c == 1][0]
            return HandRank.TWO_PAIR, pairs + [kicker]
        
        # One pair
        if counts[0] == 2:
            pair_rank = [r for r, c in rank_counts.items() if c == 2][0]
            kickers = sorted([r for r, c in rank_counts.items() if c == 1], reverse=True)
            return HandRank.PAIR, [pair_rank] + kickers
        
        # High card
        return HandRank.HIGH_CARD, ranks


def is_flush(suits: List[str]) -> bool:
    """Check if all cards have the same suit."""
    return len(set(suits)) == 1


def is_straight(ranks: List[int]) -> Tuple[bool, int]:
    """Check if cards form a straight.
    
    Returns:
        Tuple of (is_straight, high_card_rank)
    """
    # Sort ranks in descending order
    sorted_ranks = sorted(set(ranks), reverse=True)
    
    # Check for regular straight
    if len(sorted_ranks) == 5:
        if sorted_ranks[0] - sorted_ranks[4] == 4:
            return True, sorted_ranks[0]
    
    # Check for A-2-3-4-5 (wheel) straight
    if sorted_ranks == [14, 5, 4, 3, 2]:
        return True, 5  # In a wheel, the high card is 5, not Ace
    
    return False, 0


def find_n_of_a_kind(ranks: List[int], n: int) -> List[int]:
    """Find all ranks that appear exactly n times.
    
    Args:
        ranks: List of card ranks
        n: Number of occurrences to find
        
    Returns:
        List of ranks that appear n times, sorted in descending order
    """
    rank_counts = Counter(ranks)
    return sorted([r for r, c in rank_counts.items() if c == n], reverse=True)


def evaluate_hand(cards: List[Card]) -> Tuple[HandRank, List[int]]:
    """Convenience function to evaluate a hand.
    
    Args:
        cards: List of 5-7 cards
        
    Returns:
        Tuple of (HandRank, kickers)
    """
    return HandEvaluator.evaluate_hand(cards)


def compare_hands(hand1: List[Card], hand2: List[Card], 
                  community: List[Card] = None) -> int:
    """Compare two poker hands.
    
    Args:
        hand1: First player's hole cards (2 cards)
        hand2: Second player's hole cards (2 cards)
        community: Community cards (0-5 cards), optional
        
    Returns:
        0 if hand1 wins, 1 if hand2 wins, -1 if tie
    """
    if community is None:
        community = []
    
    # Combine hole cards with community cards
    full_hand1 = hand1 + community
    full_hand2 = hand2 + community
    
    # Evaluate both hands
    rank1, kickers1 = evaluate_hand(full_hand1)
    rank2, kickers2 = evaluate_hand(full_hand2)
    
    # Compare hand ranks
    if rank1.value > rank2.value:
        return 0
    elif rank2.value > rank1.value:
        return 1
    
    # Same rank, compare kickers
    for k1, k2 in zip(kickers1, kickers2):
        if k1 > k2:
            return 0
        elif k2 > k1:
            return 1
    
    # Complete tie
    return -1
