"""游戏树导航器模块。

本模块实现了游戏树的导航和管理功能：
- 游戏树的构建和遍历（懒加载方式）
- 节点导航和路径追踪
- 行动历史管理
- 使用PokerEnvironment动态获取合法行动
"""

from typing import List, Optional, Dict
from copy import deepcopy
from models.core import GameStage, Action, ActionType, GameState, Card
from viewer.models import GameTreeNode, NodeType
from environment.poker_environment import PokerEnvironment


class GameTreeNavigator:
    """游戏树导航器 - 管理游戏树的导航和状态。
    
    使用懒加载方式构建游戏树：只在用户点击节点时才展开子节点。
    这样可以避免一开始就构建整棵树导致的性能问题。
    
    Attributes:
        _root: 游戏树根节点
        _current_node: 当前选中的节点
        _nodes: 所有节点的字典（node_id -> node）
        _env: 扑克环境，用于获取合法行动
    """
    
    def __init__(
        self, 
        root: Optional[GameTreeNode] = None,
        initial_stack: int = 1000,
        small_blind: int = 5,
        big_blind: int = 10,
        max_depth: int = 6,
        max_raises_per_street: int = 2
    ):
        """初始化游戏树导航器。
        
        Args:
            root: 根节点，如果为None则创建默认根节点
            initial_stack: 初始筹码
            small_blind: 小盲注
            big_blind: 大盲注
            max_depth: 最大树深度（限制树的大小）
            max_raises_per_street: 每条街最大加注次数（默认2次）
        """
        self._initial_stack = initial_stack
        self._small_blind = small_blind
        self._big_blind = big_blind
        self._max_depth = max_depth
        self._max_raises_per_street = max_raises_per_street
        
        # 创建扑克环境用于获取合法行动
        self._env = PokerEnvironment(
            initial_stack=initial_stack,
            small_blind=small_blind,
            big_blind=big_blind,
            max_raises_per_street=max_raises_per_street
        )
        
        self._nodes: Dict[str, GameTreeNode] = {}
        
        if root is None:
            self._root = self._create_default_root()
        else:
            self._root = root
        
        self._current_node = self._root
        self._index_node(self._root)
        
        # 展开根节点的第一层子节点
        self._expand_node(self._root)
        # 展开玩家0的决策节点
        if self._root.children:
            self._expand_node(self._root.children[0])
    
    def _create_default_root(self) -> GameTreeNode:
        """创建默认的游戏树根节点。
        
        Returns:
            根节点（Game Begin）
        """
        root = GameTreeNode(
            node_id="root",
            stage=GameStage.PREFLOP,
            player=-1,
            action=None,
            parent=None,
            children=[],
            pot=self._small_blind + self._big_blind,
            stacks=[
                self._initial_stack - self._small_blind,
                self._initial_stack - self._big_blind
            ],
            board_cards=[],
            action_history=[],
            node_type=NodeType.ROOT
        )
        
        # 创建玩家0（小盲位）的决策节点
        player0_node = GameTreeNode(
            node_id="root_p0",
            stage=GameStage.PREFLOP,
            player=0,
            action=None,
            parent=root,
            children=[],
            pot=root.pot,
            stacks=root.stacks.copy(),
            board_cards=[],
            action_history=[],
            node_type=NodeType.PLAYER
        )
        root.children.append(player0_node)
        self._index_node(player0_node)
        
        return root
    
    def _index_node(self, node: GameTreeNode) -> None:
        """索引单个节点。"""
        self._nodes[node.node_id] = node
    
    def _index_nodes(self, node: GameTreeNode) -> None:
        """递归索引所有节点。"""
        self._nodes[node.node_id] = node
        for child in node.children:
            self._index_nodes(child)

    def _expand_node(self, node: GameTreeNode) -> None:
        """展开节点的子节点（懒加载）。
        
        只有当节点没有子节点时才会展开。
        
        Args:
            node: 要展开的节点
        """
        # 如果已经有子节点，不需要再展开
        if node.children:
            return
        
        # 终端节点不需要展开
        if node.node_type == NodeType.TERMINAL:
            return
        
        # 发牌节点：添加决策节点
        if node.node_type == NodeType.CHANCE:
            self._add_post_chance_node(node)
            return
        
        # 玩家节点：获取合法行动并创建子节点
        game_state = self._create_game_state_from_node(node)
        legal_actions = self._env.get_legal_actions(game_state)
        
        # 过滤行动，只保留关键选项
        filtered_actions = self._filter_actions(legal_actions, game_state)
        
        for action in filtered_actions:
            child_node = self._create_child_node(node, action, game_state)
            node.children.append(child_node)
            self._index_node(child_node)
    
    def _filter_actions(self, legal_actions: List[Action], game_state: GameState) -> List[Action]:
        """过滤行动列表，只保留关键的行动选项。
        
        德州扑克规则：
        - 如果可以过牌（check），则不应该有弃牌（fold）选项
        - 如果需要跟注，则有fold和call选项
        - 支持 RAISE_SMALL、RAISE_BIG 和 ALL_IN 三种加注类型
        """
        filtered = []
        raise_actions = []
        has_check = False
        
        # 所有加注类型（包括 ALL_IN）
        raise_types = {ActionType.RAISE, ActionType.RAISE_SMALL, ActionType.RAISE_BIG, ActionType.ALL_IN}
        
        for action in legal_actions:
            if action.action_type == ActionType.CHECK:
                has_check = True
                filtered.append(action)
            elif action.action_type == ActionType.CALL:
                filtered.append(action)
            elif action.action_type == ActionType.FOLD:
                # 暂时保存，稍后决定是否添加
                pass
            elif action.action_type in raise_types:
                raise_actions.append(action)
        
        # 如果不能过牌，才添加弃牌选项
        if not has_check:
            for action in legal_actions:
                if action.action_type == ActionType.FOLD:
                    filtered.insert(0, action)  # 弃牌放在最前面
                    break
        
        # 添加所有加注选项（RAISE_SMALL、RAISE_BIG 和 ALL_IN）
        # 按金额排序后添加
        if raise_actions:
            raise_actions.sort(key=lambda a: a.amount if a.amount else 0)
            filtered.extend(raise_actions)
        
        return filtered
    
    def _create_game_state_from_node(self, node: GameTreeNode) -> GameState:
        """从游戏树节点创建GameState对象。
        
        正确计算current_bets，这对于确定合法行动至关重要。
        """
        # 计算当前阶段的下注
        current_bets = self._calculate_current_bets(node)
        
        dummy_hands = [
            (Card(14, 'h'), Card(13, 'h')),
            (Card(14, 's'), Card(13, 's'))
        ]
        
        return GameState(
            player_hands=dummy_hands,
            community_cards=node.board_cards.copy(),
            pot=node.pot,
            player_stacks=node.stacks.copy(),
            current_bets=current_bets,
            button_position=0,
            stage=node.stage,
            action_history=node.action_history.copy(),
            current_player=node.player if node.player >= 0 else 0
        )
    
    def _calculate_current_bets(self, node: GameTreeNode) -> List[int]:
        """计算当前阶段每个玩家的下注金额。
        
        这是确定合法行动的关键：
        - 如果两个玩家下注相等，可以过牌
        - 如果下注不等，需要跟注或加注
        
        注意：只计算当前阶段的下注，不包括之前阶段的下注。
        """
        # 找到当前阶段开始的位置
        # 通过回溯父节点找到阶段变化点
        current_stage = node.stage
        
        # 找到当前阶段的行动
        # 从节点向上回溯，找到阶段开始的位置
        stage_start_index = 0
        
        # 遍历行动历史，找到当前阶段开始的位置
        # 阶段变化发生在：翻牌前结束后进入翻牌，翻牌结束后进入转牌，等等
        # 我们需要找到最后一次阶段变化的位置
        
        # 简化方法：从节点向上回溯，找到阶段不同的祖先节点
        ancestor = node.parent
        while ancestor is not None:
            if ancestor.stage != current_stage:
                # 找到了阶段变化点，当前阶段的行动从这里开始
                stage_start_index = len(ancestor.action_history)
                break
            ancestor = ancestor.parent
        
        # 获取当前阶段的行动
        current_stage_actions = node.action_history[stage_start_index:]
        
        # 翻牌前初始状态：小盲5，大盲10
        if node.stage == GameStage.PREFLOP:
            bets = [self._small_blind, self._big_blind]
        else:
            # 翻牌后每个阶段开始时下注为0
            bets = [0, 0]
        
        # 根据当前阶段的行动历史更新下注
        # 翻牌后，玩家1（大盲位）先行动
        if node.stage != GameStage.PREFLOP:
            current_actor = 1  # 翻牌后大盲位先行动
        else:
            current_actor = 0  # 翻牌前小盲位先行动
        
        # 所有加注类型（包括 ALL_IN）
        raise_types = {ActionType.RAISE, ActionType.RAISE_SMALL, ActionType.RAISE_BIG, ActionType.ALL_IN}
        
        for action in current_stage_actions:
            if action.action_type == ActionType.CALL:
                # 跟注：补齐到对手的下注
                bets[current_actor] = max(bets)
            elif action.action_type in raise_types:
                # 加注：当前玩家的下注 = 对手下注 + 加注金额
                # action.amount 是加注的总金额（包括跟注部分）
                bets[current_actor] = max(bets) + action.amount
            elif action.action_type == ActionType.CHECK:
                # 过牌不改变下注
                pass
            elif action.action_type == ActionType.FOLD:
                # 弃牌不改变下注
                pass
            
            # 切换到下一个玩家
            current_actor = 1 - current_actor
        
        return bets
    
    def _create_child_node(
        self, 
        parent_node: GameTreeNode, 
        action: Action,
        game_state: GameState
    ) -> GameTreeNode:
        """根据行动创建子节点。"""
        action_str = self._action_to_string(action)
        node_id = f"{parent_node.node_id}_{action_str}"
        
        new_stacks = parent_node.stacks.copy()
        new_pot = parent_node.pot
        current_player = parent_node.player
        
        # 所有加注类型（包括 ALL_IN）
        raise_types = {ActionType.RAISE, ActionType.RAISE_SMALL, ActionType.RAISE_BIG, ActionType.ALL_IN}
        
        if action.action_type == ActionType.FOLD:
            node_type = NodeType.TERMINAL
            next_player = -1
        elif action.action_type == ActionType.CHECK:
            node_type = NodeType.PLAYER
            next_player = 1 - current_player
        elif action.action_type == ActionType.CALL:
            call_amount = abs(game_state.current_bets[1 - current_player] - game_state.current_bets[current_player])
            new_stacks[current_player] -= call_amount
            new_pot += call_amount
            node_type = NodeType.PLAYER
            next_player = 1 - current_player
        elif action.action_type in raise_types:
            new_stacks[current_player] -= action.amount
            new_pot += action.amount
            node_type = NodeType.PLAYER
            next_player = 1 - current_player
        else:
            node_type = NodeType.PLAYER
            next_player = 1 - current_player
        
        # 检查是否进入下一阶段
        enters_next_stage = self._check_stage_transition(parent_node, action, game_state)
        new_stage = parent_node.stage
        
        if enters_next_stage and node_type != NodeType.TERMINAL:
            next_stage = self._get_next_stage(parent_node.stage)
            if next_stage:
                # 创建过渡节点，子节点将是CHANCE节点
                node_type = NodeType.PLAYER
                new_stage = parent_node.stage
            elif parent_node.stage == GameStage.RIVER:
                node_type = NodeType.TERMINAL
        
        child_node = GameTreeNode(
            node_id=node_id,
            stage=new_stage,
            player=next_player,
            action=action,
            parent=parent_node,
            children=[],
            pot=new_pot,
            stacks=new_stacks,
            board_cards=parent_node.board_cards.copy(),
            action_history=parent_node.action_history + [action],
            node_type=node_type
        )
        
        # 如果进入下一阶段，立即添加CHANCE节点
        if enters_next_stage and child_node.node_type != NodeType.TERMINAL:
            next_stage = self._get_next_stage(parent_node.stage)
            if next_stage:
                chance_node = GameTreeNode(
                    node_id=f"{child_node.node_id}_{next_stage.value}",
                    stage=next_stage,
                    player=-1,
                    action=None,
                    parent=child_node,
                    children=[],
                    pot=child_node.pot,
                    stacks=child_node.stacks.copy(),
                    board_cards=child_node.board_cards.copy(),
                    action_history=child_node.action_history.copy(),
                    node_type=NodeType.CHANCE
                )
                child_node.children.append(chance_node)
                self._index_node(chance_node)
        
        return child_node

    def _check_stage_transition(
        self, 
        parent_node: GameTreeNode, 
        action: Action,
        game_state: GameState
    ) -> bool:
        """检查是否需要进入下一阶段。
        
        阶段转换条件：
        - 翻牌前：大盲位过牌（无人加注时），或者有人加注后被跟注
        - 翻牌后：双方都过牌，或者有人加注后被跟注
        """
        if action.action_type == ActionType.FOLD:
            return False
        
        # 所有加注类型（包括 ALL_IN）
        raise_types = {ActionType.RAISE, ActionType.RAISE_SMALL, ActionType.RAISE_BIG, ActionType.ALL_IN}
        
        current_player = parent_node.player
        
        if parent_node.stage == GameStage.PREFLOP:
            # 翻牌前的阶段转换
            action_history = parent_node.action_history + [action]
            
            if action.action_type == ActionType.CHECK and current_player == 1:
                # 大盲位过牌，进入翻牌
                return True
            if action.action_type == ActionType.CALL:
                # 检查是否有加注（包括 RAISE_SMALL、RAISE_BIG 和 ALL_IN）
                has_raise = any(a.action_type in raise_types for a in action_history[:-1])
                if has_raise:
                    return True
        else:
            # 翻牌后的阶段转换
            # 需要只计算当前阶段的行动，不包括之前阶段的行动
            stage_actions = self._get_current_stage_actions(parent_node, action)
            
            if action.action_type == ActionType.CHECK:
                # 检查当前阶段是否有两个过牌
                check_count = sum(1 for a in stage_actions if a.action_type == ActionType.CHECK)
                if check_count >= 2:
                    return True
            elif action.action_type == ActionType.CALL:
                # 检查当前阶段是否有加注（包括 RAISE_SMALL、RAISE_BIG 和 ALL_IN）
                has_raise = any(a.action_type in raise_types for a in stage_actions[:-1])
                if has_raise:
                    return True
        
        return False
    
    def _get_current_stage_actions(
        self, 
        parent_node: GameTreeNode, 
        action: Action
    ) -> List[Action]:
        """获取当前阶段的行动历史（不包括之前阶段的行动）。
        
        通过回溯父节点找到阶段变化点，然后返回从那个点开始的行动。
        """
        current_stage = parent_node.stage
        all_actions = parent_node.action_history + [action]
        
        # 从节点向上回溯，找到阶段开始的位置
        stage_start_index = 0
        ancestor = parent_node.parent
        while ancestor is not None:
            if ancestor.stage != current_stage:
                # 找到了阶段变化点
                stage_start_index = len(ancestor.action_history)
                break
            ancestor = ancestor.parent
        
        # 返回当前阶段的行动
        return all_actions[stage_start_index:]
    
    def _get_next_stage(self, current_stage: GameStage) -> Optional[GameStage]:
        """获取下一个游戏阶段。"""
        stage_order = [GameStage.PREFLOP, GameStage.FLOP, GameStage.TURN, GameStage.RIVER]
        try:
            idx = stage_order.index(current_stage)
            if idx < len(stage_order) - 1:
                return stage_order[idx + 1]
        except ValueError:
            pass
        return None
    
    def _add_post_chance_node(self, chance_node: GameTreeNode) -> None:
        """在发牌节点后添加决策节点。"""
        decision_node = GameTreeNode(
            node_id=f"{chance_node.node_id}_decision",
            stage=chance_node.stage,
            player=1,  # 翻牌后大盲位先行动
            action=None,
            parent=chance_node,
            children=[],
            pot=chance_node.pot,
            stacks=chance_node.stacks.copy(),
            board_cards=chance_node.board_cards.copy(),
            action_history=chance_node.action_history.copy(),
            node_type=NodeType.PLAYER
        )
        chance_node.children.append(decision_node)
        self._index_node(decision_node)
    
    def _action_to_string(self, action: Action) -> str:
        """将行动转换为字符串ID。"""
        # 所有加注类型都使用 raise_金额 格式（ALL_IN 使用 all_in_金额 格式）
        raise_types = {ActionType.RAISE, ActionType.RAISE_SMALL, ActionType.RAISE_BIG}
        if action.action_type in raise_types:
            return f"raise_{action.amount}"
        if action.action_type == ActionType.ALL_IN:
            return f"all_in_{action.amount}"
        return action.action_type.value
    
    # ========== 公共API ==========
    
    def get_root(self) -> GameTreeNode:
        """获取游戏树根节点。"""
        return self._root
    
    def get_current_node(self) -> GameTreeNode:
        """获取当前选中的节点。"""
        return self._current_node
    
    def get_children(self, node: Optional[GameTreeNode] = None) -> List[GameTreeNode]:
        """获取指定节点的子节点列表（会触发懒加载）。"""
        if node is None:
            node = self._current_node
        # 懒加载：展开节点
        self._expand_node(node)
        return node.children.copy()
    
    def navigate_to(self, node: GameTreeNode) -> bool:
        """导航到指定节点（会触发懒加载）。"""
        if node.node_id not in self._nodes:
            return False
        self._current_node = node
        # 懒加载：展开当前节点
        self._expand_node(node)
        return True
    
    def navigate_to_by_id(self, node_id: str) -> bool:
        """通过节点ID导航到指定节点。"""
        if node_id not in self._nodes:
            return False
        self._current_node = self._nodes[node_id]
        self._expand_node(self._current_node)
        return True
    
    def get_path_to_root(self, node: Optional[GameTreeNode] = None) -> List[GameTreeNode]:
        """获取从根节点到指定节点的路径。"""
        if node is None:
            node = self._current_node
        return node.get_path_to_root()
    
    def get_available_actions(self, node: Optional[GameTreeNode] = None) -> List[str]:
        """获取指定节点的可用行动。"""
        if node is None:
            node = self._current_node
        
        if node.is_terminal():
            return []
        
        # 懒加载
        self._expand_node(node)
        
        raise_types = (ActionType.RAISE, ActionType.RAISE_SMALL, ActionType.RAISE_BIG)
        actions = []
        for child in node.children:
            if child.action is not None:
                action_name = child.action.action_type.value
                if child.action.action_type in raise_types:
                    action_name = f"raise_{child.action.amount}"
                elif child.action.action_type == ActionType.ALL_IN:
                    action_name = f"all_in_{child.action.amount}"
                actions.append(action_name)
        
        return actions
    
    def navigate_by_action(self, action_name: str) -> bool:
        """通过行动名称导航到子节点。"""
        self._expand_node(self._current_node)
        
        raise_types = (ActionType.RAISE, ActionType.RAISE_SMALL, ActionType.RAISE_BIG)
        for child in self._current_node.children:
            if child.action is not None:
                child_action_name = child.action.action_type.value
                if child.action.action_type in raise_types:
                    child_action_name = f"raise_{child.action.amount}"
                elif child.action.action_type == ActionType.ALL_IN:
                    child_action_name = f"all_in_{child.action.amount}"
                
                if child_action_name == action_name:
                    self._current_node = child
                    self._expand_node(child)
                    return True
        return False
    
    def navigate_to_parent(self) -> bool:
        """导航到父节点。"""
        if self._current_node.parent is None:
            return False
        self._current_node = self._current_node.parent
        return True
    
    def get_node_by_id(self, node_id: str) -> Optional[GameTreeNode]:
        """通过ID获取节点。"""
        return self._nodes.get(node_id)
    
    def get_all_nodes(self) -> List[GameTreeNode]:
        """获取所有已索引的节点。"""
        return list(self._nodes.values())
    
    def get_depth(self, node: Optional[GameTreeNode] = None) -> int:
        """获取节点深度。"""
        if node is None:
            node = self._current_node
        return node.get_depth()
    
    def is_at_root(self) -> bool:
        """判断当前是否在根节点。"""
        return self._current_node == self._root
    
    def reset_to_root(self) -> None:
        """重置导航到根节点。"""
        self._current_node = self._root
    
    def get_legal_actions_for_node(self, node: Optional[GameTreeNode] = None) -> List[Action]:
        """获取节点的合法行动列表（Action对象）。"""
        if node is None:
            node = self._current_node
        
        if node.is_terminal() or node.node_type == NodeType.CHANCE:
            return []
        
        self._expand_node(node)
        return [child.action for child in node.children if child.action is not None]

    def add_child(
        self, 
        parent: GameTreeNode, 
        action: Action,
        stage: Optional[GameStage] = None,
        player: int = -1,
        pot: Optional[int] = None,
        stacks: Optional[List[int]] = None,
        board_cards: Optional[List] = None,
        node_type: NodeType = NodeType.PLAYER
    ) -> GameTreeNode:
        """向指定节点添加子节点（用于测试）。"""
        action_str = self._action_to_string(action)
        node_id = f"{parent.node_id}_{action_str}"
        
        new_action_history = parent.action_history.copy()
        new_action_history.append(action)
        
        child = GameTreeNode(
            node_id=node_id,
            stage=stage if stage is not None else parent.stage,
            player=player,
            action=action,
            parent=parent,
            children=[],
            pot=pot if pot is not None else parent.pot,
            stacks=stacks if stacks is not None else parent.stacks.copy(),
            board_cards=board_cards if board_cards is not None else parent.board_cards.copy(),
            action_history=new_action_history,
            node_type=node_type
        )
        
        parent.children.append(child)
        self._index_node(child)
        
        return child
