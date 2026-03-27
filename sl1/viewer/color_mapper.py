"""策略颜色映射器模块。

本模块实现了策略到颜色的映射功能：
- 将策略概率映射为颜色
- 支持不同行动的颜色编码
- 支持混合策略的颜色混合
- 支持条状组合显示策略分布
"""

from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass

from viewer.models import BarSegment


@dataclass(frozen=True)
class Color:
    """颜色类，表示RGBA颜色。
    
    Attributes:
        r: 红色分量 (0-255)
        g: 绿色分量 (0-255)
        b: 蓝色分量 (0-255)
        a: 透明度分量 (0-255)，255为完全不透明
    """
    r: int
    g: int
    b: int
    a: int = 255
    
    def __post_init__(self):
        """验证颜色值范围。"""
        for name, value in [('r', self.r), ('g', self.g), ('b', self.b), ('a', self.a)]:
            if not 0 <= value <= 255:
                raise ValueError(f"颜色分量 {name} 必须在 0-255 范围内，当前值: {value}")
    
    def to_tuple(self) -> Tuple[int, int, int, int]:
        """转换为RGBA元组。"""
        return (self.r, self.g, self.b, self.a)
    
    def to_rgb_tuple(self) -> Tuple[int, int, int]:
        """转换为RGB元组。"""
        return (self.r, self.g, self.b)
    
    def to_hex(self) -> str:
        """转换为十六进制颜色字符串。"""
        return f"#{self.r:02x}{self.g:02x}{self.b:02x}"
    
    def to_hex_with_alpha(self) -> str:
        """转换为带透明度的十六进制颜色字符串。"""
        return f"#{self.r:02x}{self.g:02x}{self.b:02x}{self.a:02x}"
    
    @classmethod
    def from_hex(cls, hex_str: str) -> 'Color':
        """从十六进制字符串创建颜色。
        
        Args:
            hex_str: 十六进制颜色字符串（如"#FF0000"或"FF0000"）
            
        Returns:
            Color对象
        """
        hex_str = hex_str.lstrip('#')
        if len(hex_str) == 6:
            r = int(hex_str[0:2], 16)
            g = int(hex_str[2:4], 16)
            b = int(hex_str[4:6], 16)
            return cls(r, g, b)
        elif len(hex_str) == 8:
            r = int(hex_str[0:2], 16)
            g = int(hex_str[2:4], 16)
            b = int(hex_str[4:6], 16)
            a = int(hex_str[6:8], 16)
            return cls(r, g, b, a)
        else:
            raise ValueError(f"无效的十六进制颜色字符串: {hex_str}")


class StrategyColorMapper:
    """策略颜色映射器 - 将策略概率映射为颜色。
    
    颜色编码规则（根据设计文档需求6）：
    - 蓝色系: 弃牌（FOLD）
    - 绿色系: 过牌/跟注（CHECK/CALL）
    - 红色系: 加注类动作（RAISE_SMALL浅红、RAISE_BIG深红、ALL_IN最深红）
    - 混合色: 混合策略（根据概率混合颜色）
    
    这是一个纯函数类，相同的输入总是产生相同的输出。
    """
    
    # 预定义的行动颜色（根据设计文档的颜色方案）
    # FOLD: 蓝色 (R < 100, G > 100, B > 200)
    # CHECK/CALL: 绿色 (R < 100, G > 150, B < 150)
    # RAISE类: 红色 (R > 180, G < 150, B < 150)
    ACTION_COLORS = {
        'FOLD': Color(66, 133, 244),       # 蓝色 - 弃牌
        'CHECK': Color(52, 168, 83),       # 绿色 - 过牌
        'CALL': Color(52, 168, 83),        # 绿色 - 跟注
        'CHECK/CALL': Color(52, 168, 83),  # 绿色 - 过牌/跟注
        'RAISE': Color(234, 67, 53),       # 红色 - 加注
        'RAISE_SMALL': Color(255, 138, 128),  # 浅红色 - 小加注
        'RAISE_BIG': Color(234, 67, 53),      # 深红色 - 大加注
        'RAISE_POT': Color(234, 67, 53),      # 深红色 - 底池加注
        'ALL_IN': Color(183, 28, 28),         # 最深红色 - 全下
    }
    
    # 默认颜色（用于未知行动）
    DEFAULT_COLOR = Color(128, 128, 128)  # 灰色
    
    # 背景颜色（用于无策略或禁用的格子）
    BACKGROUND_COLOR = Color(248, 249, 250)  # 浅灰色
    DISABLED_COLOR = Color(200, 200, 200)    # 禁用灰色
    
    def __init__(self):
        """初始化颜色映射器。"""
        pass
    
    def get_cell_color(self, strategy: Dict[str, float]) -> Color:
        """根据策略分布返回单元格颜色。
        
        颜色计算规则：
        1. 如果策略为空，返回背景色
        2. 如果某个行动概率超过95%，返回该行动的纯色
        3. 否则，根据概率加权混合所有行动的颜色
        
        这是一个纯函数：相同的输入总是产生相同的输出。
        
        Args:
            strategy: 策略概率字典 {行动: 概率}
            
        Returns:
            混合后的颜色
        """
        if not strategy:
            return self.BACKGROUND_COLOR
        
        # 检查是否有主导行动（概率超过95%）
        for action, prob in strategy.items():
            if prob >= 0.95:
                return self.get_action_color(action)
        
        # 混合颜色
        return self._blend_colors(strategy)
    
    def get_action_color(self, action: str) -> Color:
        """获取特定行动的颜色。
        
        颜色规则（根据设计文档需求6.2-6.4）：
        - FOLD: 蓝色系 (R < 100, G > 100, B > 200)
        - CHECK/CALL: 绿色系 (R < 100, G > 150, B < 150)
        - RAISE类: 红色系 (R > 180, G < 150, B < 150)
        
        这是一个纯函数：相同的输入总是产生相同的输出。
        
        Args:
            action: 行动名称
            
        Returns:
            行动对应的颜色
        """
        # 标准化行动名称（转大写，处理常见变体）
        normalized = action.upper().strip()
        
        # 处理FOLD
        if normalized == 'FOLD':
            return self.ACTION_COLORS['FOLD']
        
        # 处理CHECK/CALL类
        if normalized in ['CHECK', 'CALL', 'CHECK/CALL']:
            return self.ACTION_COLORS['CHECK/CALL']
        
        # 处理ALL_IN（最深红色）
        if normalized == 'ALL_IN' or normalized == 'ALLIN':
            return self.ACTION_COLORS['ALL_IN']
        
        # 处理RAISE类
        if normalized.startswith('RAISE'):
            if 'SMALL' in normalized:
                return self.ACTION_COLORS['RAISE_SMALL']
            elif 'BIG' in normalized or 'POT' in normalized:
                return self.ACTION_COLORS['RAISE_BIG']
            else:
                return self.ACTION_COLORS.get('RAISE', self.DEFAULT_COLOR)
        
        return self.ACTION_COLORS.get(normalized, self.DEFAULT_COLOR)
    
    def get_disabled_color(self) -> Color:
        """获取禁用状态的颜色。
        
        Returns:
            禁用颜色
        """
        return self.DISABLED_COLOR
    
    def get_background_color(self) -> Color:
        """获取背景颜色。
        
        Returns:
            背景颜色
        """
        return self.BACKGROUND_COLOR
    
    def _blend_colors(self, strategy: Dict[str, float]) -> Color:
        """根据策略概率混合颜色。
        
        使用加权平均混合颜色分量。
        
        Args:
            strategy: 策略概率字典
            
        Returns:
            混合后的颜色
        """
        total_r = 0.0
        total_g = 0.0
        total_b = 0.0
        total_weight = 0.0
        
        for action, prob in strategy.items():
            if prob > 0:
                color = self.get_action_color(action)
                total_r += color.r * prob
                total_g += color.g * prob
                total_b += color.b * prob
                total_weight += prob
        
        if total_weight < 1e-10:
            return self.BACKGROUND_COLOR
        
        # 归一化（处理概率之和不为1的情况）
        r = int(round(total_r / total_weight))
        g = int(round(total_g / total_weight))
        b = int(round(total_b / total_weight))
        
        # 确保在有效范围内
        r = max(0, min(255, r))
        g = max(0, min(255, g))
        b = max(0, min(255, b))
        
        return Color(r, g, b)
    
    def get_color_legend(self) -> Dict[str, Color]:
        """获取颜色图例。
        
        Returns:
            行动到颜色的映射字典
        """
        return {
            '弃牌 (Fold)': self.ACTION_COLORS['FOLD'],
            '过牌/跟注 (Check/Call)': self.ACTION_COLORS['CHECK/CALL'],
            '小加注 (Raise Small)': self.ACTION_COLORS['RAISE_SMALL'],
            '大加注 (Raise Big)': self.ACTION_COLORS['RAISE_BIG'],
            '全下 (All-In)': self.ACTION_COLORS['ALL_IN'],
        }
    
    def strategy_to_color_intensity(
        self, 
        strategy: Dict[str, float],
        action: str
    ) -> float:
        """获取特定行动在策略中的颜色强度。
        
        Args:
            strategy: 策略概率字典
            action: 目标行动
            
        Returns:
            颜色强度 (0.0-1.0)
        """
        if not strategy:
            return 0.0
        
        # 标准化行动名称
        normalized = action.upper().strip()
        
        # 查找匹配的行动
        for act, prob in strategy.items():
            if act.upper().strip() == normalized:
                return min(1.0, max(0.0, prob))
        
        return 0.0
    
    def get_bar_segments(
        self,
        strategy: Dict[str, float],
        action_order: Optional[List[str]] = None
    ) -> List[BarSegment]:
        """将策略转换为条状段列表。
        
        根据设计文档需求6.5和6.6：
        - 条状宽度按照各动作概率比例分配
        - 概率为0的动作不显示
        
        Args:
            strategy: 动作概率字典 {action: probability}
            action_order: 动作显示顺序（可选），如果不提供则按策略字典顺序
            
        Returns:
            条状段列表，按顺序排列，概率为0的动作不包含
        """
        if not strategy:
            return []
        
        # 确定动作顺序
        if action_order:
            actions = [a for a in action_order if a in strategy]
        else:
            actions = list(strategy.keys())
        
        # 计算非零概率的总和（用于归一化）
        total_prob = sum(prob for prob in strategy.values() if prob > 0)
        
        if total_prob < 1e-10:
            return []
        
        segments = []
        for action in actions:
            prob = strategy.get(action, 0.0)
            
            # 跳过概率为0的动作（需求6.6）
            if prob <= 0:
                continue
            
            # 获取动作颜色
            color = self.get_action_color(action)
            
            # 计算宽度比例（需求6.5：按概率比例分配宽度）
            # 宽度比例等于概率值（归一化后）
            width_ratio = prob / total_prob
            
            segment = BarSegment(
                action=action,
                probability=prob,
                color=color.to_rgb_tuple(),
                width_ratio=width_ratio
            )
            segments.append(segment)
        
        return segments
