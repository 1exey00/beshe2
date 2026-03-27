"""CFR变体选择器模块。

本模块提供不同CFR变体的实现，包括：
- 标准CFR
- CFR+（正遗憾值截断）
- LCFR（线性加权CFR）
- DCFR（折扣CFR）

主要组件：
- CFRVariant: CFR变体枚举
- CFRVariantConfig: CFR变体配置
- CFRVariantSelector: CFR变体选择器
"""

from dataclasses import dataclass
from enum import Enum
from typing import Tuple


class CFRVariant(Enum):
    """CFR变体枚举。
    
    定义了四种CFR变体：
    - STANDARD: 标准CFR算法
    - CFR_PLUS: CFR+，使用正遗憾值截断
    - LCFR: 线性CFR，使用线性迭代加权
    - DCFR: 折扣CFR，使用折扣因子
    """
    STANDARD = "standard"
    CFR_PLUS = "cfr_plus"
    LCFR = "lcfr"
    DCFR = "dcfr"


@dataclass
class CFRVariantConfig:
    """CFR变体配置。
    
    Attributes:
        variant: CFR变体类型
        regret_floor: 遗憾值下限（CFR+使用0）
        discount_alpha: DCFR的alpha参数（正遗憾折扣指数）
        discount_beta: DCFR的beta参数（负遗憾折扣指数）
        discount_gamma: DCFR的gamma参数（策略折扣指数）
    """
    variant: CFRVariant = CFRVariant.CFR_PLUS
    regret_floor: float = 0.0
    discount_alpha: float = 1.5
    discount_beta: float = 0.0
    discount_gamma: float = 2.0
    
    def __post_init__(self):
        """验证配置参数。"""
        if self.discount_alpha < 0:
            raise ValueError(f"discount_alpha必须非负，当前值: {self.discount_alpha}")
        if self.discount_beta < 0:
            raise ValueError(f"discount_beta必须非负，当前值: {self.discount_beta}")
        if self.discount_gamma < 0:
            raise ValueError(f"discount_gamma必须非负，当前值: {self.discount_gamma}")



class CFRVariantSelector:
    """CFR变体选择器。
    
    根据配置提供不同CFR变体的权重计算方法。
    """
    
    def __init__(self, config: CFRVariantConfig = None):
        """初始化CFR变体选择器。
        
        Args:
            config: CFR变体配置，如果为None则使用默认配置
        """
        self.config = config or CFRVariantConfig()
    
    def compute_lcfr_weight(self, iteration: int) -> float:
        """计算LCFR线性权重。
        
        LCFR（Linear CFR）使用线性迭代加权，
        权重与迭代次数成正比，使后期迭代的贡献更大。
        
        权重公式: w(t) = t
        
        Args:
            iteration: 当前迭代次数（从1开始）
            
        Returns:
            线性权重（等于迭代次数）
            
        Raises:
            ValueError: 如果迭代次数小于1
        """
        if iteration < 1:
            raise ValueError(f"迭代次数必须大于等于1，当前值: {iteration}")
        return float(iteration)
    
    def compute_dcfr_discount(self, iteration: int, 
                               current_iteration: int) -> Tuple[float, float, float]:
        """计算DCFR折扣因子。
        
        DCFR（Discounted CFR）对历史遗憾值和策略应用折扣，
        使近期迭代的贡献更大。
        
        折扣公式:
        - 正遗憾折扣: t^alpha / (t^alpha + 1)
        - 负遗憾折扣: t^beta / (t^beta + 1)
        - 策略折扣: (t/T)^gamma
        
        Args:
            iteration: 样本的迭代次数（从1开始）
            current_iteration: 当前迭代次数（从1开始）
            
        Returns:
            (正遗憾折扣, 负遗憾折扣, 策略折扣) 的元组
            
        Raises:
            ValueError: 如果迭代次数无效
        """
        if iteration < 1:
            raise ValueError(f"样本迭代次数必须大于等于1，当前值: {iteration}")
        if current_iteration < 1:
            raise ValueError(f"当前迭代次数必须大于等于1，当前值: {current_iteration}")
        if iteration > current_iteration:
            raise ValueError(
                f"样本迭代次数({iteration})不能大于当前迭代次数({current_iteration})"
            )
        
        t = float(iteration)
        T = float(current_iteration)
        alpha = self.config.discount_alpha
        beta = self.config.discount_beta
        gamma = self.config.discount_gamma
        
        # 正遗憾折扣: t^alpha / (t^alpha + 1)
        t_alpha = t ** alpha
        positive_discount = t_alpha / (t_alpha + 1.0)
        
        # 负遗憾折扣: t^beta / (t^beta + 1)
        t_beta = t ** beta
        negative_discount = t_beta / (t_beta + 1.0)
        
        # 策略折扣: (t/T)^gamma
        strategy_discount = (t / T) ** gamma
        
        return positive_discount, negative_discount, strategy_discount
    
    def get_iteration_weight(self, iteration: int, 
                              current_iteration: int = None) -> float:
        """获取迭代权重。
        
        根据配置的CFR变体返回相应的迭代权重。
        
        Args:
            iteration: 样本的迭代次数
            current_iteration: 当前迭代次数（DCFR需要）
            
        Returns:
            迭代权重
        """
        variant = self.config.variant
        
        if variant == CFRVariant.STANDARD:
            # 标准CFR：所有迭代权重相等
            return 1.0
        
        elif variant == CFRVariant.CFR_PLUS:
            # CFR+：所有迭代权重相等（正遗憾截断在RegretProcessor中处理）
            return 1.0
        
        elif variant == CFRVariant.LCFR:
            # LCFR：线性权重
            return self.compute_lcfr_weight(iteration)
        
        elif variant == CFRVariant.DCFR:
            # DCFR：使用策略折扣作为权重
            if current_iteration is None:
                raise ValueError("DCFR变体需要提供current_iteration参数")
            _, _, strategy_discount = self.compute_dcfr_discount(
                iteration, current_iteration
            )
            return strategy_discount
        
        else:
            raise ValueError(f"未知的CFR变体: {variant}")
