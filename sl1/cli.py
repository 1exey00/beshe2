#!/usr/bin/env python3
"""德州扑克AI训练系统命令行界面。

本模块提供命令行接口，支持以下功能：
- train: 启动训练会话
- evaluate: 运行模型评估
- analyze: 策略分析
- list: 列出检查点
- generate-abstraction: 生成卡牌抽象
- abstraction-info: 显示抽象信息
- help: 显示帮助信息

使用示例：
    python cli.py train --config config.json --episodes 1000
    python cli.py evaluate --model checkpoint.pt --opponent random --games 100
    python cli.py analyze --model checkpoint.pt --output analysis.json
    python cli.py list --checkpoint-dir checkpoints
    python cli.py generate-abstraction --output abstractions/default --flop-buckets 1000
    python cli.py abstraction-info --path abstractions/default
"""

import argparse
import sys
import json
from pathlib import Path
from typing import Optional, List, Dict, Any

from models.core import TrainingConfig, GameState, GameStage, Card
from utils.config_manager import ConfigManager
from utils.checkpoint_manager import CheckpointManager


def create_parser() -> argparse.ArgumentParser:
    """创建命令行参数解析器。
    
    Returns:
        配置好的ArgumentParser对象
    """
    parser = argparse.ArgumentParser(
        prog='poker-ai',
        description='德州扑克AI训练系统 - 训练、评估和分析扑克AI模型',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例:
  %(prog)s train --config config.json --episodes 1000
  %(prog)s evaluate --model checkpoint.pt --opponent random --games 100
  %(prog)s analyze --model checkpoint.pt --output analysis.json
  %(prog)s list --checkpoint-dir checkpoints
        '''
    )
    
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # train 子命令
    train_parser = subparsers.add_parser(
        'train',
        help='启动训练会话',
        description='启动德州扑克AI模型训练'
    )
    train_parser.add_argument(
        '--config', '-c',
        type=str,
        help='训练配置文件路径（JSON格式）'
    )
    train_parser.add_argument(
        '--episodes', '-e',
        type=int,
        help='训练回合数（覆盖配置文件中的值）'
    )
    train_parser.add_argument(
        '--checkpoint-dir', '-d',
        type=str,
        default='checkpoints',
        help='检查点保存目录（默认: checkpoints）'
    )
    train_parser.add_argument(
        '--resume', '-r',
        type=str,
        help='从指定检查点恢复训练'
    )
    train_parser.add_argument(
        '--learning-rate', '-lr',
        type=float,
        help='学习率（覆盖配置文件中的值）'
    )
    train_parser.add_argument(
        '--batch-size', '-b',
        type=int,
        help='批次大小（覆盖配置文件中的值）'
    )
    
    # evaluate 子命令
    eval_parser = subparsers.add_parser(
        'evaluate',
        help='运行模型评估',
        description='评估训练好的AI模型性能'
    )
    eval_parser.add_argument(
        '--model', '-m',
        type=str,
        required=True,
        help='模型检查点文件路径'
    )
    eval_parser.add_argument(
        '--opponent', '-o',
        type=str,
        default='random',
        choices=['random', 'fixed', 'call-only', 'always-fold', 'aggressive'],
        help='对手策略类型（默认: random）'
    )
    eval_parser.add_argument(
        '--games', '-g',
        type=int,
        default=100,
        help='评估对局数（默认: 100）'
    )
    eval_parser.add_argument(
        '--output', '-out',
        type=str,
        help='评估结果输出文件路径（JSON格式）'
    )
    
    # analyze 子命令
    analyze_parser = subparsers.add_parser(
        'analyze',
        help='策略分析',
        description='分析训练好的AI策略'
    )
    analyze_parser.add_argument(
        '--model', '-m',
        type=str,
        required=True,
        help='模型检查点文件路径'
    )
    analyze_parser.add_argument(
        '--state', '-s',
        type=str,
        help='要分析的游戏状态（JSON格式字符串或文件路径）'
    )
    analyze_parser.add_argument(
        '--output', '-out',
        type=str,
        help='分析结果输出文件路径'
    )
    analyze_parser.add_argument(
        '--heatmap',
        action='store_true',
        help='生成策略热图'
    )
    analyze_parser.add_argument(
        '--compare',
        type=str,
        nargs='+',
        help='比较多个模型（提供多个检查点路径）'
    )
    
    # list 子命令
    list_parser = subparsers.add_parser(
        'list',
        help='列出检查点',
        description='列出所有可用的模型检查点'
    )
    list_parser.add_argument(
        '--checkpoint-dir', '-d',
        type=str,
        default='checkpoints',
        help='检查点目录（默认: checkpoints）'
    )
    list_parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='显示详细信息'
    )
    
    # generate-abstraction 子命令
    gen_abs_parser = subparsers.add_parser(
        'generate-abstraction',
        help='生成卡牌抽象',
        description='生成Potential-Aware卡牌抽象，用于减少翻后阶段的状态空间'
    )
    gen_abs_parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='抽象结果输出目录路径'
    )
    gen_abs_parser.add_argument(
        '--flop-buckets',
        type=int,
        default=5000,
        help='翻牌阶段桶数量（默认: 5000）'
    )
    gen_abs_parser.add_argument(
        '--turn-buckets',
        type=int,
        default=5000,
        help='转牌阶段桶数量（默认: 5000）'
    )
    gen_abs_parser.add_argument(
        '--river-buckets',
        type=int,
        default=5000,
        help='河牌阶段桶数量（默认: 5000）'
    )
    gen_abs_parser.add_argument(
        '--preflop-buckets',
        type=int,
        default=169,
        help='翻牌前桶数量（默认: 169，即无抽象）'
    )
    gen_abs_parser.add_argument(
        '--equity-bins',
        type=int,
        default=50,
        help='Equity直方图区间数（默认: 50）'
    )
    gen_abs_parser.add_argument(
        '--potential-aware',
        action='store_true',
        default=True,
        help='使用Potential-Aware抽象（默认: 启用）'
    )
    gen_abs_parser.add_argument(
        '--no-potential-aware',
        action='store_true',
        help='禁用Potential-Aware抽象，使用Distribution-Aware'
    )
    gen_abs_parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='随机种子（默认: 42）'
    )
    gen_abs_parser.add_argument(
        '--kmeans-restarts',
        type=int,
        default=25,
        help='k-means重启次数（默认: 25）'
    )
    gen_abs_parser.add_argument(
        '--workers',
        type=int,
        default=0,
        help='并行工作进程数（默认: 0，使用所有CPU核心）'
    )
    
    # abstraction-info 子命令
    abs_info_parser = subparsers.add_parser(
        'abstraction-info',
        help='显示抽象信息',
        description='显示已生成抽象的配置和统计信息'
    )
    abs_info_parser.add_argument(
        '--path', '-p',
        type=str,
        required=True,
        help='抽象文件目录路径'
    )
    abs_info_parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='显示详细信息'
    )
    abs_info_parser.add_argument(
        '--json',
        action='store_true',
        help='以JSON格式输出'
    )
    
    return parser


def cmd_train(args: argparse.Namespace) -> int:
    """执行训练命令。
    
    Args:
        args: 解析后的命令行参数
        
    Returns:
        退出码（0表示成功）
    """
    # 延迟导入以避免循环依赖
    from training.training_engine import TrainingEngine
    
    config_manager = ConfigManager()
    
    # 加载或创建配置
    if args.config:
        try:
            config = config_manager.load_config(args.config)
            print(f"已加载配置文件: {args.config}")
        except FileNotFoundError:
            print(f"错误: 配置文件不存在: {args.config}")
            return 1
        except ValueError as e:
            print(f"错误: 配置无效: {e}")
            return 1
    else:
        config = config_manager.get_default_config()
        print("使用默认配置")
    
    # 应用命令行参数覆盖（使用 dataclasses.asdict 保留所有参数）
    from dataclasses import asdict
    config_dict = asdict(config)
    
    if args.episodes:
        config_dict['num_episodes'] = args.episodes
    if args.learning_rate:
        config_dict['learning_rate'] = args.learning_rate
    if args.batch_size:
        config_dict['batch_size'] = args.batch_size
    
    # 验证配置
    errors = config_manager.validate_config(config_dict)
    if errors:
        print(f"错误: 配置参数无效:")
        for error in errors:
            print(f"  - {error}")
        return 1
    
    config = TrainingConfig(**config_dict)
    
    # 创建训练引擎
    try:
        engine = TrainingEngine(config, checkpoint_dir=args.checkpoint_dir)
        print(f"训练引擎已初始化")
        print(f"  检查点目录: {args.checkpoint_dir}")
        print(f"  训练回合数: {config.num_episodes}")
        print(f"  学习率: {config.learning_rate}")
        print(f"  批次大小: {config.batch_size}")
        print(f"  检查点间隔: {config.checkpoint_interval}")
        print(f"CFR+SL 参数:")
        print(f"  CFR迭代次数/更新: {config.cfr_iterations_per_update}")
        print(f"  网络训练步数: {config.network_train_steps}")
    except Exception as e:
        print(f"错误: 无法初始化训练引擎: {e}")
        return 1
    
    # 从检查点恢复（如果指定）
    if args.resume:
        try:
            engine.load_checkpoint(args.resume)
            print(f"已从检查点恢复: {args.resume}")
        except FileNotFoundError:
            print(f"错误: 检查点文件不存在: {args.resume}")
            return 1
        except Exception as e:
            print(f"错误: 无法加载检查点: {e}")
            return 1
    
    # 开始训练
    try:
        result = engine.train()
        print("\n训练完成!")
        print(f"  总回合数: {result['total_episodes']}")
        print(f"  最终胜率: {result['win_rate']:.2%}")
        print(f"  平均奖励: {result['avg_reward']:.2f}")
        return 0
    except KeyboardInterrupt:
        print("\n训练被用户中断")
        return 0
    except Exception as e:
        print(f"错误: 训练过程中发生异常: {e}")
        return 1


def cmd_evaluate(args: argparse.Namespace) -> int:
    """执行评估命令。
    
    Args:
        args: 解析后的命令行参数
        
    Returns:
        退出码（0表示成功）
    """
    # 延迟导入
    from analysis.evaluator import (
        Evaluator, RandomStrategy, FixedStrategy, 
        CallOnlyStrategy, AlwaysFoldStrategy
    )
    from models.networks import PolicyNetwork
    from utils.checkpoint_manager import CheckpointManager
    
    # 检查模型文件是否存在
    if not Path(args.model).exists():
        print(f"错误: 模型文件不存在: {args.model}")
        return 1
    
    # 创建对手策略
    opponent_map = {
        'random': RandomStrategy(),
        'fixed': FixedStrategy(prefer_aggressive=False),
        'aggressive': FixedStrategy(prefer_aggressive=True),
        'call-only': CallOnlyStrategy(),
        'always-fold': AlwaysFoldStrategy()
    }
    opponent = opponent_map.get(args.opponent, RandomStrategy())
    
    # 加载模型
    try:
        import torch
        checkpoint_manager = CheckpointManager()
        
        # 先加载检查点以获取网络架构信息
        checkpoint_data = torch.load(args.model, map_location='cpu', weights_only=False)
        
        # 从模型权重推断网络架构
        # 第一层权重形状为 [hidden_dim_0, input_dim]
        first_layer_weight = checkpoint_data['model_state_dict']['network.0.weight']
        hidden_dims = []
        
        # 遍历所有层获取隐藏层维度
        layer_idx = 0
        while f'network.{layer_idx}.weight' in checkpoint_data['model_state_dict']:
            weight = checkpoint_data['model_state_dict'][f'network.{layer_idx}.weight']
            hidden_dims.append(weight.shape[0])
            layer_idx += 2  # 跳过bias层
        
        # 最后一层是输出层，不算隐藏层
        if hidden_dims:
            action_dim = hidden_dims.pop()
            
        model = PolicyNetwork(
            input_dim=370,
            hidden_dims=hidden_dims,
            action_dim=action_dim
        )
        model, _, metadata = checkpoint_manager.load(args.model, model)
        print(f"已加载模型: {args.model}")
        print(f"  网络架构: {hidden_dims}")
        print(f"  训练回合数: {metadata.get('episode_number', 'N/A')}")
        print(f"  训练胜率: {metadata.get('win_rate', 0):.2%}")
    except Exception as e:
        print(f"错误: 无法加载模型: {e}")
        return 1
    
    # 运行评估
    print(f"\n开始评估...")
    print(f"  对手策略: {opponent.name}")
    print(f"  评估对局数: {args.games}")
    
    try:
        evaluator = Evaluator()
        result = evaluator.evaluate(
            model=model,
            opponent=opponent,
            num_games=args.games,
            model_name=Path(args.model).stem
        )
        
        print(f"\n评估结果:")
        print(f"  胜率: {result.win_rate:.2%}")
        print(f"  胜/负/平: {result.wins}/{result.losses}/{result.ties}")
        print(f"  平均盈利: {result.avg_profit:.2f}")
        print(f"  盈利标准差: {result.std_profit:.2f}")
        print(f"  总盈利: {result.total_profit:.2f}")
        
        # 保存结果（如果指定）
        if args.output:
            evaluator.save_results(result, args.output)
            print(f"\n结果已保存到: {args.output}")
        
        return 0
    except Exception as e:
        print(f"错误: 评估过程中发生异常: {e}")
        return 1


def cmd_analyze(args: argparse.Namespace) -> int:
    """执行分析命令。
    
    Args:
        args: 解析后的命令行参数
        
    Returns:
        退出码（0表示成功）
    """
    # 延迟导入
    from analysis.strategy_analyzer import StrategyAnalyzer
    
    # 检查模型文件是否存在
    if not Path(args.model).exists():
        print(f"错误: 模型文件不存在: {args.model}")
        return 1
    
    try:
        analyzer = StrategyAnalyzer()
        analyzer.load_model(args.model)
        print(f"已加载模型: {args.model}")
    except Exception as e:
        print(f"错误: 无法加载模型: {e}")
        return 1
    
    # 比较多个模型
    if args.compare:
        checkpoint_paths = {Path(p).stem: p for p in [args.model] + args.compare}
        
        # 创建一个示例状态用于比较
        state = _create_sample_state()
        
        try:
            comparison = analyzer.compare_strategies(checkpoint_paths, state)
            print("\n策略比较结果:")
            print(f"  状态: {comparison.state_description}")
            for model_name, strategy in comparison.strategies.items():
                print(f"\n  {model_name}:")
                for action, prob in strategy.items():
                    print(f"    {action}: {prob:.2%}")
            
            if args.output:
                analyzer.save_analysis(comparison, args.output)
                print(f"\n结果已保存到: {args.output}")
            
            return 0
        except Exception as e:
            print(f"错误: 策略比较失败: {e}")
            return 1
    
    # 分析特定状态
    if args.state:
        try:
            # 尝试解析为JSON
            if Path(args.state).exists():
                with open(args.state, 'r') as f:
                    state_dict = json.load(f)
            else:
                state_dict = json.loads(args.state)
            
            state = _parse_game_state(state_dict)
        except json.JSONDecodeError:
            print(f"错误: 无效的JSON格式: {args.state}")
            return 1
        except Exception as e:
            print(f"错误: 无法解析游戏状态: {e}")
            return 1
    else:
        # 使用示例状态
        state = _create_sample_state()
        print("使用示例游戏状态进行分析")
    
    try:
        # 获取决策解释
        explanation = analyzer.explain_decision(state)
        
        print("\n决策分析:")
        print(f"  状态描述:\n{explanation.state_description}")
        print(f"\n  行动概率:")
        for ap in explanation.action_probabilities:
            print(f"    {ap.action_type}: {ap.probability:.2%}")
        print(f"\n  推荐行动: {explanation.recommended_action}")
        print(f"  期望价值: {explanation.expected_value:.2f}")
        print(f"\n  决策理由:\n{explanation.reasoning}")
        
        # 生成热图（如果指定）
        if args.heatmap:
            hand_range = _generate_sample_hand_range()
            heatmap = analyzer.generate_strategy_heatmap(hand_range)
            
            heatmap_path = args.output.replace('.json', '_heatmap.png') if args.output else 'strategy_heatmap.png'
            hand_labels = [f"{h[0]}{h[1]}" for h in hand_range[:10]]  # 简化标签
            analyzer.plot_strategy_heatmap(
                heatmap[:10],  # 只显示前10个
                hand_labels=hand_labels,
                save_path=heatmap_path
            )
            print(f"\n热图已保存到: {heatmap_path}")
        
        # 保存结果（如果指定）
        if args.output:
            analyzer.save_analysis(explanation, args.output)
            print(f"\n结果已保存到: {args.output}")
        
        return 0
    except Exception as e:
        print(f"错误: 分析过程中发生异常: {e}")
        return 1


def cmd_list(args: argparse.Namespace) -> int:
    """执行列出检查点命令。
    
    Args:
        args: 解析后的命令行参数
        
    Returns:
        退出码（0表示成功）
    """
    checkpoint_dir = Path(args.checkpoint_dir)
    
    if not checkpoint_dir.exists():
        print(f"检查点目录不存在: {checkpoint_dir}")
        return 0
    
    checkpoint_manager = CheckpointManager(args.checkpoint_dir)
    checkpoints = checkpoint_manager.list_checkpoints()
    
    if not checkpoints:
        print(f"在 {checkpoint_dir} 中没有找到检查点")
        return 0
    
    print(f"找到 {len(checkpoints)} 个检查点:\n")
    
    for i, cp in enumerate(checkpoints, 1):
        print(f"{i}. {Path(cp.path).name}")
        if args.verbose:
            print(f"   路径: {cp.path}")
            print(f"   回合数: {cp.episode_number}")
            print(f"   时间: {cp.timestamp}")
            print(f"   胜率: {cp.win_rate:.2%}")
            print(f"   平均奖励: {cp.avg_reward:.2f}")
            print()
        else:
            print(f"   回合: {cp.episode_number} | 胜率: {cp.win_rate:.2%} | 时间: {cp.timestamp.strftime('%Y-%m-%d %H:%M')}")
    
    return 0


def cmd_generate_abstraction(args: argparse.Namespace) -> int:
    """执行生成抽象命令。
    
    生成Potential-Aware卡牌抽象，用于减少翻后阶段的状态空间。
    显示进度条和预计剩余时间。
    
    Args:
        args: 解析后的命令行参数
        
    Returns:
        退出码（0表示成功）
        
    需求：14.1
    """
    # 延迟导入以避免循环依赖
    from abstraction.data_classes import AbstractionConfig
    from abstraction.card_abstraction import CardAbstraction
    import time
    
    # 验证参数
    if args.flop_buckets <= 0:
        print(f"错误: 翻牌桶数量必须为正数，当前值: {args.flop_buckets}")
        return 1
    if args.turn_buckets <= 0:
        print(f"错误: 转牌桶数量必须为正数，当前值: {args.turn_buckets}")
        return 1
    if args.river_buckets <= 0:
        print(f"错误: 河牌桶数量必须为正数，当前值: {args.river_buckets}")
        return 1
    if args.preflop_buckets <= 0 or args.preflop_buckets > 169:
        print(f"错误: 翻牌前桶数量必须在1-169之间，当前值: {args.preflop_buckets}")
        return 1
    if args.equity_bins <= 0:
        print(f"错误: Equity区间数必须为正数，当前值: {args.equity_bins}")
        return 1
    
    # 确定是否使用Potential-Aware
    use_potential_aware = args.potential_aware and not args.no_potential_aware
    
    # 创建配置
    try:
        config = AbstractionConfig(
            preflop_buckets=args.preflop_buckets,
            flop_buckets=args.flop_buckets,
            turn_buckets=args.turn_buckets,
            river_buckets=args.river_buckets,
            equity_bins=args.equity_bins,
            kmeans_restarts=args.kmeans_restarts,
            use_potential_aware=use_potential_aware,
            random_seed=args.seed,
            num_workers=args.workers,
        )
    except ValueError as e:
        print(f"错误: 配置参数无效: {e}")
        return 1
    
    # 显示配置信息
    print("=" * 60)
    print("卡牌抽象生成")
    print("=" * 60)
    print(f"\n配置参数:")
    print(f"  输出目录: {args.output}")
    print(f"  翻牌前桶数: {config.preflop_buckets}")
    print(f"  翻牌桶数: {config.flop_buckets}")
    print(f"  转牌桶数: {config.turn_buckets}")
    print(f"  河牌桶数: {config.river_buckets}")
    print(f"  Equity区间数: {config.equity_bins}")
    print(f"  k-means重启次数: {config.kmeans_restarts}")
    print(f"  Potential-Aware: {'是' if config.use_potential_aware else '否'}")
    print(f"  随机种子: {config.random_seed}")
    print(f"  工作进程数: {config.num_workers if config.num_workers > 0 else '自动'}")
    print()
    
    # 创建抽象管理器
    abstraction = CardAbstraction(config)
    
    # 开始生成
    print("开始生成抽象...")
    print("注意: 这可能需要较长时间，取决于桶数量和CPU性能")
    print()
    
    start_time = time.time()
    
    try:
        # 显示进度信息
        stages = ['河牌', '转牌', '翻牌', '翻牌前']
        total_stages = len(stages)
        
        print(f"[阶段 0/{total_stages}] 初始化...")
        
        # 生成抽象（内部会按阶段处理）
        result = abstraction.generate_abstraction()
        
        elapsed_time = time.time() - start_time
        
        print(f"\n[完成] 抽象生成完成!")
        print(f"  总耗时: {_format_time(elapsed_time)}")
        
    except KeyboardInterrupt:
        print("\n\n抽象生成被用户中断")
        return 1
    except Exception as e:
        print(f"\n错误: 抽象生成失败: {e}")
        return 1
    
    # 保存结果
    print(f"\n保存抽象结果到: {args.output}")
    try:
        abstraction.save(args.output)
        print("保存成功!")
    except Exception as e:
        print(f"错误: 保存失败: {e}")
        return 1
    
    # 显示统计信息
    print("\n" + "=" * 60)
    print("抽象统计信息")
    print("=" * 60)
    
    stats = abstraction.get_abstraction_stats()
    
    for stage in ['preflop', 'flop', 'turn', 'river']:
        stage_stats = stats['stages'].get(stage, {})
        stage_name = {
            'preflop': '翻牌前',
            'flop': '翻牌',
            'turn': '转牌',
            'river': '河牌'
        }.get(stage, stage)
        
        print(f"\n{stage_name}阶段:")
        print(f"  桶数量: {stage_stats.get('count', 0)}")
        print(f"  平均桶大小: {stage_stats.get('avg_size', 0):.2f}")
        print(f"  最大桶大小: {stage_stats.get('max_size', 0)}")
        print(f"  最小桶大小: {stage_stats.get('min_size', 0)}")
    
    if stats.get('wcss'):
        print(f"\nWCSS (聚类质量指标):")
        for stage, wcss in stats['wcss'].items():
            stage_name = {
                'preflop': '翻牌前',
                'flop': '翻牌',
                'turn': '转牌',
                'river': '河牌'
            }.get(stage, stage)
            print(f"  {stage_name}: {wcss:.4f}")
    
    print(f"\n生成耗时: {_format_time(stats.get('generation_time', 0))}")
    print("\n抽象生成完成!")
    
    return 0


def cmd_abstraction_info(args: argparse.Namespace) -> int:
    """执行显示抽象信息命令。
    
    显示已生成抽象的配置和统计信息。
    
    Args:
        args: 解析后的命令行参数
        
    Returns:
        退出码（0表示成功）
        
    需求：15.4
    """
    # 延迟导入
    from abstraction.card_abstraction import CardAbstraction
    from abstraction.abstraction_evaluator import AbstractionEvaluator
    
    # 检查路径是否存在
    if not Path(args.path).exists():
        print(f"错误: 抽象目录不存在: {args.path}")
        return 1
    
    # 加载抽象
    try:
        abstraction = CardAbstraction()
        result = abstraction.load(args.path)
        print(f"已加载抽象: {args.path}")
    except FileNotFoundError as e:
        print(f"错误: 抽象文件不存在: {e}")
        return 1
    except Exception as e:
        print(f"错误: 无法加载抽象: {e}")
        return 1
    
    # 获取统计信息
    stats = abstraction.get_abstraction_stats()
    
    # JSON格式输出
    if args.json:
        import json
        evaluator = AbstractionEvaluator()
        report = evaluator.generate_report_dict(result)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0
    
    # 普通格式输出
    print("\n" + "=" * 60)
    print("抽象配置信息")
    print("=" * 60)
    
    config = stats.get('config', {})
    print(f"\n配置参数:")
    print(f"  翻牌前桶数: {config.get('preflop_buckets', 'N/A')}")
    print(f"  翻牌桶数: {config.get('flop_buckets', 'N/A')}")
    print(f"  转牌桶数: {config.get('turn_buckets', 'N/A')}")
    print(f"  河牌桶数: {config.get('river_buckets', 'N/A')}")
    print(f"  Equity区间数: {config.get('equity_bins', 'N/A')}")
    print(f"  k-means重启次数: {config.get('kmeans_restarts', 'N/A')}")
    print(f"  Potential-Aware: {'是' if config.get('use_potential_aware', False) else '否'}")
    print(f"  随机种子: {config.get('random_seed', 'N/A')}")
    
    print("\n" + "=" * 60)
    print("抽象统计信息")
    print("=" * 60)
    
    stage_names = {
        'preflop': '翻牌前',
        'flop': '翻牌',
        'turn': '转牌',
        'river': '河牌'
    }
    
    for stage in ['preflop', 'flop', 'turn', 'river']:
        stage_stats = stats['stages'].get(stage, {})
        stage_name = stage_names.get(stage, stage)
        
        print(f"\n{stage_name}阶段:")
        print(f"  桶数量: {stage_stats.get('count', 0)}")
        print(f"  平均桶大小: {stage_stats.get('avg_size', 0):.2f}")
        print(f"  最大桶大小: {stage_stats.get('max_size', 0)}")
        print(f"  最小桶大小: {stage_stats.get('min_size', 0)}")
        
        if args.verbose:
            # 显示更详细的信息
            evaluator = AbstractionEvaluator()
            mapping = result._get_mapping(stage)
            if mapping is not None:
                bucket_stats = evaluator.get_bucket_size_distribution(mapping)
                print(f"  桶大小标准差: {bucket_stats.std_size:.2f}")
    
    # WCSS信息
    wcss = stats.get('wcss', {})
    if wcss:
        print(f"\nWCSS (聚类质量指标):")
        for stage, value in wcss.items():
            stage_name = stage_names.get(stage, stage)
            print(f"  {stage_name}: {value:.4f}")
    
    # 生成时间
    gen_time = stats.get('generation_time', 0)
    if gen_time > 0:
        print(f"\n生成耗时: {_format_time(gen_time)}")
    
    # 详细模式下显示更多信息
    if args.verbose:
        evaluator = AbstractionEvaluator()
        report = evaluator.generate_report(result)
        
        print(f"\n总桶数: {report.total_buckets}")
        print(f"压缩比: {report.compression_ratio:.2f}x")
        
        # 估算内存使用
        total_mappings = 0
        if result.preflop_mapping is not None:
            total_mappings += len(result.preflop_mapping)
        if result.flop_mapping is not None:
            total_mappings += len(result.flop_mapping)
        if result.turn_mapping is not None:
            total_mappings += len(result.turn_mapping)
        if result.river_mapping is not None:
            total_mappings += len(result.river_mapping)
        
        # 假设每个映射条目占用4字节（int32）
        memory_mb = (total_mappings * 4) / (1024 * 1024)
        print(f"估算内存使用: {memory_mb:.2f} MB")
    
    return 0


def _format_time(seconds: float) -> str:
    """格式化时间显示。
    
    Args:
        seconds: 秒数
        
    Returns:
        格式化的时间字符串
    """
    if seconds < 60:
        return f"{seconds:.1f}秒"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}分{secs:.1f}秒"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}小时{minutes}分{secs:.1f}秒"


def _create_sample_state() -> GameState:
    """创建一个示例游戏状态用于分析。
    
    Returns:
        示例GameState对象
    """
    return GameState(
        player_hands=[
            (Card(14, 'h'), Card(13, 'h')),  # AK同花
            (Card(7, 'd'), Card(2, 'c'))     # 对手手牌（虚拟）
        ],
        community_cards=[Card(12, 'h'), Card(11, 'h'), Card(5, 's')],  # 翻牌
        pot=100,
        player_stacks=[950, 950],
        current_bets=[0, 0],
        button_position=0,
        stage=GameStage.FLOP,
        action_history=[],
        current_player=0
    )


def _parse_game_state(state_dict: Dict[str, Any]) -> GameState:
    """从字典解析游戏状态。
    
    Args:
        state_dict: 状态字典
        
    Returns:
        GameState对象
    """
    # 解析手牌
    player_hands = []
    for hand in state_dict.get('player_hands', []):
        cards = tuple(Card(c['rank'], c['suit']) for c in hand)
        player_hands.append(cards)
    
    # 解析公共牌
    community_cards = [
        Card(c['rank'], c['suit']) 
        for c in state_dict.get('community_cards', [])
    ]
    
    # 解析游戏阶段
    stage_map = {
        'preflop': GameStage.PREFLOP,
        'flop': GameStage.FLOP,
        'turn': GameStage.TURN,
        'river': GameStage.RIVER
    }
    stage = stage_map.get(state_dict.get('stage', 'preflop'), GameStage.PREFLOP)
    
    return GameState(
        player_hands=player_hands,
        community_cards=community_cards,
        pot=state_dict.get('pot', 0),
        player_stacks=state_dict.get('player_stacks', [1000, 1000]),
        current_bets=state_dict.get('current_bets', [0, 0]),
        button_position=state_dict.get('button_position', 0),
        stage=stage,
        action_history=[],
        current_player=state_dict.get('current_player', 0)
    )


def _generate_sample_hand_range() -> List[tuple]:
    """生成示例手牌范围用于热图。
    
    Returns:
        手牌组合列表
    """
    hands = []
    # 生成一些典型手牌
    high_cards = [14, 13, 12, 11, 10]  # A, K, Q, J, 10
    suits = ['h', 'd', 'c', 's']
    
    for i, rank1 in enumerate(high_cards):
        for rank2 in high_cards[i:]:
            # 同花
            hands.append((Card(rank1, 'h'), Card(rank2, 'h')))
            # 不同花
            if rank1 != rank2:
                hands.append((Card(rank1, 'h'), Card(rank2, 'd')))
    
    return hands[:20]  # 限制数量


def main(argv: Optional[List[str]] = None) -> int:
    """CLI主入口点。
    
    Args:
        argv: 命令行参数列表（如果为None则使用sys.argv）
        
    Returns:
        退出码
    """
    parser = create_parser()
    args = parser.parse_args(argv)
    
    # 如果没有指定命令，显示帮助
    if not args.command:
        parser.print_help()
        return 0
    
    # 执行对应的命令
    command_handlers = {
        'train': cmd_train,
        'evaluate': cmd_evaluate,
        'analyze': cmd_analyze,
        'list': cmd_list,
        'generate-abstraction': cmd_generate_abstraction,
        'abstraction-info': cmd_abstraction_info,
    }
    
    handler = command_handlers.get(args.command)
    if handler:
        return handler(args)
    else:
        parser.print_help()
        return 1


if __name__ == '__main__':
    sys.exit(main())
