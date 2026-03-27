#!/usr/bin/env python3
"""
从验证数据场景文件中提取训练所需的必要信息

提取的字段：
- 输入特征：hero_equity, range_equity, solver_equity, eqr
- 目标值：ev, action_ev, strategy
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def extract_scenario_data(scenario: Dict[str, Any]) -> Dict[str, Any]:
    """
    从单个场景中提取训练所需的数据
    
    参数:
        scenario: 原始场景数据
    
    返回:
        提取后的精简数据
    """
    # 提取输入特征
    hero_equity_data = scenario.get("heroEquity", {})
    range_equity_data = scenario.get("rangeEquity", {})
    
    extracted = {
        "id": scenario.get("id"),
        # 输入特征
        "hero_equity": hero_equity_data.get("equity"),
        "range_equity": range_equity_data.get("equity"),
        "solver_equity": scenario.get("solverEquity"),
        "eqr": scenario.get("eqr"),
        # 目标值
        "ev": scenario.get("ev"),
        "action_ev": scenario.get("actionEV"),
        "strategy": scenario.get("strategy"),
    }
    
    return extracted


def validate_extracted_data(data: Dict[str, Any]) -> bool:
    """
    验证提取的数据是否有效
    
    返回:
        True 如果数据有效，否则 False
    """
    required_fields = ["hero_equity", "range_equity", "solver_equity", "eqr", "ev", "action_ev", "strategy"]
    
    for field in required_fields:
        value = data.get(field)
        if value is None:
            return False
        
        # 检查数值字段
        if field in ["hero_equity", "range_equity", "solver_equity", "eqr", "ev"]:
            if not isinstance(value, (int, float)):
                return False
            # 检查NaN和Inf
            import math
            if math.isnan(value) or math.isinf(value):
                return False
        
        # 检查字典字段
        if field in ["action_ev", "strategy"]:
            if not isinstance(value, dict):
                return False
            for v in value.values():
                if not isinstance(v, (int, float)):
                    return False
                if math.isnan(v) or math.isinf(v):
                    return False
    
    return True


def process_file(input_path: Path) -> List[Dict[str, Any]]:
    """
    处理单个JSON文件
    
    返回:
        提取的有效数据列表
    """
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    scenarios = data.get("scenarios", [])
    extracted_list = []
    
    for scenario in scenarios:
        extracted = extract_scenario_data(scenario)
        if validate_extracted_data(extracted):
            extracted_list.append(extracted)
    
    return extracted_list


def main():
    parser = argparse.ArgumentParser(description="从验证数据中提取训练数据")
    parser.add_argument("--input-dir", type=str, default="experiments/validation_data",
                        help="输入数据目录")
    parser.add_argument("--output-dir", type=str, default="experiments/training_data",
                        help="输出数据目录")
    parser.add_argument("--max-files", type=int, default=None,
                        help="最大处理文件数")
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 查找所有batch文件
    json_files = sorted([f for f in input_dir.glob("*.json") if "batch" in f.name])
    
    if args.max_files:
        json_files = json_files[:args.max_files]
    
    logger.info(f"找到 {len(json_files)} 个场景文件")
    
    all_data = []
    total_scenarios = 0
    valid_scenarios = 0
    
    for i, json_file in enumerate(json_files):
        logger.info(f"处理文件 [{i+1}/{len(json_files)}]: {json_file.name}")
        
        extracted = process_file(json_file)
        all_data.extend(extracted)
        
        # 统计
        with open(json_file, 'r') as f:
            data = json.load(f)
            total_scenarios += len(data.get("scenarios", []))
        valid_scenarios += len(extracted)
    
    logger.info(f"\n提取完成:")
    logger.info(f"  总场景数: {total_scenarios}")
    logger.info(f"  有效场景数: {valid_scenarios}")
    logger.info(f"  跳过场景数: {total_scenarios - valid_scenarios}")
    
    # 保存提取的数据
    output_file = output_dir / "extracted_training_data.json"
    with open(output_file, 'w') as f:
        json.dump({
            "metadata": {
                "total_scenarios": total_scenarios,
                "valid_scenarios": valid_scenarios,
                "source_files": len(json_files)
            },
            "data": all_data
        }, f)
    
    logger.info(f"\n数据已保存到: {output_file}")
    
    # 打印数据统计
    if all_data:
        import numpy as np
        
        hero_equities = [d["hero_equity"] for d in all_data]
        range_equities = [d["range_equity"] for d in all_data]
        solver_equities = [d["solver_equity"] for d in all_data]
        eqrs = [d["eqr"] for d in all_data]
        evs = [d["ev"] for d in all_data]
        
        logger.info("\n数据统计:")
        logger.info(f"  Hero Equity: mean={np.mean(hero_equities):.4f}, std={np.std(hero_equities):.4f}, "
                   f"min={np.min(hero_equities):.4f}, max={np.max(hero_equities):.4f}")
        logger.info(f"  Range Equity: mean={np.mean(range_equities):.4f}, std={np.std(range_equities):.4f}, "
                   f"min={np.min(range_equities):.4f}, max={np.max(range_equities):.4f}")
        logger.info(f"  Solver Equity: mean={np.mean(solver_equities):.4f}, std={np.std(solver_equities):.4f}, "
                   f"min={np.min(solver_equities):.4f}, max={np.max(solver_equities):.4f}")
        logger.info(f"  EQR: mean={np.mean(eqrs):.4f}, std={np.std(eqrs):.4f}, "
                   f"min={np.min(eqrs):.4f}, max={np.max(eqrs):.4f}")
        logger.info(f"  EV: mean={np.mean(evs):.4f}, std={np.std(evs):.4f}, "
                   f"min={np.min(evs):.4f}, max={np.max(evs):.4f}")


if __name__ == "__main__":
    main()
