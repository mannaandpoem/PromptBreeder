import json
import os
from dataclasses import dataclass, field
from typing import Dict
from collections import defaultdict
from datetime import datetime


@dataclass
class ModelUsage:
    """单个模型的使用统计"""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    call_count: int = 0

    def add_usage(self, input_tokens: int, output_tokens: int):
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.total_tokens += (input_tokens + output_tokens)
        self.call_count += 1


@dataclass
class TokenTracker:
    """Token使用跟踪器"""
    model_usage: Dict[str, ModelUsage] = field(default_factory=lambda: defaultdict(ModelUsage))
    start_time: datetime = field(default_factory=datetime.now)

    def add_usage(self, model: str, input_tokens: int, output_tokens: int):
        self.model_usage[model].add_usage(input_tokens, output_tokens)

    def get_total_usage(self) -> Dict[str, Dict]:
        """获取所有模型的使用统计"""
        usage = {}
        for model, stats in self.model_usage.items():
            usage[model] = {
                "input_tokens": stats.input_tokens,
                "output_tokens": stats.output_tokens,
                "total_tokens": stats.total_tokens,
                "call_count": stats.call_count
            }
        usage['total'] = {"total_cost": self.calculate_estimated_cost()}
        return usage

    def print_usage_report(self):
        """打印使用报告"""
        duration = datetime.now() - self.start_time
        print("\n=== Token Usage Report ===")
        print(f"Duration: {duration}")
        print("\nPer Model Statistics:")

        for model, stats in self.model_usage.items():
            print(f"\n{model}:")
            print(f"  Calls: {stats.call_count}")
            print(f"  Input Tokens: {stats.input_tokens:,}")
            print(f"  Output Tokens: {stats.output_tokens:,}")
            print(f"  Total Tokens: {stats.total_tokens:,}")

        total_cost = self.calculate_estimated_cost()
        if total_cost:
            print(f"\nEstimated Total Cost: ${total_cost:.3f}")

    def calculate_estimated_cost(self) -> float:
        """计算估算成本 (可根据实际价格调整)"""
        prices = {
            "gpt-4o": {"input": 0.0025, "output": 0.010},
            "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
            "deepseek-chat": {"input": 0.00014, "output": 0.00028},
            "claude-3-5-sonnet-v2": {"input": 0.003, "output": 0.015},  # alias of newer 3.5 sonnet
            "claude-3-5-sonnet-20240620": {"input": 0.003, "output": 0.015},
        }

        total_cost = 0.0
        for model, stats in self.model_usage.items():
            if model in prices:
                price = prices[model]
                input_cost = (stats.input_tokens / 1000) * price["input"]
                output_cost = (stats.output_tokens / 1000) * price["output"]
                total_cost += input_cost + output_cost

        return total_cost

    def save_to_json(self, filepath: str):
        """
        将使用统计保存为JSON文件

        Args:
            filepath: str, JSON文件保存路径
        """
        # 创建要保存的数据字典
        data = {
            "timestamp": self.start_time.isoformat(),
            "duration": (datetime.now() - self.start_time).total_seconds(),
            "usage": self.get_total_usage()
        }

        # 确保目标目录存在
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # 保存为JSON文件
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)


# 全局单例实例
_token_tracker = None


def get_token_tracker() -> TokenTracker:
    """获取TokenTracker全局实例"""
    global _token_tracker
    if _token_tracker is None:
        _token_tracker = TokenTracker()
    return _token_tracker