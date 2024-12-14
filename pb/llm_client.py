from openai import OpenAI
from dataclasses import dataclass
from typing import Dict, List, Optional
import functools
import tiktoken
import time
import re

from pb.token_manager import get_token_tracker


@dataclass
class TokenStats:
    """Token统计信息"""
    input_tokens: int
    response_tokens: int
    total_tokens: int


@dataclass
class LLMResponse:
    """LLM响应结果，包含响应内容和token统计"""
    content: Optional[str]
    token_stats: TokenStats


def check_tokens(max_input_tokens: Optional[int] = None,
                 encoding_name: str = "cl100k_base"):
    """
    装饰器: 用于检查和计算LLM输入消息的token数量，并返回token统计
    """

    def calculate_message_tokens(messages: List[Dict], encoding) -> int:
        """计算消息列表的总token数"""
        total_tokens = 0
        content = messages
        tokens = len(encoding.encode(content))
        total_tokens += tokens
        # for message in messages:
        #     # content = message.get("content", "")
        #     # role = message.get("role", "")
        #     content = message
        #     tokens = len(encoding.encode(content))
        #     total_tokens += tokens
        return total_tokens

    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # 获取token跟踪器实例
            token_tracker = get_token_tracker()

            # 获取model参数
            model = kwargs.get("model", "unknown-model")

            messages = None
            if args:
                messages = args[0]
                model = args[1] if len(args) > 1 else model
            elif "messages" in kwargs:
                messages = kwargs["messages"]

            if not messages:
                response = await func(*args, **kwargs)
                # return LLMResponse(
                #     content=response,
                #     token_stats=TokenStats(0, 0, 0)
                # )
                return response

            encoding = tiktoken.get_encoding(encoding_name)
            input_tokens = calculate_message_tokens(messages, encoding)

            if max_input_tokens and input_tokens > max_input_tokens:
                raise ValueError(
                    f"Input messages token count ({input_tokens}) "
                    f"exceeds limit ({max_input_tokens})"
                )

            response = await func(*args, **kwargs)
            response_tokens = len(encoding.encode(response)) if response else 0
            total_tokens = input_tokens + response_tokens

            # 更新全局统计
            token_tracker.add_usage(model, input_tokens, response_tokens)

            # return LLMResponse(
            #     content=response,
            #     token_stats=TokenStats(
            #         input_tokens=input_tokens,
            #         response_tokens=response_tokens,
            #         total_tokens=total_tokens
            #     )
            # )
            return response
        return wrapper

    return decorator

def extract_content(xml_string, tag):
    # 构建正则表达式，匹配指定的标签内容
    pattern = rf'<{tag}>(.*?)</{tag}>'
    match = re.search(pattern, xml_string, re.DOTALL)  # 使用 re.DOTALL 以匹配换行符
    return match.group(1).strip() if match else None





#
# # 使用示例
# async def main():
#     messages = [
#         {"role": "user", "content": "Hello, how are you?"}
#     ]
#
#     # 多次调用API测试
#     for model in ["gpt-4o", "gpt-4o", "gpt-4o-mini"]:
#         result = await responser(messages, model=model)
#         print(f"\nModel: {model}")
#         print(f"Response: {result.content}")
#         print(f"Input tokens: {result.token_stats.input_tokens}")
#         print(f"Response tokens: {result.token_stats.response_tokens}")
#
#     # 打印最终统计信息
#     get_token_tracker().print_usage_report()
#
#
# if __name__ == "__main__":
#     import asyncio
#
#     asyncio.run(main())