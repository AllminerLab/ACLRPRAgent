import json
import os
import re
from typing import Optional, Generator, Dict, Any

import openai
import requests
from anthropic import Anthropic
from openai import OpenAI
import sys
import logging
from logger import init_logger
from llm_api_config import CLAUDE_KEY_API
import tiktoken
base_dir = os.path.split(os.path.realpath(__file__))[0]
program = os.path.basename(sys.argv[0])
logger = logging.getLogger(program)
if not logger.handlers:
    logger = init_logger('../data/log')

def count_tokens(text: str, model: str) -> int:
    """
    统计文本的token数
    """
    try:
        # 根据模型选择合适的编码器
        if "gpt" in model.lower():
            # GPT模型使用cl100k_base编码
            encoding = tiktoken.get_encoding("cl100k_base")
        elif "claude" in model.lower():
            # Claude模型的token计算方式
            # 注意：Claude的实际token计算可能不同，这里使用近似方法
            encoding = tiktoken.get_encoding("cl100k_base")
        else:
            # 默认编码器
            encoding = tiktoken.get_encoding("cl100k_base")

        return len(encoding.encode(text))
    except Exception as e:
        # 如果无法准确计算，使用粗略估算（1 token ≈ 4字符）
        return len(text) // 4


def stream_openai_response(base_url=None, api_key=None, model=None, prompt=None, temperature=0.7, timeout=300, maxtoken=8096, response_format=None, token_recorder=None,**kwargs):
    """
    流式接收OpenAI响应，解决返回结果较多的问题

    参数:
        api_key: OpenAI API密钥（可选，默认从环境变量读取）
        model: 使用的模型，默认 gpt-3.5-turbo
        **kwargs: 其他参数如 temperature, max_tokens 等

    返回:
        完整的响应文本
    """
    try:
        client = Anthropic(api_key=CLAUDE_KEY_API, base_url="https://chat.cloudapi.vip")
        full_response = ''
        with client.messages.stream(
                model=model,
                max_tokens=maxtoken,
                temperature=temperature,
                timeout=timeout,
                messages=[{"role": "user", "content": prompt}],

        ) as stream:
            for text in stream.text_stream:
                full_response += text
                print(text, end="", flush=True)  # 实时打印

            # 获取最终的usage信息
            final_message = stream.get_final_message()
            prompt_tokens = final_message.usage.input_tokens
            completion_tokens = final_message.usage.output_tokens

            token_recorder.add_record(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens)

            return full_response


    except Exception as e:
        print(f"错误: {e}")
        return None


    except Exception as e:
        print(f"错误: {e}")
        return None



class SeekClient:
    def __init__(self, api_key: Optional[str] = None, model:str="deepseek-chat", base_url: str = "https://api.deepseek.com"):
        """
        初始化DeepSeek客户端

        Args:
            api_key: DeepSeek API密钥，如果不提供则从环境变量DEEPSEEK_API_KEY读取
            base_url: API基础URL
        """
        self.api_key = api_key or os.getenv('DEEPSEEK_API_KEY')
        if not self.api_key:
            raise ValueError(
                "API key is required. Please set DEEPSEEK_API_KEY environment variable or pass api_key parameter.")

        self.base_url = base_url.rstrip('/')
        self.headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        self.model = model


    def chat_stream(self,
                    messages: list,
                    model: str = "deepseek-chat",
                    temperature: float = 1.0,
                    max_tokens: Optional[int] = None,
                    **kwargs) -> Generator[Dict[str, Any], None, None]:
        """
        流式聊天接口

        Args:
            messages: 消息列表，格式: [{"role": "user", "content": "Hello"}]
            model: 模型名称
            temperature: 温度参数
            max_tokens: 最大令牌数
            **kwargs: 其他参数

        Yields:
            Dict: 流式响应数据
        """
        url = f"{self.base_url}/v1/chat/completions"

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "temperature": temperature,
            **kwargs
        }

        if max_tokens:
            payload["max_tokens"] = max_tokens

        # 1022 todo
        # 请求在流中包含用量，兼容OpenAI风格；服务端不支持时会被忽略
        payload["stream_options"] = {"include_usage": True}

        try:
            with requests.post(url, headers=self.headers, json=payload, stream=True) as response:
                response.raise_for_status()

                for line in response.iter_lines(decode_unicode=True):
                    if line.strip():
                        if line.startswith('data: '):
                            data = line[6:]  # 移除 'data: ' 前缀

                            if data.strip() == '[DONE]':
                                break

                            try:
                                chunk = json.loads(data)
                                yield chunk
                            except json.JSONDecodeError:
                                continue

        except requests.exceptions.RequestException as e:
            raise Exception(f"Request failed: {e}")

    def chat_stream_text(self,
                         messages: list,
                         model: str = "deepseek-chat",
                         temperature: float = 1.0,
                         max_tokens: Optional[int] = None,
                         **kwargs) -> Generator[str, None, None]:
        """
        流式聊天接口，只返回文本内容

        Args:
            messages: 消息列表
            model: 模型名称
            temperature: 温度参数
            max_tokens: 最大令牌数
            **kwargs: 其他参数

        Yields:
            str: 流式文本内容
        """
        for chunk in self.chat_stream(messages, model, temperature, max_tokens, **kwargs):
            if 'choices' in chunk and chunk['choices']:
                delta = chunk['choices'][0].get('delta', {})
                if 'content' in delta and delta['content']:
                    yield delta['content']


def deepseek_stream_calling(api_key, prompt, base_url: str = "https://api.deepseek.com", model="deepseek-chat", temperature=0.2, max_tokens=8096, token_recorder=None, **kwargs):
    """高级流式输出示例，展示完整的响应数据"""
    client = SeekClient(api_key=api_key, base_url=base_url)

    messages = [
        {"role": "system", "content": "你是一个专业的医疗需求分析助手。"},
        {"role": "user", "content": prompt}
    ]
    # print(prompt)

    # print("完整流式响应数据:")
    # print("-" * 50)

    try:
        full_content = ""
        for chunk in client.chat_stream(messages, model=model, temperature=temperature, max_tokens=max_tokens):
            # 打印完整的chunk数据（调试用）
            # print(f"Chunk: {json.dumps(chunk, ensure_ascii=False, indent=2)}")

            if 'choices' in chunk and chunk['choices']:
                choice = chunk['choices'][0]

                if 'delta' in choice and 'content' in choice['delta']:
                    content = choice['delta']['content']
                    full_content += content
                    print(content, end='', flush=True)

                # 检查是否完成
                if choice.get('finish_reason') :
                    if choice['finish_reason'] == 'stop':
                        print(f"\n\n调用成功")

            # 捕获用量（不同实现格式可能不同，这里做了多种兼容）
            if token_recorder is not None:
                # 1) 顶层 usage 字段
                if 'usage' in chunk and isinstance(chunk['usage'], dict):
                    token_recorder.add_record(prompt_tokens=chunk['usage'].get('prompt_tokens'),
                                completion_tokens=chunk['usage'].get('completion_tokens'),
                                total_tokens=chunk['usage'].get('total_tokens'))

                # 2) 一些兼容OpenAI的新流式格式会有 type 字段
                #    如 type == 'response.completed' 时附带 usage
                if chunk.get('type') in ('response.completed', 'response.summary'):
                    if 'usage' in chunk and isinstance(chunk['usage'], dict):
                        token_recorder.add_record(prompt_tokens=chunk['usage'].get('prompt_tokens'),
                                                  completion_tokens=chunk['usage'].get('completion_tokens'),
                                                  total_tokens=chunk['usage'].get('total_tokens'))

        # print(f"\n\n完整内容:\n{full_content}")
        return full_content

    except Exception as e:
        return None



def qwen_stream_completion0(content: str, api_key: str, base_url: str="https://dashscope.aliyuncs.com/compatible-mode/v1",
                           enable_thinking: bool = True, model: str='qwen3-max', temperature=0.6,
                           system_prompt: str ='你是一位资深的运动医学专家，专门研究ACL重建术后患者的康复评估') -> dict:
    """
    流式调用 qwen3-max 并解析 JSON 结果

    Args:
        content: 用户输入内容
        enable_thinking: 是否启用思考模式

    Returns:
        解析后的 JSON 字典
    """
    client = OpenAI(api_key=api_key, base_url=base_url)

    # 构建请求参数
    request_params = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content}],
        "stream": True,
        "temperature": temperature
    }

    # qwen3-max 思考模式配置
    if enable_thinking:
        request_params["extra_body"] = {"enable_thinking": True}

    try:
        stream = client.chat.completions.create(**request_params)

        reasoning_content = ""  # 思考过程
        result_text = ""  # 最终回复

        for chunk in stream:
            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta

            # 处理思考内容（qwen3-max 特有）
            if enable_thinking and hasattr(delta, 'reasoning_content') and delta.reasoning_content:
                reasoning_content += delta.reasoning_content
                print(f"\033[90m{delta.reasoning_content}\033[0m", end="", flush=True)  # 灰色显示思考

            # 处理正式回复
            if delta.content:
                result_text += delta.content
                print(delta.content, end="", flush=True)

        print()  # 换行

        if reasoning_content:
            logger.debug(f"思考过程：\n{reasoning_content}")
        # logger.info(f"读取内容如下：\n{result_text}")

        # 解析 JSON（处理 markdown 代码块包裹的情况）
        parsed_text = result_text.strip()
        if parsed_text.startswith("```"):
            # 提取代码块内容
            parts = parsed_text.split("```")
            if len(parts) >= 2:
                parsed_text = parts[1]
                # 移除语言标识符（如 json）
                if parsed_text.startswith("json"):
                    parsed_text = parsed_text[4:]
                elif parsed_text.startswith("\n"):
                    parsed_text = parsed_text[1:]
        parsed_text = parsed_text.strip()

        result = json.loads(parsed_text)
        return result

    except Exception as e:
        logger.error(f"流式调用失败: {e}")
        return None

def qwen_stream_completion(content: str, api_key: str, base_url: str="https://dashscope.aliyuncs.com/compatible-mode/v1",
                           enable_thinking: bool = True, model: str='qwen3-max', temperature=0.6,
                           system_prompt: str ='你是一位资深的运动医学专家，专门研究ACL重建术后患者的康复评估', api_attempt_times: int=3):

    for i in range(api_attempt_times):
        response = single_qwen_stream_completion(content=content,
                                                 api_key=api_key,
                                                 base_url=base_url,
                                                 enable_thinking=enable_thinking,
                                                 model=model,
                                                 temperature=temperature,
                                                 system_prompt=system_prompt)
        if response:
            logger.info("LLM API调用成功！")
            return response
        else:
            logger.info("LLM API异常，重新调用中...")

    return None

def single_qwen_stream_completion(content: str, api_key: str, base_url: str="https://dashscope.aliyuncs.com/compatible-mode/v1",
                           enable_thinking: bool = True, model: str='qwen3-max', temperature=0.6,
                           system_prompt: str ='你是一位资深的运动医学专家，专门研究ACL重建术后患者的康复评估') -> dict:
    """
    流式调用 qwen3-max 并解析 JSON 结果

    Args:
        content: 用户输入内容
        enable_thinking: 是否启用思考模式

    Returns:
        解析后的 JSON 字典
    """
    client = OpenAI(api_key=api_key, base_url=base_url)

    # 构建请求参数
    request_params = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content}],
        "stream": True,
        "temperature": temperature
    }

    # qwen3-max 思考模式配置
    if enable_thinking:
        request_params["extra_body"] = {"enable_thinking": True}

    try:
        stream = client.chat.completions.create(**request_params)

        reasoning_content = ""  # 思考过程
        result_text = ""  # 最终回复

        for chunk in stream:
            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta

            # 处理思考内容（qwen3-max 特有）
            if enable_thinking and hasattr(delta, 'reasoning_content') and delta.reasoning_content:
                reasoning_content += delta.reasoning_content
                print(f"\033[90m{delta.reasoning_content}\033[0m", end="", flush=True)  # 灰色显示思考

            # 处理正式回复
            if delta.content:
                result_text += delta.content
                print(delta.content, end="", flush=True)

        print()  # 换行

        # if reasoning_content:
        #     logger.debug(f"思考过程：\n{reasoning_content}")
        # logger.info(f"读取内容如下：\n{result_text}")

        # ==========================================================
        # 🚀 核心修复：鲁棒的 JSON 提取与解析逻辑
        # ==========================================================

        parsed_result = None
        clean_text = result_text.strip()

        # 方案 A: 尝试直接解析（应对纯 JSON 返回）
        try:
            parsed_result = json.loads(clean_text)
        except json.JSONDecodeError:
            # 方案 B: 使用正则提取 Markdown 代码块中的 JSON
            # 匹配 ```json {...} ``` 或 ``` {...} ```
            try:
                # 寻找第一个 { 开头，最后一个 } 结尾的字符串，跨行匹配
                match = re.search(r'\{.*\}', clean_text, re.DOTALL)
                if match:
                    json_str = match.group()
                    parsed_result = json.loads(json_str)
            except Exception as e:
                logger.warning(f"正则提取JSON失败: {e}")

        # 如果最终解析成功，返回字典
        if parsed_result and isinstance(parsed_result, dict):
            return parsed_result

        # ==========================================================
        # 🛡️ 兜底策略：如果解析失败，不要返回空，而是返回包含原始文本的字典
        # ==========================================================
        logger.warning(f"警告：模型返回内容无法解析为JSON，返回原始文本。原始内容预览: {clean_text[:50]}...")

        # 尝试构造一个伪造的成功响应，避免后续流程崩溃
        return None

    except Exception as e:
        logger.error(f"流式调用失败: {e}")
        return None

if __name__ == "__main__":
    qwen_stream_completion(
        api_key='sk-2d407625a6904a50baca3f79cb7d3b1a',
        content="请用JSON格式返回今天的天气信息",
        enable_thinking=True)