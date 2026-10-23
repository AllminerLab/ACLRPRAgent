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
    "Count text tokens."
    try:
        # Choose the tokenizer for the model.
        if "gpt" in model.lower():
            # Use cl100k_base for GPT models.
            encoding = tiktoken.get_encoding("cl100k_base")
        elif "claude" in model.lower():
            # Estimate Claude token usage.
            # Actual Claude tokenization may differ; this is an approximation.
            encoding = tiktoken.get_encoding("cl100k_base")
        else:
            # Default tokenizer
            encoding = tiktoken.get_encoding("cl100k_base")

        return len(encoding.encode(text))
    except Exception as e:
        # Fallback estimate: approximately 4 characters per token.
        return len(text) // 4


def stream_openai_response(base_url=None, api_key=None, model=None, prompt=None, temperature=0.7, timeout=300, maxtoken=8096, response_format=None, token_recorder=None,**kwargs):
    "Receive a streamed OpenAI response and return the complete response text. api_key is optional and otherwise read from the environment; model defaults to gpt-3.5-turbo; kwargs accepts temperature, max_tokens and other options."
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
                print(text, end="", flush=True)  # Print streaming output immediately.

            # Read final usage information.
            final_message = stream.get_final_message()
            prompt_tokens = final_message.usage.input_tokens
            completion_tokens = final_message.usage.output_tokens

            token_recorder.add_record(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens)

            return full_response


    except Exception as e:
        print(f"""Error: {e}""")
        return None


    except Exception as e:
        print(f"""Error: {e}""")
        return None



class SeekClient:
    def __init__(self, api_key: Optional[str] = None, model:str="deepseek-chat", base_url: str = "https://api.deepseek.com"):
        "Initialize the DeepSeek client with api_key and base_url. If api_key is omitted, read DEEPSEEK_API_KEY from the environment."
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
        "Yield streamed response dictionaries for messages, model, temperature, max_tokens and additional kwargs. Each message contains role and content."
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
        # Request usage in the stream; unsupported servers may ignore this option.
        payload["stream_options"] = {"include_usage": True}

        try:
            with requests.post(url, headers=self.headers, json=payload, stream=True) as response:
                response.raise_for_status()

                for line in response.iter_lines(decode_unicode=True):
                    if line.strip():
                        if line.startswith('data: '):
                            data = line[6:]  # Remove the 'data: ' prefix.

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
        "Yield only text from a streamed response for messages, model, temperature, max_tokens and additional kwargs."
        for chunk in self.chat_stream(messages, model, temperature, max_tokens, **kwargs):
            if 'choices' in chunk and chunk['choices']:
                delta = chunk['choices'][0].get('delta', {})
                if 'content' in delta and delta['content']:
                    yield delta['content']


def deepseek_stream_calling(api_key, prompt, base_url: str = "https://api.deepseek.com", model="deepseek-chat", temperature=0.2, max_tokens=8096, token_recorder=None, **kwargs):
    "Demonstrate streaming with complete response metadata."
    client = SeekClient(api_key=api_key, base_url=base_url)

    messages = [
        {"role": "system", "content": "You are a professional medical requirements analysis assistant."},
        {"role": "user", "content": prompt}
    ]
    # print(prompt)

    # print("Full streaming response:")
    # print("-" * 50)

    try:
        full_content = ""
        for chunk in client.chat_stream(messages, model=model, temperature=temperature, max_tokens=max_tokens):
            # Print complete chunks for debugging.
            # print(f"Chunk: {json.dumps(chunk, ensure_ascii=False, indent=2)}")

            if 'choices' in chunk and chunk['choices']:
                choice = chunk['choices'][0]

                if 'delta' in choice and 'content' in choice['delta']:
                    content = choice['delta']['content']
                    full_content += content
                    print(content, end='', flush=True)

                # Check whether the response is complete.
                if choice.get('finish_reason') :
                    if choice['finish_reason'] == 'stop':
                        print(f"""\n\nCall succeeded""")

            # Handle usage information from different API response formats.
            if token_recorder is not None:
                # 1. Top-level usage field
                if 'usage' in chunk and isinstance(chunk['usage'], dict):
                    token_recorder.add_record(prompt_tokens=chunk['usage'].get('prompt_tokens'),
                                completion_tokens=chunk['usage'].get('completion_tokens'),
                                total_tokens=chunk['usage'].get('total_tokens'))

                # 2. Some OpenAI-compatible streams include a type field.
                # For example, response.completed may carry usage data.
                if chunk.get('type') in ('response.completed', 'response.summary'):
                    if 'usage' in chunk and isinstance(chunk['usage'], dict):
                        token_recorder.add_record(prompt_tokens=chunk['usage'].get('prompt_tokens'),
                                                  completion_tokens=chunk['usage'].get('completion_tokens'),
                                                  total_tokens=chunk['usage'].get('total_tokens'))

        # print(f"\n\nFull content:\n{full_content}")
        return full_content

    except Exception as e:
        return None



def qwen_stream_completion0(content: str, api_key: str, base_url: str="https://dashscope.aliyuncs.com/compatible-mode/v1",
                           enable_thinking: bool = True, model: str='qwen3-max', temperature=0.6,
                           system_prompt: str ="You are a senior sports medicine specialist specializing in post-ACLR rehabilitation assessment.") -> dict:
    "Stream a Qwen response and return parsed JSON. content is the user prompt; enable_thinking selects reasoning mode."
    client = OpenAI(api_key=api_key, base_url=base_url)

    # Build request parameters.
    request_params = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content}],
        "stream": True,
        "temperature": temperature
    }

    # Configure qwen3-max reasoning mode.
    if enable_thinking:
        request_params["extra_body"] = {"enable_thinking": True}

    try:
        stream = client.chat.completions.create(**request_params)

        reasoning_content = ""  # Reasoning content
        result_text = ""  # Final response

        for chunk in stream:
            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta

            # Handle qwen3-max reasoning content.
            if enable_thinking and hasattr(delta, 'reasoning_content') and delta.reasoning_content:
                reasoning_content += delta.reasoning_content
                print(f"\033[90m{delta.reasoning_content}\033[0m", end="", flush=True)  # Display reasoning in gray.

            # Handle the final response.
            if delta.content:
                result_text += delta.content
                print(delta.content, end="", flush=True)

        print()  # Start a new line.

        if reasoning_content:
            logger.debug(f"""Reasoning:\n{reasoning_content}""")
        # logger.info(f"Extracted content:\n{result_text}")

        # Parse JSON, allowing Markdown code fences.
        parsed_text = result_text.strip()
        if parsed_text.startswith("```"):
            # Extract fenced content.
            parts = parsed_text.split("```")
            if len(parts) >= 2:
                parsed_text = parts[1]
                # Remove the language identifier, such as json.
                if parsed_text.startswith("json"):
                    parsed_text = parsed_text[4:]
                elif parsed_text.startswith("\n"):
                    parsed_text = parsed_text[1:]
        parsed_text = parsed_text.strip()

        result = json.loads(parsed_text)
        return result

    except Exception as e:
        logger.error(f"""Streaming call failed: {e}""")
        return None

def qwen_stream_completion(content: str, api_key: str, base_url: str="https://dashscope.aliyuncs.com/compatible-mode/v1",
                           enable_thinking: bool = True, model: str='qwen3-max', temperature=0.6,
                           system_prompt: str ="You are a senior sports medicine specialist specializing in post-ACLR rehabilitation assessment.", api_attempt_times: int=3):

    for i in range(api_attempt_times):
        response = single_qwen_stream_completion(content=content,
                                                 api_key=api_key,
                                                 base_url=base_url,
                                                 enable_thinking=enable_thinking,
                                                 model=model,
                                                 temperature=temperature,
                                                 system_prompt=system_prompt)
        if response:
            logger.info("LLM API call succeeded")
            return response
        else:
            logger.info("LLM API error; retrying")

    return None

def single_qwen_stream_completion(content: str, api_key: str, base_url: str="https://dashscope.aliyuncs.com/compatible-mode/v1",
                           enable_thinking: bool = True, model: str='qwen3-max', temperature=0.6,
                           system_prompt: str ="You are a senior sports medicine specialist specializing in post-ACLR rehabilitation assessment.") -> dict:
    "Stream a Qwen response and return parsed JSON. content is the user prompt; enable_thinking selects reasoning mode."
    client = OpenAI(api_key=api_key, base_url=base_url)

    # Build request parameters.
    request_params = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content}],
        "stream": True,
        "temperature": temperature
    }

    # Configure qwen3-max reasoning mode.
    if enable_thinking:
        request_params["extra_body"] = {"enable_thinking": True}

    try:
        stream = client.chat.completions.create(**request_params)

        reasoning_content = ""  # Reasoning content
        result_text = ""  # Final response

        for chunk in stream:
            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta

            # Handle qwen3-max reasoning content.
            if enable_thinking and hasattr(delta, 'reasoning_content') and delta.reasoning_content:
                reasoning_content += delta.reasoning_content
                print(f"\033[90m{delta.reasoning_content}\033[0m", end="", flush=True)  # Display reasoning in gray.

            # Handle the final response.
            if delta.content:
                result_text += delta.content
                print(delta.content, end="", flush=True)

        print()  # Start a new line.

        # if reasoning_content:
        #     logger.debug(f"Reasoning:\n{reasoning_content}")
        # logger.info(f"Extracted content:\n{result_text}")

        # ==========================================================
        # Robust JSON extraction and parsing
        # ==========================================================

        parsed_result = None
        clean_text = result_text.strip()

        # Approach A: Parse a plain JSON response directly.
        try:
            parsed_result = json.loads(clean_text)
        except json.JSONDecodeError:
            # Approach B: Extract JSON from a Markdown code fence.
            # Match ```json {...} ``` or ``` {...} ```.
            try:
                # Match from the first opening brace to the final closing brace across lines.
                match = re.search(r'\{.*\}', clean_text, re.DOTALL)
                if match:
                    json_str = match.group()
                    parsed_result = json.loads(json_str)
            except Exception as e:
                logger.warning(f"""Regex-based JSON extraction failed: {e}""")

        # Return a dictionary if parsing succeeds.
        if parsed_result and isinstance(parsed_result, dict):
            return parsed_result

        # ==========================================================
        # Fallback: preserve the original text in a dictionary when parsing fails.
        # ==========================================================
        logger.warning(f"""Warning: Model response could not be parsed as JSON; returning text. Preview: {clean_text[:50]}...""")

        # Wrap the raw response to preserve the caller interface.
        return None

    except Exception as e:
        logger.error(f"""Streaming call failed: {e}""")
        return None

if __name__ == "__main__":
    qwen_stream_completion(
        api_key='sk-2d407625a6904a50baca3f79cb7d3b1a',
        content="Return today's weather information in JSON format.",
        enable_thinking=True)