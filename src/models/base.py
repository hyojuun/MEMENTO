import time
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Tuple
import openai
from openai import OpenAI
import anthropic
import sys
import re

# from src.utils import print

# API setting constants
API_MAX_RETRY = 25
API_RETRY_SLEEP = 10


class ModelFactory:
    def __init__(self, model_name, config=None):
        self.model_name = model_name
        self.config = config
        self.model_type = self.get_model_type()
        print(f"Model type: {self.model_type}", "blue")

    def get_model(self):
        if self.model_type == 'openai':
            return OpenAIModel(model_name=self.model_name, config=self.config)
        elif self.model_type == 'anthropic':
            return AnthropicModel(model_name=self.model_name, config=self.config)
        elif self.model_type == 'openrouter':
            return OpenRouterModel(model_name=self.model_name, config=self.config)
        else:
            return VLLMModel(model_name=self.model_name, config=self.config)
        
    def get_model_type(self):
        # OpenRouter 설정이 있는지 확인
        if self.config and self.config.get("use_openrouter", False):
            return 'openrouter'
        elif 'gpt' in self.model_name:
            return 'openai'
        elif 'claude' in self.model_name:
            return 'anthropic'
        else:
            return 'vllm'

class AbstractModel(ABC):
    def __init__(self, model_name):
        self.model_name = model_name
        
    @abstractmethod
    def _set_model(self):
        pass

class VLLMModel(AbstractModel):
    def __init__(self, model_name: str, config: dict):
        super().__init__(model_name)
        self.config = config
        self.url = self.config.get("url", "http://localhost")
        self.port = self.config.get("port", 8000)
        self.model = self._set_model()
    
    def _set_model(self):
        base_url = f"{self.url}:{self.port}/v1"
        print(f"Connecting to vLLM server at {base_url}", "blue")
        return OpenAI(
            base_url = base_url,
            api_key="EMPTY",
        )


    def generate_response(self, messages: list[dict], **kwargs) -> list[str]:
        outputs = []
        num_samples = kwargs.get("n", 1)
        temperature = kwargs.get("temperature", 0.0)
        max_tokens = kwargs.get("max_tokens", 1000)
        stop = kwargs.get("stop", None)
        
        # 요청 전 0.5초 지연 추가
        time.sleep(0.5)
        
        for _ in range(API_MAX_RETRY):
            try:
                response = self.model.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    n=num_samples,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stop=stop,
                )
                outputs = [choice.message.content for choice in response.choices]
                break
            except openai.APIError as e:
                print(f"OpenAI API returned an API Error: {e}", "red")
                print(f"Retrying in {API_RETRY_SLEEP} seconds...", "yellow")
                time.sleep(API_RETRY_SLEEP)
            except Exception as e:
                print(f"Unexpected error: {type(e).__name__}: {e}", "red")
                print(f"Retrying in {API_RETRY_SLEEP} seconds...", "yellow")
                time.sleep(API_RETRY_SLEEP)

        if len(outputs) == 0:
            raise Exception("No response from vLLM server after multiple retries")

        return outputs
    


class OpenRouterModel(AbstractModel):
    def __init__(self, model_name: str, config: dict):
        super().__init__(model_name)
        self.config = config
        self.model = self._set_model()
        
    def _set_model(self):
        print(f"Connecting to OpenRouter API with model {self.model_name}", "blue")
        return OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.config.get("api_key"),
            default_headers={
                "HTTP-Referer": "https://example.com/habitatllm",  # 필요에 따라 수정
                "X-Title": "HabitatLLM Experiment"
            }
        )

    def generate_response(self, messages: list[dict], **kwargs) -> List[str]:
        outputs = []
        num_samples = kwargs.get("n", 1)
        temperature = kwargs.get("temperature", 0.0)
        max_tokens = kwargs.get("max_tokens", 1000)
        stop = kwargs.get("stop", None)
        
        # OpenRouter 특정 파라미터
        transforms = kwargs.get("transforms", None)
        route = kwargs.get("route", None)
        
        for _ in range(API_MAX_RETRY):
            try:
                api_params = {
                    "model": self.model_name,
                    "messages": messages,
                    "n": num_samples,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
                
                if stop:
                    api_params["stop"] = stop
                if transforms:
                    api_params["transforms"] = transforms
                if route:
                    api_params["route"] = route
                
                response = self.model.chat.completions.create(**api_params)
                outputs = [choice.message.content for choice in response.choices]
                
                # 응답 형식 확인 및 수정
                outputs = [self.ensure_proper_response_format(output) for output in outputs]
                break
            except openai.APIError as e:
                print(f"OpenRouter API returned an API Error: {e}", "red")
                print(f"Retrying in {API_RETRY_SLEEP} seconds...", "yellow")
                time.sleep(API_RETRY_SLEEP)
            except Exception as e:
                print(f"Unexpected error: {type(e).__name__}: {e}", "red")
                print(f"Retrying in {API_RETRY_SLEEP} seconds...", "yellow")
                time.sleep(API_RETRY_SLEEP)

        if len(outputs) == 0:
            raise Exception("No response from OpenRouter API after multiple retries")

        return outputs
    
    def ensure_proper_response_format(self, response):
        """
        응답이 올바른 형식인지 확인하고 필요한 경우 수정합니다.
        특히 액션 지시문이 [대괄호] 형식을 따르는지 확인합니다.
        """
        # 대괄호 포맷이 필요하지만 없는 경우
        if "Action" in response and "]" not in response and "[" not in response:
            # 액션 라인 찾기
            lines = response.split('\n')
            for i, line in enumerate(lines):
                if "Action" in line and ":" in line:
                    # 액션 이름과 인자 추출
                    action_parts = line.split(":", 1)
                    if len(action_parts) == 2:
                        action_prefix = action_parts[0].strip()
                        action_content = action_parts[1].strip()
                        
                        # 괄호 형식이 있는지 확인
                        if "(" in action_content and ")" in action_content:
                            # (param) 형식을 [param] 형식으로 변환
                            param_match = re.search(r"(\w+)\s*\(([^)]*)\)", action_content)
                            if param_match:
                                action_name = param_match.group(1)
                                action_args = param_match.group(2)
                                lines[i] = f"{action_prefix}: {action_name}[{action_args}]"
                        else:
                            # 단순히 [대괄호] 추가
                            lines[i] = f"{action_prefix}: {action_content}[]"
            
            response = "\n".join(lines)
        
        return response


class AnthropicModel(AbstractModel):
    def __init__(self, model_name: str, config: dict):
        super().__init__(model_name)
        self.config = config
        self.model = self._set_model()
        
    def _set_model(self):
        return anthropic.Anthropic(api_key=self.config["api_key"])


    def generate_response(self, message_list: list[dict], **kwargs) -> List[str]:
        sys_msg = ""
        if message_list[0]["role"] == "system":
            sys_msg = message_list[0]["content"]
            messages = message_list[1:]

        temperature = kwargs.get("temperature", 0.0)
        max_tokens = kwargs.get("max_tokens", 1000)
        stop = kwargs.get("stop", None)

        outputs = []
        for _ in range(API_MAX_RETRY):
            try:
                response = self.model.messages.create(
                    model=self.model_name,
                    messages=messages,
                    stop_sequences=[stop],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    system=sys_msg,
                )
                outputs = [content.text.strip() for content in response.content]
                break
            except anthropic.APIError as e:
                print(type(e), e)
                time.sleep(API_RETRY_SLEEP)
        if len(outputs) == 0:
            raise Exception("No response from OpenAI API")

        return outputs
    


class OpenAIModel(AbstractModel):
    def __init__(self, model_name: str, config: dict):
        super().__init__(model_name)
        self.config = config
        self.model = self._set_model()
        
    def _set_model(self):
        return OpenAI(api_key=self.config["api_key"])


    def generate_response(self, messages: list[dict], **kwargs) -> List[str]:
        outputs = []
        num_samples = kwargs.get("n", 1)
        temperature = kwargs.get("temperature", 0.0)
        max_tokens = kwargs.get("max_tokens", 1000)
        stop = kwargs.get("stop", None)
        
        for _ in range(API_MAX_RETRY):
            try:
                response = self.model.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    n=num_samples,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stop=stop,
                )
                outputs = [choice.message.content for choice in response.choices]
                break
            except openai.APIError as e:
                # Handle API error here, e.g. retry or log
                print(f"OpenAI API returned an API Error: {e}")
                time.sleep(API_RETRY_SLEEP)

        if len(outputs) == 0:
            raise Exception("No response from OpenAI API")

        return outputs
    
    
if __name__ == "__main__":
    # Test VLLMModel with Meta-Llama-3-8B-Instruct
    config = {
        "url": "http://165.132.144.225",
        "port": 8008
    }
    
    model_factory = ModelFactory(model_name="meta-llama/Meta-Llama-3-8B-Instruct", config=config)
    model = model_factory.get_model()
    
    # Test message
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"}
    ]
    
    # Generate response
    try:
        responses = model.generate_response(
            messages=messages,
            temperature=0.7,
            max_tokens=500
        )
        
        print("\nModel Response:")
        for i, response in enumerate(responses):
            print(f"Response {i+1}:\n{response}\n")
    except Exception as e:
        print(f"Error generating response: {e}")
    