import openai
from abc import ABC, abstractmethod
import asyncio
import time
from functools import partial

import os
from typing import Dict, List, Optional
from omegaconf import DictConfig, OmegaConf


from src.models.base import ModelFactory


API_MAX_RETRY = 20
API_RETRY_SLEEP = 20

def generate_message(multimodal_prompt, image_detail="auto"):
    # Converts the multimodal prompt to the OpenAI format.
    content = []
    for prompt_type, prompt_value in multimodal_prompt:
        if prompt_type == "text":
            message_item = {"type": "text", "text": prompt_value}
        else:
            message_item = {
                "type": "image_url",
                "image_url": {
                    "url": prompt_value,
                    "detail": image_detail,
                },
            }
        content.append(message_item)
    return {"role": "user", "content": content}


class HabitatModel:
    def __init__(self, conf: DictConfig):
        self.llm_conf = conf
        self.generation_params = self.llm_conf.generation_params
        self.model_name = self.llm_conf.model_name
        # try:
        #     # self.openai_api_key = os.getenv("OPENAI_API_KEY")
        #     self.openai_api_key = self.llm_conf.openai_api_key
        #     assert len(self.openai_api_key) > 0, ValueError("No OPENAI_API_KEY keys provided")
        # except Exception:
        #     raise ValueError("No OPENAI API keys provided")
        
        # super().__init__(
        #     model_name=self.model_name, 
        #     api_key=self.openai_api_key
        # )
        
        self.model_factory = ModelFactory(self.model_name, self.llm_conf)
        self.model = self.model_factory.get_model()
        
        self.verbose = self.llm_conf.verbose
        self.message_history: List[Dict] = []
        self.keep_message_history = self.llm_conf.keep_message_history
        
    def _validate_conf(self):
        if self.generation_params.stream:
            raise ValueError("Streaming not supported")
    
    def generate(
        self, 
        prompt, 
        stop: Optional[str] = None,
        max_length: Optional[int] = None,
        generation_args=None,
        request_timeout: int = 40,
        ):
        """
        Generate a response autoregressively.
        :param prompt: A string with the input to the language model.
        :param image: Image input
        :param stop: A string that determines when to stop generation
        :param max_length: The max number of tokens to generate.
        :param request_timeout: maximum time before timeout.
        :param generation_args: contains arguments like the grammar definition. We don't use this here
        """

        params = OmegaConf.to_object(self.generation_params)

        # Override stop if provided
        if stop is None and len(self.generation_params.stop) > 0:
            stop = self.generation_params.stop
        params["stop"] = stop

        # Override max_length if provided
        if max_length is not None:
            params["max_tokens"] = max_length

        messages = self.message_history.copy()
        # Add system message if no messages
        if len(messages) == 0:
            messages.append({"role": "system", "content": self.llm_conf.system_message})

        params["request_timeout"] = request_timeout
        if type(prompt) is str:
            # Add current message
            messages.append({"role": "user", "content": prompt})

        else:
            # Multimodal prompt
            image_detail = "low"  # high/low/auto
            messages.append(generate_message(prompt, image_detail=image_detail))
        
        org_response = self.model.generate_response(messages, **params)
        text_response = org_response[0]

        self.response = text_response

        # Update message history
        if self.keep_message_history:
            self.message_history = messages.copy()
            self.message_history.append({"role": "assistant", "content": text_response})

        if stop is not None:
            self.response = text_response.split(stop)[0]
        return self.response
    
        