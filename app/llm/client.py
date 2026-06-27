# -*- coding: utf-8 -*-
import logging

logger = logging.getLogger(__name__)


class LLMUnavailable(Exception):
    pass


class LLMClient:
    def __init__(self, settings):
        self.settings = settings
        self._client = None
        if settings.llm_api_key:
            try:
                from openai import OpenAI
                self._client = OpenAI(
                    base_url=settings.llm_base_url,
                    api_key=settings.llm_api_key,
                )
            except Exception as e:
                logger.warning("LLM client init failed: %s", e)
                self._client = None

    @property
    def available(self) -> bool:
        return self._client is not None

    def chat(self, messages, tools=None, stream=False):
        if not self.available:
            raise LLMUnavailable("no LLM client configured")
        kwargs = {"model": self.settings.llm_model, "messages": messages, "stream": stream}
        if tools is not None:
            kwargs["tools"] = tools
        resp = self._client.chat.completions.create(**kwargs)
        if stream:
            return resp  # 调用方自行迭代
        msg = resp.choices[0].message
        return {"content": msg.content or "", "tool_calls": getattr(msg, "tool_calls", None)}
