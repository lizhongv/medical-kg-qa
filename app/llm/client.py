# -*- coding: utf-8 -*-
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
            except Exception:
                self._client = None

    @property
    def available(self) -> bool:
        return self._client is not None

    def chat(self, messages, tools=None, stream=False):
        if not self.available:
            raise LLMUnavailable("no LLM client configured")
        resp = self._client.chat.completions.create(
            model=self.settings.llm_model,
            messages=messages,
            tools=tools,
            stream=stream,
        )
        if stream:
            return resp  # 调用方自行迭代
        msg = resp.choices[0].message
        return {"content": msg.content or "", "tool_calls": getattr(msg, "tool_calls", None)}
