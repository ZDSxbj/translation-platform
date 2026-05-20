"""LLM API client wrapper. Supports OpenAI-compatible APIs."""

from openai import OpenAI


class LLMService:
    """Thin wrapper around the OpenAI-compatible chat completions API."""

    def __init__(self, api_key: str, base_url: str, model: str,
                 max_tokens: int = 8192, temperature: float = 0.0,
                 timeout: float = 600.0):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)

    def chat(self, messages: list[dict], **kwargs) -> str:
        """Send a chat completion request and return the response text."""
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            temperature=kwargs.get("temperature", self.temperature),
        )
        if resp.choices and resp.choices[0].message:
            return resp.choices[0].message.content or ""
        return ""

    def check_connection(self) -> dict:
        """Quick connectivity check (returns ok / error info)."""
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=1,
                temperature=0,
            )
            return {"ok": True, "model": self.model}
        except Exception as e:
            return {"ok": False, "error": str(e)}
