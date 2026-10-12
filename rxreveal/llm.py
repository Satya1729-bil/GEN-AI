"""Answer generation with the Google Gemini API (free tier works)."""

from __future__ import annotations

from typing import Protocol


class LLM(Protocol):
    def generate(self, system: str, prompt: str) -> str: ...


class LLMError(RuntimeError):
    """Raised with a message that is safe to show to the user."""


class GeminiLLM:
    def __init__(self, api_key: str, model: str, temperature: float = 0.2):
        if not api_key:
            raise LLMError("GEMINI_API_KEY is not set.")
        from google import genai  # imported here so tests stay light

        self._client = genai.Client(api_key=api_key)
        self.model = model
        self.temperature = temperature

    def generate(self, system: str, prompt: str) -> str:
        from google.genai import types

        try:
            response = self._client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system,
                    temperature=self.temperature,
                    max_output_tokens=1024,
                ),
            )
        except Exception as exc:  # the SDK raises several error types
            message = str(exc)
            if "429" in message or "RESOURCE_EXHAUSTED" in message:
                raise LLMError(
                    "The Gemini free-tier limit was reached. Wait a minute and try again."
                ) from exc
            if "API key" in message or "401" in message or "403" in message:
                raise LLMError("The Gemini API key was rejected. Check GEMINI_API_KEY.") from exc
            if "404" in message or "not found" in message.lower():
                raise LLMError(
                    f"The model '{self.model}' was not found. Set GEMINI_MODEL to a current model."
                ) from exc
            raise LLMError(f"Gemini request failed: {message[:300]}") from exc

        text = (response.text or "").strip()
        if not text:
            raise LLMError("Gemini returned an empty answer. Try rephrasing the question.")
        return text
