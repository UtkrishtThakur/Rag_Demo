"""Local Ollama LLM client for RAG answer generation.

This module is intentionally thin:
- Connects to Ollama running on the host or another reachable endpoint
- Sends prompts and returns generated text
- Adds debug logs for request timing and failures
- Supports retries and health checks

Expected environment variables (typically set in config.py / .env):
- OLLAMA_HOST: base URL for Ollama, e.g.
    - http://host.docker.internal:11434
    - http://172.17.0.1:11434
    - http://localhost:11434 (only if running outside Docker)
- OLLAMA_MODEL: model name, e.g. qwen3:8b
- OLLAMA_TEMPERATURE: float, e.g. 0.2
- OLLAMA_TIMEOUT: request timeout in seconds, e.g. 120
- OLLAMA_TOP_P: nucleus sampling, e.g. 0.9
- OLLAMA_TOP_K: top-k sampling, e.g. 40
- OLLAMA_NUM_CTX: context window, e.g. 4096 or 8192
- OLLAMA_NUM_PREDICT: max tokens to generate, e.g. 512
- OLLAMA_REPEAT_PENALTY: e.g. 1.1
- OLLAMA_RETRIES: number of retries, e.g. 1 or 2

Usage:
    from llm import OllamaClient

    llm = OllamaClient()
    answer = llm.generate("Hello")
    print(answer)
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Generator, Iterable, Optional

try:
    import ollama
except Exception as exc:  # pragma: no cover
    ollama = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


DEFAULT_OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://host.docker.internal:11434")
DEFAULT_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", os.getenv("MODEL_NAME", "qwen3:8b"))
DEFAULT_TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", "0.2"))
DEFAULT_TOP_P = float(os.getenv("OLLAMA_TOP_P", "0.9"))
DEFAULT_TOP_K = int(os.getenv("OLLAMA_TOP_K", "40"))
DEFAULT_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "4096"))
DEFAULT_NUM_PREDICT = int(os.getenv("OLLAMA_NUM_PREDICT", "512"))
DEFAULT_REPEAT_PENALTY = float(os.getenv("OLLAMA_REPEAT_PENALTY", "1.1"))
DEFAULT_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "120"))
DEFAULT_RETRIES = int(os.getenv("OLLAMA_RETRIES", "1"))
DEFAULT_STREAM = os.getenv("OLLAMA_STREAM", "false").lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class GenerationResult:
    text: str
    model: str
    duration_s: float
    prompt_eval_count: Optional[int] = None
    eval_count: Optional[int] = None
    total_duration_ns: Optional[int] = None


class OllamaClient:
    """Thin wrapper around the Ollama Python client with logging and retries."""

    def __init__(
        self,
        host: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        num_ctx: int | None = None,
        num_predict: int | None = None,
        repeat_penalty: float | None = None,
        timeout: int | None = None,
        retries: int | None = None,
        stream: bool | None = None,
        system_prompt: str | None = None,
    ) -> None:
        if ollama is None:  # pragma: no cover
            raise RuntimeError(f"ollama package is required but could not be imported: {_IMPORT_ERROR}")

        self.host = (host or DEFAULT_OLLAMA_HOST).strip()
        self.model = (model or DEFAULT_OLLAMA_MODEL).strip()
        self.temperature = DEFAULT_TEMPERATURE if temperature is None else float(temperature)
        self.top_p = DEFAULT_TOP_P if top_p is None else float(top_p)
        self.top_k = DEFAULT_TOP_K if top_k is None else int(top_k)
        self.num_ctx = DEFAULT_NUM_CTX if num_ctx is None else int(num_ctx)
        self.num_predict = DEFAULT_NUM_PREDICT if num_predict is None else int(num_predict)
        self.repeat_penalty = DEFAULT_REPEAT_PENALTY if repeat_penalty is None else float(repeat_penalty)
        self.timeout = DEFAULT_TIMEOUT if timeout is None else int(timeout)
        self.retries = DEFAULT_RETRIES if retries is None else int(retries)
        self.stream = DEFAULT_STREAM if stream is None else bool(stream)
        self.system_prompt = system_prompt or (
            "You are a precise RAG assistant. Answer only from the provided context. "
            "If the answer is not in the context, say you could not find it. "
            "Always mention source document names and page numbers when available."
        )

        self.client = ollama.Client(host=self.host)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.debug(
            "Initialized OllamaClient host=%s model=%s temp=%.3f top_p=%.3f top_k=%s num_ctx=%s num_predict=%s timeout=%s retries=%s stream=%s",
            self.host,
            self.model,
            self.temperature,
            self.top_p,
            self.top_k,
            self.num_ctx,
            self.num_predict,
            self.timeout,
            self.retries,
            self.stream,
        )

    def health_check(self) -> Dict[str, Any]:
        """Check whether Ollama is reachable and the model can be listed."""
        self.logger.debug("Health check: connecting to Ollama at %s", self.host)
        try:
            tags = self.client.list()
            models = []
            for item in tags.get("models", []):
                if isinstance(item, dict):
                    models.append(item.get("name", ""))
                else:
                    models.append(getattr(item, "model", getattr(item, "name", "")))

            available = self.model in models if models else None
            self.logger.debug("Health check success. models=%s", models)
            return {
                "ok": True,
                "host": self.host,
                "model": self.model,
                "available": available,
                "models": models,
            }
        except Exception as exc:
            self.logger.exception("Health check failed: %s", exc)
            return {
                "ok": False,
                "host": self.host,
                "model": self.model,
                "error": str(exc),
            }

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        *,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        num_ctx: int | None = None,
        num_predict: int | None = None,
        repeat_penalty: float | None = None,
        stream: bool | None = None,
        raw: bool = False,
        keep_alive: str = "5m",
    ) -> str:
        """Generate a response from Ollama and return plain text."""
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")

        sp = system_prompt or self.system_prompt
        options = self._build_options(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_ctx=num_ctx,
            num_predict=num_predict,
            repeat_penalty=repeat_penalty,
        )
        use_stream = self.stream if stream is None else bool(stream)

        self.logger.debug(
            "Generate request started model=%s prompt_chars=%d system_chars=%d stream=%s options=%s",
            self.model,
            len(prompt),
            len(sp),
            use_stream,
            options,
        )
        self.logger.debug("Prompt preview: %s", self._preview(prompt))

        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            start = time.perf_counter()
            try:
                if use_stream:
                    text = self._generate_stream(prompt=prompt, system_prompt=sp, options=options, raw=raw, keep_alive=keep_alive)
                else:
                    response = self.client.generate(
                        model=self.model,
                        prompt=prompt,
                        system=sp,
                        options=options,
                        stream=False,
                        raw=raw,
                        keep_alive=keep_alive,
                    )
                    text = self._extract_text(response)

                elapsed = time.perf_counter() - start
                self.logger.debug(
                    "Generate request succeeded model=%s duration=%.3fs response_chars=%d",
                    self.model,
                    elapsed,
                    len(text),
                )
                self.logger.debug("Response preview: %s", self._preview(text))
                return text

            except Exception as exc:
                last_error = exc
                elapsed = time.perf_counter() - start
                self.logger.exception(
                    "Generate request failed attempt=%d/%d model=%s duration=%.3fs error=%s",
                    attempt + 1,
                    self.retries + 1,
                    self.model,
                    elapsed,
                    exc,
                )
                if attempt < self.retries:
                    self.logger.debug("Retrying Ollama request after failure...")
                    time.sleep(0.5)

        raise RuntimeError(f"Ollama generation failed after {self.retries + 1} attempt(s): {last_error}")

    def generate_with_result(
        self,
        prompt: str,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> GenerationResult:
        """Generate text and return response metadata."""
        start = time.perf_counter()
        text = self.generate(prompt, system_prompt=system_prompt, **kwargs)
        duration = time.perf_counter() - start
        return GenerationResult(text=text, model=self.model, duration_s=duration)

    def generate_stream(
        self,
        prompt: str,
        system_prompt: str | None = None,
        *,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        num_ctx: int | None = None,
        num_predict: int | None = None,
        repeat_penalty: float | None = None,
        raw: bool = False,
        keep_alive: str = "5m",
    ) -> Generator[str, None, None]:
        """Yield generated tokens/text chunks from Ollama.

        This is useful if you later want to stream into a UI.
        """
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")

        sp = system_prompt or self.system_prompt
        options = self._build_options(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_ctx=num_ctx,
            num_predict=num_predict,
            repeat_penalty=repeat_penalty,
        )

        self.logger.debug(
            "Generate stream started model=%s prompt_chars=%d system_chars=%d options=%s",
            self.model,
            len(prompt),
            len(sp),
            options,
        )

        stream = self.client.generate(
            model=self.model,
            prompt=prompt,
            system=sp,
            options=options,
            stream=True,
            raw=raw,
            keep_alive=keep_alive,
        )

        for event in stream:
            chunk = self._extract_stream_chunk(event)
            if chunk:
                yield chunk

    def list_models(self) -> list[str]:
        """Return model names currently visible to Ollama."""
        data = self.client.list()
        models: list[str] = []
        for item in data.get("models", []):
            if isinstance(item, dict):
                name = item.get("name")
                if name:
                    models.append(name)
            else:
                name = getattr(item, "model", None) or getattr(item, "name", None)
                if name:
                    models.append(name)
        return models

    def _build_options(
        self,
        *,
        temperature: float | None,
        top_p: float | None,
        top_k: int | None,
        num_ctx: int | None,
        num_predict: int | None,
        repeat_penalty: float | None,
    ) -> Dict[str, Any]:
        return {
            "temperature": self.temperature if temperature is None else float(temperature),
            "top_p": self.top_p if top_p is None else float(top_p),
            "top_k": self.top_k if top_k is None else int(top_k),
            "num_ctx": self.num_ctx if num_ctx is None else int(num_ctx),
            "num_predict": self.num_predict if num_predict is None else int(num_predict),
            "repeat_penalty": self.repeat_penalty if repeat_penalty is None else float(repeat_penalty),
        }

    def _generate_stream(
        self,
        *,
        prompt: str,
        system_prompt: str,
        options: Dict[str, Any],
        raw: bool,
        keep_alive: str,
    ) -> str:
        pieces: list[str] = []
        stream = self.client.generate(
            model=self.model,
            prompt=prompt,
            system=system_prompt,
            options=options,
            stream=True,
            raw=raw,
            keep_alive=keep_alive,
        )
        for event in stream:
            chunk = self._extract_stream_chunk(event)
            if chunk:
                pieces.append(chunk)
        text = "".join(pieces)
        self.logger.debug("Stream generation completed response_chars=%d", len(text))
        return text

    @staticmethod
    def _extract_text(response: Any) -> str:
        """Normalize the response returned by ollama.Client.generate(stream=False)."""
        if response is None:
            return ""
        if isinstance(response, str):
            return response
        if isinstance(response, dict):
            if "response" in response:
                return str(response.get("response", ""))
            if "message" in response and isinstance(response["message"], dict):
                return str(response["message"].get("content", ""))
            return str(response)
        if hasattr(response, "response"):
            return str(getattr(response, "response", ""))
        if hasattr(response, "message"):
            msg = getattr(response, "message")
            if isinstance(msg, dict):
                return str(msg.get("content", ""))
            return str(getattr(msg, "content", ""))
        return str(response)

    @staticmethod
    def _extract_stream_chunk(event: Any) -> str:
        if event is None:
            return ""
        if isinstance(event, str):
            return event
        if isinstance(event, dict):
            if "response" in event:
                return str(event.get("response", ""))
            if "message" in event and isinstance(event["message"], dict):
                return str(event["message"].get("content", ""))
            return ""
        if hasattr(event, "response"):
            return str(getattr(event, "response", ""))
        return ""

    @staticmethod
    def _preview(text: str, limit: int = 220) -> str:
        cleaned = " ".join(text.split())
        return cleaned if len(cleaned) <= limit else cleaned[: limit - 3] + "..."


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    client = OllamaClient()
    print(client.health_check())
    print(client.generate("Say hello in one short sentence."))
