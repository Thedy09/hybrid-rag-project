"""Answer from retrieved chunks with an LLM, or quote the chunks directly."""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass

from src.tokenize import tokenize

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "be",
    "by",
    "can",
    "do",
    "does",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "still",
    "that",
    "the",
    "to",
    "what",
    "when",
    "which",
    "why",
    "with",
}


class GeneratorError(RuntimeError):
    pass


@dataclass
class Generation:
    answer: str
    backend: str
    warning: str | None = None


class Generator:
    """Call a configured chat model, and fall back to sentences from the context."""

    def __init__(self) -> None:
        self.provider, self.backend = _select_backend()

    def generate(self, prompt: str, question: str, results: list[dict]) -> Generation:
        if self.provider == "extractive":
            return Generation(
                answer=grounded_answer(question, results),
                backend=self.backend,
            )

        try:
            answer = _call_provider(self.provider, prompt)
        except (GeneratorError, TimeoutError, json.JSONDecodeError) as exc:
            return Generation(
                answer=grounded_answer(question, results),
                backend="grounded-extractive",
                warning=f"{self.backend} failed ({exc}). Showing a grounded extractive answer instead.",
            )

        cleaned = answer.strip()
        if not cleaned:
            return Generation(
                answer=grounded_answer(question, results),
                backend="grounded-extractive",
                warning=f"{self.backend} returned an empty answer. Showing a grounded extractive answer instead.",
            )
        return Generation(answer=cleaned, backend=self.backend)


def grounded_answer(question: str, results: list[dict]) -> str:
    """Quote the retrieved sentences that overlap the question, with citations."""

    if not results:
        return "The context does not contain the answer to this question."

    question_tokens = set(tokenize(question)) - _STOPWORDS
    if not question_tokens:
        question_tokens = set(tokenize(question))

    candidates: list[tuple[float, str, object]] = []
    for source_rank, result in enumerate(results):
        document = result["document"]
        for sentence in _sentences(document.page_content):
            score = _sentence_score(question_tokens, sentence, source_rank)
            if score <= 0:
                continue
            candidates.append((score, sentence, document))

    if not candidates:
        return "The context does not contain the answer to this question."

    candidates.sort(key=lambda item: item[0], reverse=True)
    chosen: list[str] = []
    sources: list[str] = []
    seen_sentences: set[str] = set()
    for _, sentence, document in candidates:
        if sentence in seen_sentences:
            continue
        seen_sentences.add(sentence)
        chosen.append(sentence)
        citation = (
            f"{document.metadata.get('file_name', 'unknown')}, "
            f"Page {document.metadata.get('page_number', '?')}"
        )
        if citation not in sources:
            sources.append(citation)
        if len(chosen) >= 3:
            break

    return f"{' '.join(chosen)}\n\nSources: {'; '.join(sources)}."


def _sentences(text: str) -> list[str]:
    parts = _SENTENCE_SPLIT.split(text.strip())
    return [part.strip() for part in parts if len(part.strip()) >= 20]


def _sentence_score(question_tokens: set[str], sentence: str, source_rank: int) -> float:
    sentence_tokens = set(tokenize(sentence))
    if not sentence_tokens or not question_tokens:
        return 0.0
    overlap = question_tokens & sentence_tokens
    if not overlap:
        return 0.0
    recall = len(overlap) / len(question_tokens)
    precision = len(overlap) / len(sentence_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return f1 / (1 + source_rank)


def _select_backend() -> tuple[str, str]:
    requested = os.getenv("LLM_PROVIDER", "auto").strip().lower() or "auto"
    if requested == "extractive":
        return "extractive", "grounded-extractive"
    if requested == "auto":
        for provider, backend in _available_providers():
            return provider, backend
        return "extractive", "grounded-extractive"
    if requested in {"openai", "groq", "gemini", "anthropic", "ollama"}:
        return requested, _backend_name(requested)
    return "extractive", "grounded-extractive"


def _available_providers() -> list[tuple[str, str]]:
    found = []
    for provider in ("openai", "groq", "gemini", "anthropic"):
        if _api_key(provider):
            found.append((provider, _backend_name(provider)))
    if os.getenv("OLLAMA_BASE_URL", "").strip():
        found.append(("ollama", _backend_name("ollama")))
    return found


def _backend_name(provider: str) -> str:
    model = {
        "openai": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        "groq": os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        "gemini": os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        "anthropic": os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5"),
        "ollama": os.getenv("OLLAMA_MODEL", "llama3.2"),
    }[provider]
    return f"{provider}:{model}"


def _api_key(provider: str) -> str:
    if provider == "gemini":
        return os.getenv("GEMINI_API_KEY", "").strip() or os.getenv("GOOGLE_API_KEY", "").strip()
    env_name = {
        "openai": "OPENAI_API_KEY",
        "groq": "GROQ_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
    }[provider]
    return os.getenv(env_name, "").strip()


def _call_provider(provider: str, prompt: str) -> str:
    if provider in {"openai", "groq"}:
        return _openai_compatible(provider, prompt)
    if provider == "gemini":
        return _gemini(prompt)
    if provider == "anthropic":
        return _anthropic(prompt)
    if provider == "ollama":
        return _ollama(prompt)
    raise GeneratorError(f"Unknown provider: {provider}")


def _openai_compatible(provider: str, prompt: str) -> str:
    if provider == "openai":
        base = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        key = _api_key("openai")
    else:
        base = "https://api.groq.com/openai/v1"
        model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        key = _api_key("groq")
    if not key:
        raise GeneratorError(f"{provider} API key is not set")

    payload = {
        "model": model,
        "temperature": 0,
        "messages": [{"role": "user", "content": prompt}],
    }
    body = _post_json(
        f"{base}/chat/completions",
        payload,
        {"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    return body["choices"][0]["message"]["content"]


def _gemini(prompt: str) -> str:
    key = _api_key("gemini")
    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    if not key:
        raise GeneratorError("GEMINI_API_KEY is not set")
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent?key={key}"
    )
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0},
    }
    body = _post_json(url, payload, {"Content-Type": "application/json"})
    return body["candidates"][0]["content"]["parts"][0]["text"]


def _anthropic(prompt: str) -> str:
    key = _api_key("anthropic")
    model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5")
    if not key:
        raise GeneratorError("ANTHROPIC_API_KEY is not set")
    payload = {
        "model": model,
        "max_tokens": 500,
        "temperature": 0,
        "messages": [{"role": "user", "content": prompt}],
    }
    body = _post_json(
        "https://api.anthropic.com/v1/messages",
        payload,
        {
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
    )
    return body["content"][0]["text"]


def _ollama(prompt: str) -> str:
    base = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
    model = os.getenv("OLLAMA_MODEL", "llama3.2")
    payload = {
        "model": model,
        "stream": False,
        "messages": [{"role": "user", "content": prompt}],
        "options": {"temperature": 0},
    }
    body = _post_json(
        f"{base}/api/chat",
        payload,
        {"Content-Type": "application/json"},
    )
    return body["message"]["content"]


def _post_json(url: str, payload: dict, headers: dict, timeout: int = 45) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise GeneratorError(f"HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise GeneratorError(str(exc.reason)) from exc
