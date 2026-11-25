"""Chat LLM via LangChain."""

import os

import logging
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

# Bound every LLM call so a hung/slow request can't block a turn forever.
LLM_TIMEOUT_SEC = float(os.getenv("LLM_TIMEOUT_SEC", "20"))
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "0"))

DEFAULT_SYSTEM_PROMPT = """You are a friendly, helpful assistant. You can speak in Hindi, English, or Hinglish (code-mixed) naturally.
Keep responses concise and conversational—suitable for voice playback.
Respond in the same language style the user uses when possible."""


class ReliableChatOpenAI(ChatOpenAI):
    """Reject empty completions and try one alternate free route before failing.

    Tool-only responses are valid and must remain available to LangGraph.
    """

    @staticmethod
    def _usable(result):
        return any(
            _extract_text(g.message.content).strip() or g.message.tool_calls
            for g in result.generations
        )

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        last_error = None
        for route in (self.model_name, os.getenv("OPENROUTER_FALLBACK_MODEL", "nvidia/nemotron-3.5-lightning:free")):
            try:
                result = super()._generate(messages, stop=stop, run_manager=run_manager, **{**kwargs, "model": route})
                if self._usable(result):
                    return result
                raise ValueError("The language model returned an empty response")
            except Exception as exc:
                last_error = exc
                logging.getLogger("bodhi.llm").warning("Model route failed (%s); checking fallback", type(exc).__name__)
        raise RuntimeError("Interview model unavailable. Please retry the turn.") from last_error

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        last_error = None
        for route in (self.model_name, os.getenv("OPENROUTER_FALLBACK_MODEL", "nvidia/nemotron-3.5-lightning:free")):
            try:
                result = await super()._agenerate(messages, stop=stop, run_manager=run_manager, **{**kwargs, "model": route})
                if self._usable(result):
                    return result
                raise ValueError("The language model returned an empty response")
            except Exception as exc:
                last_error = exc
                logging.getLogger("bodhi.llm").warning("Model route failed (%s); checking fallback", type(exc).__name__)
        raise RuntimeError("Interview model unavailable. Please retry the turn.") from last_error

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        last_error = None
        for route in (self.model_name, os.getenv("OPENROUTER_FALLBACK_MODEL", "nvidia/nemotron-3.5-lightning:free")):
            emitted = False
            try:
                for chunk in super()._stream(messages, stop=stop, run_manager=run_manager, **{**kwargs, "model": route}):
                    if _extract_text(chunk.message.content).strip() or getattr(chunk.message, "tool_call_chunks", []):
                        emitted = True
                    yield chunk
                if emitted:
                    return
                raise ValueError("The language model returned an empty stream")
            except Exception as exc:
                if emitted:
                    # Never replay a partly spoken reply from another model.
                    raise
                last_error = exc
        raise RuntimeError("Interview model unavailable. Please retry the turn.") from last_error

    async def _astream(self, messages, stop=None, run_manager=None, **kwargs):
        last_error = None
        for route in (self.model_name, os.getenv("OPENROUTER_FALLBACK_MODEL", "nvidia/nemotron-3.5-lightning:free")):
            emitted = False
            try:
                async for chunk in super()._astream(messages, stop=stop, run_manager=run_manager, **{**kwargs, "model": route}):
                    if _extract_text(chunk.message.content).strip() or getattr(chunk.message, "tool_call_chunks", []):
                        emitted = True
                    yield chunk
                if emitted:
                    return
                raise ValueError("The language model returned an empty stream")
            except Exception as exc:
                if emitted:
                    # Never replay a partly spoken reply from another model.
                    raise
                last_error = exc
        raise RuntimeError("Interview model unavailable. Please retry the turn.") from last_error


def create_llm(api_key: str, model: str = "gemini-3.1-flash-lite-preview"):
    """Use a bounded, tool-capable free model for the interview pipeline."""
    return ReliableChatOpenAI(
        model=os.getenv("OPENROUTER_MODEL", "nvidia/nemotron-3-ultra-550b-a55b:free"),
        api_key=os.getenv("OPENROUTER_API_KEY", ""),
        base_url="https://openrouter.ai/api/v1",
        temperature=0.7,
        timeout=LLM_TIMEOUT_SEC,
        max_retries=LLM_MAX_RETRIES,
        max_tokens=int(os.getenv("LLM_MAX_TOKENS", "1600")),
        extra_body={"reasoning": {"enabled": False}},
    )


def get_reply(
    user_message: str,
    *,
    api_key: str,
    history: list[dict] | None = None,
    system_prompt: str | None = None,
    model: str = "gemini-3.1-flash-lite-preview",
) -> str:
    """
    Get LLM reply given user message and optional conversation history.
    history: list of {"role": "user"|"assistant", "content": "..."}
    """
    llm = create_llm(api_key=api_key, model=model)
    system = system_prompt or DEFAULT_SYSTEM_PROMPT

    messages = [SystemMessage(content=system)]

    if history:
        for msg in history:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            else:
                messages.append(AIMessage(content=msg["content"]))

    messages.append(HumanMessage(content=user_message))

    response = llm.invoke(messages)
    content = response.content if hasattr(response, "content") else response
    return _extract_text(content)


def _extract_text(content) -> str:
    """Normalize Gemini 3.x (list of dicts) and 2.5 (plain string) responses to plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                parts.append(item["text"])
            elif isinstance(item, str):
                parts.append(item)
        return " ".join(parts) if parts else ""
    return str(content)
