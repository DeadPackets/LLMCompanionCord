"""OpenRouter LLM API client with automatic context truncation recovery.

Provides async HTTP client for OpenRouter API with automatic retry
on context length errors.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Optional

import httpx

from llmcompanioncord.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ReasoningSettings:
    """Reasoning token settings for OpenRouter API."""

    enabled: bool = False
    effort: str | None = None
    max_tokens: int | None = None
    exclude: bool = True

    def to_api_dict(self) -> dict | None:
        """Convert to OpenRouter API reasoning parameter dict.

        Returns None if reasoning is disabled.
        """
        if not self.enabled:
            return None

        reasoning: dict = {"exclude": self.exclude}

        # Use effort or max_tokens (not both per docs)
        if self.effort is not None:
            reasoning["effort"] = self.effort
        elif self.max_tokens is not None:
            reasoning["max_tokens"] = self.max_tokens

        return reasoning


class LLMError(Exception):
    """Base exception for LLM-related errors."""

    pass


class ContextLengthError(LLMError):
    """Raised when the context is too long for the model."""

    pass


class LLMClient:
    """Async client for OpenRouter API with automatic context recovery."""

    API_URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(
        self,
        api_key: str,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        reasoning: ReasoningSettings | None = None,
    ) -> None:
        """Initialize the LLM client.

        Args:
            api_key: OpenRouter API key.
            model: Model identifier (e.g., 'anthropic/claude-3.5-sonnet').
            temperature: Sampling temperature (0.0 to 2.0).
            max_tokens: Maximum tokens in the response.
            reasoning: Optional reasoning token settings.
        """
        self._api_key = api_key
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._reasoning = reasoning or ReasoningSettings()
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the async HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=60.0)
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    async def _request(self, messages: list[dict]) -> str:
        """Make a single API request to OpenRouter.

        Args:
            messages: List of message dicts with 'role' and 'content'.

        Returns:
            Generated response text.

        Raises:
            ContextLengthError: If the context is too long.
            LLMError: For other API errors.
        """
        client = await self._get_client()

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self._model,
            "messages": messages,
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
        }

        # Add reasoning config if enabled
        reasoning_dict = self._reasoning.to_api_dict()
        if reasoning_dict:
            payload["reasoning"] = reasoning_dict
            logger.debug(f"Reasoning enabled: {reasoning_dict}")

        logger.debug(f"Sending request to OpenRouter with {len(messages)} messages")
        logger.debug(
            f"Model: {self._model}, temperature: {self._temperature}, max_tokens: {self._max_tokens}"
        )

        response = await client.post(self.API_URL, json=payload, headers=headers)

        if response.status_code == 200:
            data = response.json()
            content = data["choices"][0]["message"]["content"]

            # Log usage info if available
            usage = data.get("usage", {})
            if usage:
                logger.debug(
                    f"Token usage - prompt: {usage.get('prompt_tokens', 'N/A')}, "
                    f"completion: {usage.get('completion_tokens', 'N/A')}, "
                    f"total: {usage.get('total_tokens', 'N/A')}"
                )

            logger.debug(f"Received response: {len(content)} chars")
            return content

        # Handle errors
        error_text = response.text.lower()

        # Detect context length errors
        if response.status_code == 400 and any(
            keyword in error_text
            for keyword in ["context", "token", "length", "too long", "maximum"]
        ):
            logger.warning(f"Context length error: {response.text}")
            raise ContextLengthError(response.text)

        logger.error(f"OpenRouter API error {response.status_code}: {response.text}")
        raise LLMError(f"API error {response.status_code}: {response.text}")

    async def chat(
        self,
        messages: list[dict],
        truncate_callback: Optional[Callable[[], list[dict]]] = None,
    ) -> str:
        """Send messages to OpenRouter with automatic context recovery.

        On context length errors, calls truncate_callback to get a reduced
        message list and retries. Continues until success or no more messages
        can be truncated.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            truncate_callback: Optional callback that truncates the message buffer
                               and returns a new message list.

        Returns:
            Generated response text.

        Raises:
            LLMError: If the request fails after all recovery attempts.
        """
        attempts = 0
        max_attempts = 10  # Prevent infinite loops

        current_messages = messages
        logger.debug(
            f"Starting chat with {len(messages)} messages (including system prompt)"
        )

        while attempts < max_attempts:
            attempts += 1

            try:
                return await self._request(current_messages)
            except ContextLengthError:
                if truncate_callback is None:
                    raise

                # Only system prompt left - can't truncate more
                if len(current_messages) <= 1:
                    raise LLMError("Context too long even with only system prompt")

                current_messages = truncate_callback()
                logger.info(
                    f"Truncated context, retrying with {len(current_messages)} messages "
                    f"(attempt {attempts}/{max_attempts})"
                )

        raise LLMError(f"Failed after {max_attempts} truncation attempts")

    async def pick_emoji(
        self,
        message_content: str,
        author: str,
        available_emojis: list[str],
        context_messages: list[dict] | None = None,
        max_tokens: int = 32,
    ) -> str | None:
        """Use the LLM to pick an appropriate emoji reaction.

        Args:
            message_content: The message to react to.
            author: The author of the message.
            available_emojis: List of available emojis (standard + custom).
            context_messages: Optional recent message context.
            max_tokens: Max tokens for response (default 32).

        Returns:
            The selected emoji string, or None if selection fails.
        """
        # Build a concise prompt for emoji selection
        emoji_list = ", ".join(available_emojis[:100])  # Limit to avoid huge prompts

        system_prompt = (
            "You are an emoji picker. Given a message and available emojis, "
            "pick ONE emoji that would be a good reaction. "
            "Respond with ONLY the emoji, nothing else. "
            "Pick something contextually appropriate and natural."
        )

        user_content = f'Message from {author}: "{message_content}"\n\nAvailable emojis: {emoji_list}'

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        client = await self._get_client()

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self._model,
            "messages": messages,
            "temperature": self._temperature,
            "max_tokens": max_tokens,
        }

        logger.debug(f"Picking emoji for message from {author}")

        try:
            response = await client.post(self.API_URL, json=payload, headers=headers)

            if response.status_code == 200:
                data = response.json()
                emoji = data["choices"][0]["message"]["content"].strip()
                logger.debug(f"LLM picked emoji: {emoji}")
                return emoji

            logger.warning(f"Emoji selection API error: {response.status_code}")
            return None

        except Exception as e:
            logger.warning(f"Failed to pick emoji: {e}")
            return None
