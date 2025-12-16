"""Per-channel rolling message buffer for LLM context.

Maintains a sliding window of recent messages per Discord channel,
with support for formatting messages for LLM API calls.
"""

from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from llmcompanioncord.logger import get_logger

logger = get_logger(__name__)


@dataclass
class BufferedMessage:
    """A single message stored in the buffer."""

    author: str
    content: str
    is_bot_author: bool  # True if this message is from our bot
    timestamp: datetime
    attachment_info: Optional[str] = None  # e.g., "[2 images attached]"
    reply_to: Optional[str] = None  # e.g., "Username" (who they're replying to)


class MessageBuffer:
    """Per-channel rolling window of messages.

    Maintains separate message histories for each Discord channel,
    automatically limiting to the most recent N messages.
    """

    def __init__(self, max_size: int) -> None:
        """Initialize the message buffer.

        Args:
            max_size: Maximum number of messages to keep per channel.
        """
        self._buffers: dict[int, deque[BufferedMessage]] = {}
        self._max_size = max_size

    def _get_buffer(self, channel_id: int) -> deque[BufferedMessage]:
        """Get or create the buffer for a channel."""
        if channel_id not in self._buffers:
            self._buffers[channel_id] = deque(maxlen=self._max_size)
        return self._buffers[channel_id]

    def add_message(
        self,
        channel_id: int,
        author: str,
        content: str,
        is_bot_author: bool = False,
        attachment_info: Optional[str] = None,
        reply_to: Optional[str] = None,
    ) -> None:
        """Add a message to the channel's buffer.

        Args:
            channel_id: Discord channel ID.
            author: Display name of the message author.
            content: Message content text.
            is_bot_author: True if this message is from our bot.
            attachment_info: Human-readable attachment description.
            reply_to: Display name of the user being replied to.
        """
        buffer = self._get_buffer(channel_id)
        was_full = len(buffer) == self._max_size

        buffer.append(
            BufferedMessage(
                author=author,
                content=content,
                is_bot_author=is_bot_author,
                timestamp=datetime.now(),
                attachment_info=attachment_info,
                reply_to=reply_to,
            )
        )

        if was_full:
            logger.debug(
                f"Buffer for channel {channel_id} at max capacity ({self._max_size}), "
                "oldest message was dropped"
            )
        logger.debug(
            f"Added message from {author} to channel {channel_id} buffer "
            f"(now {len(buffer)}/{self._max_size} messages)"
        )

    def get_messages_for_llm(
        self,
        channel_id: int,
        system_prompt: str,
        bot_name: str,
    ) -> list[dict[str, str]]:
        """Format messages for the LLM API.

        Args:
            channel_id: Discord channel ID.
            system_prompt: The system prompt for the LLM.
            bot_name: The bot's display name in the server.

        Returns:
            List of message dicts with 'role' and 'content' keys,
            formatted for OpenRouter/OpenAI API.
        """
        buffer = self._get_buffer(channel_id)

        # Build system prompt with bot identity
        full_system_prompt = (
            f"{system_prompt}\n\nYour name in this server is {bot_name}."
        )

        messages: list[dict[str, str]] = [
            {"role": "system", "content": full_system_prompt}
        ]

        for msg in buffer:
            # Build content with reply context and attachments
            content_parts: list[str] = []

            if msg.reply_to:
                content_parts.append(f"(replying to {msg.reply_to})")

            # Only add author prefix for user messages, not bot messages
            # The 'assistant' role already indicates it's the bot speaking
            if msg.is_bot_author:
                content_parts.append(msg.content)
            else:
                content_parts.append(f"[{msg.author}]: {msg.content}")

            if msg.attachment_info:
                content_parts.append(msg.attachment_info)

            content = " ".join(content_parts)

            # Bot messages are 'assistant', all others are 'user'
            role = "assistant" if msg.is_bot_author else "user"
            messages.append({"role": role, "content": content})

        return messages

    def truncate_oldest(self, channel_id: int, count: int = 5) -> int:
        """Remove the oldest N messages from a channel's buffer.

        Used for recovery when context is too long for the LLM.

        Args:
            channel_id: Discord channel ID.
            count: Number of messages to remove.

        Returns:
            Number of messages actually removed.
        """
        buffer = self._get_buffer(channel_id)
        original_count = len(buffer)
        removed = 0
        for _ in range(min(count, len(buffer))):
            buffer.popleft()
            removed += 1
        logger.debug(
            f"Truncated {removed} messages from channel {channel_id} buffer "
            f"({original_count} -> {len(buffer)} messages)"
        )
        return removed

    def get_message_count(self, channel_id: int) -> int:
        """Get the current message count for a channel.

        Args:
            channel_id: Discord channel ID.

        Returns:
            Number of messages in the channel's buffer.
        """
        return len(self._get_buffer(channel_id))

    def has_buffer(self, channel_id: int) -> bool:
        """Check if a buffer exists for a channel.

        Args:
            channel_id: Discord channel ID.

        Returns:
            True if the channel has an existing buffer.
        """
        return channel_id in self._buffers

    def clear_channel(self, channel_id: int) -> None:
        """Clear all messages for a channel.

        Args:
            channel_id: Discord channel ID.
        """
        if channel_id in self._buffers:
            self._buffers[channel_id].clear()
