"""Per-channel rolling message buffer for LLM context.

Maintains a sliding window of recent messages per Discord channel,
with support for formatting messages for LLM API calls.
"""

import re
from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from llmcompanioncord.logger import get_logger

logger = get_logger(__name__)

# Regex pattern for Unicode emojis (comprehensive)
EMOJI_PATTERN = re.compile(
    "["
    "\U0001f600-\U0001f64f"  # Emoticons
    "\U0001f300-\U0001f5ff"  # Misc symbols & pictographs
    "\U0001f680-\U0001f6ff"  # Transport & map symbols
    "\U0001f1e0-\U0001f1ff"  # Flags
    "\U00002702-\U000027b0"  # Dingbats
    "\U0001f900-\U0001f9ff"  # Supplemental symbols
    "\U0001fa00-\U0001fa6f"  # Chess, etc.
    "\U0001fa70-\U0001faff"  # Symbols extended
    "\U00002600-\U000026ff"  # Misc symbols
    "\U00002300-\U000023ff"  # Misc technical
    "]+",
    flags=re.UNICODE,
)


@dataclass
class BufferedMessage:
    """A single message stored in the buffer."""

    author: str
    content: str
    is_bot_author: bool  # True if this message is from our bot
    timestamp: datetime
    attachment_info: Optional[str] = None  # e.g., "[2 images attached]"
    reply_to: Optional[str] = None  # e.g., "Username" (who they're replying to)
    image_urls: list[str] = field(default_factory=list)  # Discord CDN URLs for images


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
        image_urls: Optional[list[str]] = None,
    ) -> None:
        """Add a message to the channel's buffer.

        Args:
            channel_id: Discord channel ID.
            author: Display name of the message author.
            content: Message content text.
            is_bot_author: True if this message is from our bot.
            attachment_info: Human-readable attachment description.
            reply_to: Display name of the user being replied to.
            image_urls: List of Discord CDN URLs for images in this message.
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
                image_urls=image_urls or [],
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
        max_images: int = 0,
        avoid_emojis: Optional[list[tuple[str, int]]] = None,
    ) -> list[dict[str, Any]]:
        """Format messages for the LLM API with multimodal support.

        Args:
            channel_id: Discord channel ID.
            system_prompt: The system prompt for the LLM.
            bot_name: The bot's display name in the server.
            max_images: Maximum number of images to include (0 = text only).
            avoid_emojis: List of (emoji, frequency) tuples to discourage.

        Returns:
            List of message dicts formatted for OpenRouter/OpenAI API.
            Messages may contain multimodal content arrays when images are present.
        """
        buffer = self._get_buffer(channel_id)

        # Build system prompt with bot identity
        full_system_prompt = (
            f"{system_prompt}\n\nYour name in this server is {bot_name}."
        )

        # Add emoji penalty hint if provided
        if avoid_emojis:
            emoji_list = ", ".join(
                f"{emoji} ({count}x)" for emoji, count in avoid_emojis[:10]
            )
            full_system_prompt += (
                f"\n\nYou've been using these emojis frequently in recent messages: "
                f"{emoji_list}. "
                "Try to vary your emoji usage - pick different ones or skip emojis "
                "sometimes to keep things fresh."
            )

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": full_system_prompt}
        ]

        # First pass: collect image URLs from messages (newest first) up to max_images
        images_to_include: dict[int, list[str]] = {}  # msg_index -> URLs to include
        if max_images > 0:
            images_remaining = max_images
            buffer_list = list(buffer)
            for idx in range(len(buffer_list) - 1, -1, -1):
                msg = buffer_list[idx]
                if msg.image_urls and images_remaining > 0:
                    # Take up to images_remaining from this message
                    urls_to_take = msg.image_urls[:images_remaining]
                    images_to_include[idx] = urls_to_take
                    images_remaining -= len(urls_to_take)
                if images_remaining <= 0:
                    break

        # Second pass: build message list
        for idx, msg in enumerate(buffer):
            content_parts: list[str] = []

            if msg.reply_to:
                content_parts.append(f"(replying to {msg.reply_to})")

            # Only add author prefix for user messages, not bot messages
            # The 'assistant' role already indicates it's the bot speaking
            if msg.is_bot_author:
                content_parts.append(msg.content)
            else:
                content_parts.append(f"[{msg.author}]: {msg.content}")

            # Check if this message has images to include
            included_urls = images_to_include.get(idx, [])
            excluded_image_count = (
                len(msg.image_urls) - len(included_urls) if msg.image_urls else 0
            )

            # Add attachment info for non-included attachments
            if excluded_image_count > 0:
                content_parts.append(f"[{excluded_image_count} additional image(s)]")
            elif msg.attachment_info and not msg.image_urls:
                # Non-image attachments (video, audio, files)
                content_parts.append(msg.attachment_info)

            text_content = " ".join(content_parts)
            role = "assistant" if msg.is_bot_author else "user"

            # Build message with or without images
            if included_urls:
                # Multimodal format: content is a list
                content_array: list[dict[str, Any]] = [
                    {"type": "text", "text": text_content}
                ]
                for url in included_urls:
                    content_array.append(
                        {"type": "image_url", "image_url": {"url": url}}
                    )
                messages.append({"role": role, "content": content_array})
            else:
                # Text-only format: content is a string
                messages.append({"role": role, "content": text_content})

        return messages

    def get_recent_bot_emojis(
        self,
        channel_id: int,
        message_count: int,
    ) -> list[tuple[str, int]]:
        """Get emojis used in recent bot messages with frequency counts.

        Args:
            channel_id: Discord channel ID.
            message_count: Number of recent bot messages to scan.

        Returns:
            List of (emoji, count) tuples sorted by frequency (most used first).
        """
        buffer = self._get_buffer(channel_id)

        emoji_counter: Counter[str] = Counter()
        bot_messages_scanned = 0

        # Iterate from newest to oldest
        for msg in reversed(buffer):
            if msg.is_bot_author:
                # Extract all emojis from the message content
                emojis = EMOJI_PATTERN.findall(msg.content)
                # Split combined emoji sequences into individual emojis
                for emoji_seq in emojis:
                    for emoji in emoji_seq:
                        emoji_counter[emoji] += 1

                bot_messages_scanned += 1
                if bot_messages_scanned >= message_count:
                    break

        # Return sorted by frequency (most common first)
        return emoji_counter.most_common()

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
