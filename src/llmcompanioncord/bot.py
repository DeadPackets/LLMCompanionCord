"""Discord bot with LLM integration.

Main bot class that handles Discord events, manages message buffers,
and generates LLM responses.
"""

import random
from typing import Any, Optional

import discord
from discord import Message, app_commands

from llmcompanioncord.config_schema import Config
from llmcompanioncord.llm_client import LLMClient, LLMError, ReasoningSettings
from llmcompanioncord.logger import get_logger
from llmcompanioncord.message_buffer import MessageBuffer

logger = get_logger(__name__)

# Supported image MIME types for multimodal LLM requests
SUPPORTED_IMAGE_TYPES = {"image/png", "image/jpeg", "image/webp", "image/gif"}

# Common standard Unicode emojis for reactions
STANDARD_EMOJIS = [
    "😀",
    "😂",
    "🤣",
    "😊",
    "😍",
    "🥰",
    "😎",
    "🤔",
    "😮",
    "😢",
    "😭",
    "😤",
    "🤯",
    "🥳",
    "😴",
    "🤢",
    "🤮",
    "💀",
    "👻",
    "👽",
    "👍",
    "👎",
    "👏",
    "🙌",
    "🤝",
    "✌️",
    "🤞",
    "🤙",
    "💪",
    "🙏",
    "❤️",
    "🧡",
    "💛",
    "💚",
    "💙",
    "💜",
    "🖤",
    "🤍",
    "💔",
    "💯",
    "🔥",
    "⭐",
    "✨",
    "💫",
    "🎉",
    "🎊",
    "🏆",
    "🥇",
    "🎯",
    "💡",
    "✅",
    "❌",
    "⚠️",
    "❓",
    "❗",
    "💤",
    "💢",
    "💥",
    "💦",
    "🚀",
    "👀",
    "👁️",
    "🧠",
    "🗿",
    "💩",
    "🤡",
    "👑",
    "💎",
    "🪙",
    "📈",
    "📉",
    "🎵",
    "🎶",
    "🔔",
    "📢",
    "💬",
    "💭",
    "🗨️",
    "👋",
    "🫡",
]


class LLMCompanionBot(discord.Client):
    """Discord bot that uses LLM for conversational responses."""

    def __init__(self, config: Config) -> None:
        """Initialize the bot.

        Args:
            config: Validated configuration object.
        """
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(intents=intents)

        self.config = config

        # Build reasoning settings from config
        reasoning_cfg = config.llm.reasoning
        reasoning_settings = ReasoningSettings(
            enabled=reasoning_cfg.enabled,
            effort=reasoning_cfg.effort,
            max_tokens=reasoning_cfg.max_tokens,
            exclude=reasoning_cfg.exclude,
        )

        self.llm_client = LLMClient(
            api_key=config.llm.api_key,
            model=config.llm.model,
            temperature=config.llm.temperature,
            max_tokens=config.llm.max_tokens,
            reasoning=reasoning_settings,
        )
        self.message_buffer = MessageBuffer(config.behavior.message_window_size)

        # Setup command tree for slash commands
        self.tree = app_commands.CommandTree(self)
        self._setup_commands()

    def _setup_commands(self) -> None:
        """Setup slash commands for the bot."""

        @self.tree.command(
            name="purge",
            description="Delete the bot's messages in this channel and clear the buffer",
        )
        @app_commands.describe(
            count="Number of bot messages to delete (leave empty to delete all)"
        )
        async def purge_command(
            interaction: discord.Interaction,
            count: Optional[int] = None,
        ) -> None:
            """Purge bot messages from the channel.

            Args:
                interaction: The Discord interaction.
                count: Optional number of messages to delete. If None, deletes all.
            """
            await self._handle_purge(interaction, count)

    async def _handle_purge(
        self,
        interaction: discord.Interaction,
        count: Optional[int] = None,
    ) -> None:
        """Handle the /purge command.

        Args:
            interaction: The Discord interaction.
            count: Optional number of messages to delete. If None, deletes all.
        """
        channel = interaction.channel
        channel_name = getattr(channel, "name", "DM")

        if not channel:
            await interaction.response.send_message(
                "This command can only be used in a channel.",
                ephemeral=True,
            )
            return

        # Defer the response since this might take a while
        await interaction.response.defer(ephemeral=True)

        logger.info(
            f"Purge command invoked in #{channel_name} by {interaction.user.display_name} "
            f"(count={count})"
        )

        try:
            deleted_count = 0
            # Search through channel history for bot's messages
            async for message in channel.history(limit=500):
                if message.author == self.user:
                    if count is not None and deleted_count >= count:
                        break
                    try:
                        await message.delete()
                        deleted_count += 1
                    except discord.NotFound:
                        # Message already deleted
                        pass
                    except discord.Forbidden:
                        logger.warning(
                            f"No permission to delete message {message.id} in #{channel_name}"
                        )

            # Clear the internal buffer for this channel
            self.message_buffer.clear_channel(channel.id)

            logger.info(
                f"Purged {deleted_count} messages from #{channel_name} and cleared buffer"
            )

            await interaction.followup.send(
                f"Deleted {deleted_count} message(s) and cleared the conversation buffer.",
                ephemeral=True,
            )

        except discord.Forbidden:
            logger.error(f"No permission to delete messages in #{channel_name}")
            await interaction.followup.send(
                "I don't have permission to delete messages in this channel.",
                ephemeral=True,
            )
        except Exception as e:
            logger.exception(f"Error during purge in #{channel_name}: {e}")
            await interaction.followup.send(
                f"An error occurred while purging messages: {e}",
                ephemeral=True,
            )

    async def on_ready(self) -> None:
        """Called when the bot has connected to Discord."""
        if self.user:
            logger.info(f"Bot logged in as {self.user} (ID: {self.user.id})")
        logger.info(f"Connected to {len(self.guilds)} guild(s)")

        # Sync slash commands with Discord
        try:
            synced = await self.tree.sync()
            logger.info(f"Synced {len(synced)} slash command(s)")
        except Exception as e:
            logger.error(f"Failed to sync slash commands: {e}")

    async def on_message(self, message: Message) -> None:
        """Handle incoming messages.

        Args:
            message: The Discord message object.
        """
        # Ignore own messages
        if message.author == self.user:
            return

        # Ignore bots if configured
        if self.config.behavior.ignore_bots and message.author.bot:
            return

        # Check channel permissions
        if not self._is_channel_allowed(message.channel.id):
            return

        # Backfill channel history if this is the first message in this channel
        if not self.message_buffer.has_buffer(message.channel.id):
            await self._backfill_channel_history(message.channel)

        # Build message metadata
        attachment_info, image_urls = self._get_attachment_details(message)
        reply_to = await self._get_reply_context(message)

        # Add message to buffer
        self.message_buffer.add_message(
            channel_id=message.channel.id,
            author=message.author.display_name,
            content=message.content,
            is_bot_author=False,
            attachment_info=attachment_info,
            reply_to=reply_to,
            image_urls=image_urls,
        )

        channel_name = getattr(message.channel, "name", "DM")
        buffer_count = self.message_buffer.get_message_count(message.channel.id)
        logger.debug(
            f"Message from {message.author.display_name} in "
            f"#{channel_name}: {message.content[:50]}..."
        )
        logger.debug(
            f"Channel #{channel_name} buffer now has {buffer_count} messages "
            f"(max: {self.config.behavior.message_window_size})"
        )

        # Check if we should react to this message (independent of reply)
        await self._maybe_react(message)

        # Determine if we should reply
        if not self._should_reply(message):
            logger.debug(f"Not replying to message in #{channel_name}")
            return

        logger.info(f"Generating response for message in #{channel_name}")

        # Generate and send response
        try:
            response = await self._generate_response(message)

            if response:
                await message.reply(response)

                # Add our response to buffer
                if self.user:
                    self.message_buffer.add_message(
                        channel_id=message.channel.id,
                        author=self.user.display_name,
                        content=response,
                        is_bot_author=True,
                        image_urls=[],
                    )
                logger.debug(
                    f"Sent response ({len(response)} chars): {response[:50]}..."
                )

        except LLMError as e:
            logger.error(f"Failed to generate response: {e}")
        except Exception as e:
            logger.exception(f"Unexpected error generating response: {e}")

    async def close(self) -> None:
        """Clean up resources when the bot shuts down."""
        await self.llm_client.close()
        await super().close()

    def _is_channel_allowed(self, channel_id: int) -> bool:
        """Check if the bot should operate in this channel.

        Args:
            channel_id: Discord channel ID.

        Returns:
            True if the channel is allowed.
        """
        whitelist = self.config.channels.whitelist
        blacklist = self.config.channels.blacklist

        # Blacklist takes precedence
        if channel_id in blacklist:
            return False

        # If whitelist is set, channel must be in it
        if whitelist and channel_id not in whitelist:
            return False

        return True

    def _should_reply(self, message: Message) -> bool:
        """Determine if the bot should reply to this message.

        Args:
            message: The Discord message object.

        Returns:
            True if the bot should reply.
        """
        # Always reply if mentioned (and configured to do so)
        if self.config.behavior.always_reply_on_mention:
            if self.user and self.user.mentioned_in(message):
                logger.debug("Replying due to mention")
                return True

        # Random probability check
        roll = random.random()
        should_reply = roll < self.config.behavior.reply_probability

        if should_reply:
            logger.debug(f"Replying due to probability (rolled {roll:.3f})")

        return should_reply

    def _get_attachment_details(
        self, message: Message
    ) -> tuple[Optional[str], list[str]]:
        """Get attachment information and image URLs for multimodal support.

        Args:
            message: The Discord message object.

        Returns:
            Tuple of (attachment_info_text, image_urls).
            - attachment_info_text: Human-readable like "[2 image(s), 1 file(s) attached]"
            - image_urls: List of Discord CDN URLs for valid images (filtered by type/size)
        """
        if not message.attachments:
            return None, []

        max_size_bytes = self.config.behavior.max_image_size_mb * 1024 * 1024

        image_urls: list[str] = []
        type_counts: dict[str, int] = {}

        for attachment in message.attachments:
            content_type = attachment.content_type or ""

            # Check if it's a supported image type
            if content_type in SUPPORTED_IMAGE_TYPES:
                type_counts["image"] = type_counts.get("image", 0) + 1
                # Only include if within size limit
                if attachment.size <= max_size_bytes:
                    image_urls.append(attachment.url)
                else:
                    logger.debug(
                        f"Skipping image {attachment.filename} "
                        f"({attachment.size / 1024 / 1024:.1f}MB > "
                        f"{self.config.behavior.max_image_size_mb}MB limit)"
                    )
            elif content_type.startswith("image/"):
                # Unsupported image type (e.g., TIFF, BMP)
                type_counts["image"] = type_counts.get("image", 0) + 1
            elif content_type.startswith("video/"):
                type_counts["video"] = type_counts.get("video", 0) + 1
            elif content_type.startswith("audio/"):
                type_counts["audio"] = type_counts.get("audio", 0) + 1
            else:
                type_counts["file"] = type_counts.get("file", 0) + 1

        # Build attachment info string
        if type_counts:
            parts = []
            for type_name in sorted(type_counts.keys()):
                count = type_counts[type_name]
                parts.append(f"{count} {type_name}(s)")
            attachment_info = f"[{', '.join(parts)} attached]"
        else:
            attachment_info = None

        return attachment_info, image_urls

    async def _get_reply_context(self, message: Message) -> Optional[str]:
        """Get the display name of the user being replied to.

        Args:
            message: The Discord message object.

        Returns:
            Display name of the replied-to user, or None.
        """
        if not message.reference or not message.reference.message_id:
            return None

        try:
            ref_message = await message.channel.fetch_message(
                message.reference.message_id
            )
            return ref_message.author.display_name
        except Exception:
            return None

    async def _backfill_channel_history(self, channel: Any) -> None:
        """Backfill the message buffer with recent channel history.

        Called when the bot first encounters a channel to load existing
        conversation context.

        Args:
            channel: The Discord channel to backfill from.
        """
        channel_name = getattr(channel, "name", "DM")
        channel_id = channel.id
        limit = self.config.behavior.message_window_size

        logger.info(
            f"Backfilling channel #{channel_name} with last {limit} messages..."
        )

        try:
            # Fetch messages in reverse chronological order, then reverse to get chronological
            messages: list[Message] = []
            async for msg in channel.history(limit=limit):
                messages.append(msg)

            # Reverse to chronological order (oldest first)
            messages.reverse()

            backfilled_count = 0
            skipped_count = 0

            for msg in messages:
                # Skip bot messages if configured to ignore bots (except our own)
                if msg.author.bot and msg.author != self.user:
                    if self.config.behavior.ignore_bots:
                        skipped_count += 1
                        continue

                # Determine if this is our bot's message
                is_bot_author = msg.author == self.user

                # Get attachment info and image URLs
                attachment_info, image_urls = self._get_attachment_details(msg)

                # Get reply context (synchronously check reference, skip fetch for backfill)
                reply_to = None
                if msg.reference and msg.reference.resolved:
                    if isinstance(msg.reference.resolved, Message):
                        reply_to = msg.reference.resolved.author.display_name

                # Add to buffer
                self.message_buffer.add_message(
                    channel_id=channel_id,
                    author=msg.author.display_name,
                    content=msg.content,
                    is_bot_author=is_bot_author,
                    attachment_info=attachment_info,
                    reply_to=reply_to,
                    image_urls=image_urls,
                )
                backfilled_count += 1

            logger.info(
                f"Backfilled {backfilled_count} messages for channel #{channel_name} "
                f"(skipped {skipped_count} bot messages)"
            )
            logger.debug(
                f"Channel #{channel_name} buffer now has "
                f"{self.message_buffer.get_message_count(channel_id)} messages"
            )

        except discord.Forbidden:
            logger.warning(
                f"No permission to read history in channel #{channel_name}, "
                "starting with empty buffer"
            )
            # Create empty buffer so we don't retry
            self.message_buffer._get_buffer(channel_id)
        except Exception as e:
            logger.error(
                f"Failed to backfill channel #{channel_name}: {e}, "
                "starting with empty buffer"
            )
            # Create empty buffer so we don't retry
            self.message_buffer._get_buffer(channel_id)

    async def _generate_response(self, message: Message) -> Optional[str]:
        """Generate an LLM response for the current channel context.

        Args:
            message: The triggering Discord message.

        Returns:
            Generated response text, or None on failure.
        """
        channel_id = message.channel.id
        bot_name = self.user.display_name if self.user else "Assistant"

        # Get emoji history for penalty (if enabled)
        avoid_emojis: list[tuple[str, int]] | None = None
        if self.config.behavior.emoji_penalty_enabled:
            avoid_emojis = self.message_buffer.get_recent_bot_emojis(
                channel_id=channel_id,
                message_count=self.config.behavior.emoji_history_size,
            )
            if avoid_emojis:
                logger.debug(
                    f"Emoji penalty: avoiding {len(avoid_emojis)} recently used emojis"
                )

        # Get messages formatted for LLM (with images and emoji penalty)
        messages = self.message_buffer.get_messages_for_llm(
            channel_id=channel_id,
            system_prompt=self.config.llm.system_prompt,
            bot_name=bot_name,
            max_images=self.config.behavior.max_images,
            avoid_emojis=avoid_emojis,
        )

        context_message_count = len(messages) - 1  # Subtract system prompt
        logger.debug(
            f"Prepared {context_message_count} messages for LLM context "
            f"(channel {channel_id})"
        )

        # Define truncation callback for context length recovery
        def truncate_callback() -> list[dict]:
            removed = self.message_buffer.truncate_oldest(channel_id, count=5)
            logger.debug(f"Truncated {removed} oldest messages from buffer")
            new_messages = self.message_buffer.get_messages_for_llm(
                channel_id=channel_id,
                system_prompt=self.config.llm.system_prompt,
                bot_name=bot_name,
                max_images=self.config.behavior.max_images,
                avoid_emojis=avoid_emojis,
            )
            logger.debug(
                f"After truncation: {len(new_messages) - 1} messages in context"
            )
            return new_messages

        # Show typing indicator if configured
        if self.config.behavior.typing_indicator:
            async with message.channel.typing():
                return await self.llm_client.chat(
                    messages=messages,
                    truncate_callback=truncate_callback,
                )
        else:
            return await self.llm_client.chat(
                messages=messages,
                truncate_callback=truncate_callback,
            )

    async def _maybe_react(self, message: Message) -> None:
        """Potentially react to a message with an emoji.

        Uses configurable probability and LLM to select appropriate emoji.

        Args:
            message: The Discord message to potentially react to.
        """
        # Check if reactions are enabled
        reaction_prob = self.config.behavior.reaction_probability
        if reaction_prob <= 0:
            return

        # Roll for reaction
        roll = random.random()
        if roll >= reaction_prob:
            return

        channel_name = getattr(message.channel, "name", "DM")
        logger.debug(
            f"Reaction triggered for message in #{channel_name} (rolled {roll:.3f})"
        )

        try:
            # Get available emojis (standard + server custom)
            available_emojis = self._get_available_emojis(message)

            # Use LLM to pick an emoji
            emoji = await self.llm_client.pick_emoji(
                message_content=message.content,
                author=message.author.display_name,
                available_emojis=available_emojis,
                max_tokens=self.config.behavior.reaction_max_tokens,
            )

            if not emoji:
                logger.debug("LLM returned no emoji")
                return

            # Clean up the emoji (remove any extra text the LLM might have added)
            emoji = emoji.strip().split()[0] if emoji.strip() else None
            if not emoji:
                return

            # Try to add the reaction
            await message.add_reaction(emoji)
            logger.info(f"Reacted with {emoji} to message in #{channel_name}")

        except discord.HTTPException as e:
            # Invalid emoji or other Discord error
            logger.warning(f"Failed to add reaction: {e}")
        except Exception as e:
            logger.warning(f"Error during reaction: {e}")

    def _get_available_emojis(self, message: Message) -> list[str]:
        """Get list of available emojis for reactions.

        Includes standard Unicode emojis and server custom emojis.

        Args:
            message: The message (used to get guild context).

        Returns:
            List of emoji strings.
        """
        emojis = list(STANDARD_EMOJIS)

        # Add server custom emojis if in a guild
        if message.guild:
            for emoji in message.guild.emojis:
                # Format custom emojis as <:name:id> or <a:name:id> for animated
                if emoji.animated:
                    emojis.append(f"<a:{emoji.name}:{emoji.id}>")
                else:
                    emojis.append(f"<:{emoji.name}:{emoji.id}>")

            logger.debug(
                f"Available emojis: {len(STANDARD_EMOJIS)} standard + "
                f"{len(message.guild.emojis)} custom"
            )

        return emojis
