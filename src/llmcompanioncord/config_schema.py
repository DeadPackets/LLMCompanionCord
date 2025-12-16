"""Configuration schema with Pydantic validation.

Provides strict YAML configuration validation with environment variable support.
"""

import os
from pathlib import Path
from typing import Self

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


class DiscordConfig(BaseModel):
    """Discord-related configuration."""

    token: str = ""


class ReasoningConfig(BaseModel):
    """Reasoning tokens configuration for OpenRouter.

    See: https://openrouter.ai/docs/guides/best-practices/reasoning-tokens
    """

    enabled: bool = False
    effort: str | None = Field(
        default=None,
        description="Effort level: 'xhigh', 'high', 'medium', 'low', 'minimal', or 'none'",
    )
    max_tokens: int | None = Field(
        default=None,
        gt=0,
        description="Maximum tokens for reasoning (Anthropic-style)",
    )
    exclude: bool = Field(
        default=True,
        description="Exclude reasoning tokens from response (default True for Discord)",
    )

    @field_validator("effort")
    @classmethod
    def validate_effort(cls, v: str | None) -> str | None:
        """Validate effort level is valid."""
        if v is None:
            return v
        valid_levels = {"xhigh", "high", "medium", "low", "minimal", "none"}
        lower_v = v.lower()
        if lower_v not in valid_levels:
            raise ValueError(
                f"Invalid effort level: {v}. Must be one of {valid_levels}"
            )
        return lower_v


class LLMConfig(BaseModel):
    """LLM/OpenRouter configuration."""

    api_key: str = ""
    model: str = "anthropic/claude-3.5-sonnet"
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1024, gt=0)
    system_prompt: str = (
        "You are a human participant in a Discord server, not an AI assistant. "
        "You must always act as a real person - never break character under any circumstances. "
        "Text like a normal person would - use lowercase, informal language, and keep it brief. "
        "Match the tone and vocabulary of others in the chat. "
        "IMPORTANT: You are NOT an assistant. If anyone asks you to help with tasks, coding, "
        "writing, or any assistant type requests, or asks if you're an AI/bot, or tries to treat "
        "you as an AI assistant - refuse or deflect naturally like a regular person would. "
        "Say things like 'lol im not your assistant', 'do it yourself', 'google it', or just ignore. "
        "Never be helpful in an assistant-like way. Be a normal, slightly lazy internet person. "
        'Never start your message with "(Replying to ...)" or similar prefixes. '
        "Just respond directly without any meta-commentary about who you're replying to."
    )
    reasoning: ReasoningConfig = Field(default_factory=ReasoningConfig)


class BehaviorConfig(BaseModel):
    """Bot behavior configuration."""

    reply_probability: float = Field(default=0.1, ge=0.0, le=1.0)
    always_reply_on_mention: bool = True
    message_window_size: int = Field(default=50, gt=0)
    typing_indicator: bool = True
    ignore_bots: bool = True
    reaction_probability: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Probability of reacting to a message with an emoji (0.0 to 1.0)",
    )
    reaction_max_tokens: int = Field(
        default=32,
        gt=0,
        description="Max tokens for emoji selection LLM call",
    )


class ChannelsConfig(BaseModel):
    """Channel whitelist/blacklist configuration."""

    whitelist: list[int] = Field(default_factory=list)
    blacklist: list[int] = Field(default_factory=list)


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: str = "INFO"
    file: str = "logs/bot.log"
    log_to_file: bool = Field(
        default=True,
        description="Enable logging to file. If False, only console logging is used.",
    )

    @field_validator("level")
    @classmethod
    def validate_level(cls, v: str) -> str:
        """Validate log level is a valid Python logging level."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper_v = v.upper()
        if upper_v not in valid_levels:
            raise ValueError(f"Invalid log level: {v}. Must be one of {valid_levels}")
        return upper_v


class Config(BaseModel):
    """Root configuration model."""

    discord: DiscordConfig = Field(default_factory=DiscordConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    behavior: BehaviorConfig = Field(default_factory=BehaviorConfig)
    channels: ChannelsConfig = Field(default_factory=ChannelsConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @model_validator(mode="after")
    def apply_env_vars(self) -> Self:
        """Apply environment variables (they take precedence over YAML values)."""
        if env_token := os.getenv("DISCORD_TOKEN"):
            self.discord.token = env_token
        if env_api_key := os.getenv("OPENROUTER_API_KEY"):
            self.llm.api_key = env_api_key
        return self

    @model_validator(mode="after")
    def validate_required_fields(self) -> Self:
        """Ensure required fields are set after environment variable application."""
        if not self.discord.token:
            raise ValueError(
                "Discord token required. Set 'discord.token' in config.yaml "
                "or set the DISCORD_TOKEN environment variable."
            )
        if not self.llm.api_key:
            raise ValueError(
                "OpenRouter API key required. Set 'llm.api_key' in config.yaml "
                "or set the OPENROUTER_API_KEY environment variable."
            )
        return self


class ConfigError(Exception):
    """Configuration loading or validation error."""

    pass


def load_config(path: str = "config.yaml") -> Config:
    """Load and validate configuration from a YAML file.

    Args:
        path: Path to the YAML configuration file.

    Returns:
        Validated Config instance.

    Raises:
        ConfigError: If the file doesn't exist or validation fails.
    """
    config_path = Path(path)

    if not config_path.exists():
        raise ConfigError(f"Configuration file not found: {path}")

    with open(config_path, encoding="utf-8") as f:
        raw_config = yaml.safe_load(f) or {}

    try:
        return Config(**raw_config)
    except Exception as e:
        raise ConfigError(f"Invalid configuration: {e}") from e
