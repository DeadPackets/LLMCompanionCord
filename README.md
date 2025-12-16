# LLMCompanionCord

A fully configurable LLM companion Discord bot that uses OpenRouter as the API gateway.

## Features

- **Configurable reply probability** - Set a % chance for the bot to randomly reply to messages
- **Always reply on mention** - Optionally always respond when @mentioned
- **Customizable system prompt** - Define the bot's personality and behavior
- **Rolling message window** - Maintains context from the last N messages per channel
- **Automatic history backfill** - Loads recent channel history when first encountering a channel
- **Automatic context recovery** - If a request fails due to context length, automatically truncates and retries
- **Per-channel message buffers** - Each channel has its own conversation history
- **Channel whitelist/blacklist** - Control which channels the bot operates in
- **Slash commands** - `/purge` command to delete bot messages and clear buffer
- **Centralized logging** - Console and file logging with configurable levels
- **Strict YAML validation** - Pydantic-based configuration validation
- **Environment variable support** - Secrets can be set via env vars (take precedence over config file)

## Quick Start

### Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip
- A Discord bot token ([Discord Developer Portal](https://discord.com/developers/applications))
- An OpenRouter API key ([OpenRouter](https://openrouter.ai/))

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/LLMCompanionCord.git
   cd LLMCompanionCord
   ```

2. Install dependencies:
   ```bash
   uv sync
   ```

3. Copy the example config and fill in your values:
   ```bash
   cp config.example.yaml config.yaml
   ```

4. Edit `config.yaml` with your Discord token and OpenRouter API key.

### Running the Bot

**With uv (recommended):**
```bash
uv run python -m llmcompanioncord
```

**Or directly:**
```bash
python -m llmcompanioncord
```

### Running with Docker

1. Copy and configure your config file:
   ```bash
   cp config.example.yaml config.yaml
   # Edit config.yaml with your settings
   ```

2. Build and run with Docker Compose:
   ```bash
   docker compose up -d
   ```

3. View logs:
   ```bash
   docker compose logs -f
   ```

**Using environment variables with Docker:**
```bash
DISCORD_TOKEN=your_token OPENROUTER_API_KEY=your_key docker compose up -d
```

Or create a `.env` file:
```env
DISCORD_TOKEN=your_discord_token
OPENROUTER_API_KEY=your_openrouter_key
```

## Configuration

All configuration is done via `config.yaml`. See `config.example.yaml` for a fully documented example.

### Configuration Options

| Section | Option | Type | Default | Description |
|---------|--------|------|---------|-------------|
| `discord.token` | string | - | Discord bot token |
| `llm.api_key` | string | - | OpenRouter API key |
| `llm.model` | string | `anthropic/claude-3.5-sonnet` | Model to use |
| `llm.temperature` | float | `0.7` | Sampling temperature (0.0-2.0) |
| `llm.max_tokens` | int | `1024` | Maximum response tokens |
| `llm.system_prompt` | string | (see example) | Bot personality prompt |
| `behavior.reply_probability` | float | `0.1` | Chance to reply (0.0-1.0) |
| `behavior.always_reply_on_mention` | bool | `true` | Always reply when @mentioned |
| `behavior.message_window_size` | int | `50` | Messages to keep as context |
| `behavior.typing_indicator` | bool | `true` | Show typing while generating |
| `behavior.ignore_bots` | bool | `true` | Ignore messages from bots |
| `channels.whitelist` | list[int] | `[]` | Only respond in these channels |
| `channels.blacklist` | list[int] | `[]` | Never respond in these channels |
| `logging.level` | string | `INFO` | Log level (DEBUG/INFO/WARNING/ERROR) |
| `logging.file` | string | `logs/bot.log` | Log file path |

### Environment Variables

These environment variables override config file values:

| Variable | Description |
|----------|-------------|
| `DISCORD_TOKEN` | Discord bot token |
| `OPENROUTER_API_KEY` | OpenRouter API key |

## Slash Commands

| Command | Description |
|---------|-------------|
| `/purge [count]` | Delete the bot's messages in the current channel and clear the conversation buffer. If `count` is provided, only deletes that many messages. If omitted, deletes all bot messages (up to 500). |

## Project Structure

```
LLMCompanionCord/
├── src/llmcompanioncord/
│   ├── __init__.py
│   ├── __main__.py        # Package entry point
│   ├── main.py            # Application entry point
│   ├── bot.py             # Discord bot logic
│   ├── config_schema.py   # Pydantic configuration models
│   ├── llm_client.py      # OpenRouter API client
│   ├── message_buffer.py  # Rolling message window
│   └── logger.py          # Centralized logging
├── config.example.yaml    # Example configuration
├── config.yaml            # Your configuration (gitignored)
├── Dockerfile             # Docker build file
├── docker-compose.yml     # Docker Compose configuration
├── pyproject.toml         # Project dependencies
└── README.md
```

## Discord Bot Setup

1. Go to the [Discord Developer Portal](https://discord.com/developers/applications)
2. Create a new application
3. Go to the "Bot" section and create a bot
4. Enable the "Message Content Intent" under Privileged Gateway Intents
5. Copy the bot token to your `config.yaml` or set as `DISCORD_TOKEN` env var
6. Go to OAuth2 > URL Generator, select:
   - Scopes: `bot`, `applications.commands`
   - Bot Permissions: `Send Messages`, `Read Message History`, `Manage Messages`
7. Use the generated URL to invite the bot to your server

## License

MIT License - see [LICENSE](LICENSE) for details.
