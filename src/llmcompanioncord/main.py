"""LLM Companion Discord Bot - Entry Point.

This module serves as the main entry point for running the bot.
"""

import argparse
import sys

from llmcompanioncord.bot import LLMCompanionBot
from llmcompanioncord.config_schema import ConfigError, load_config
from llmcompanioncord.logger import get_logger, setup_logging


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="LLM Companion Discord Bot",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="config.yaml",
        help="Path to the configuration file",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point for the bot."""
    args = parse_args()

    # Load config first (before logging setup to get log config)
    try:
        config = load_config(args.config)
    except ConfigError as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        sys.exit(1)

    # Setup logging with config values
    setup_logging(
        level=config.logging.level,
        log_file=config.logging.file,
        log_to_file=config.logging.log_to_file,
    )
    logger = get_logger(__name__)

    logger.info("Starting LLM Companion Bot...")
    logger.info(f"Config file: {args.config}")
    logger.info(f"Model: {config.llm.model}")
    logger.info(f"Reply probability: {config.behavior.reply_probability}")
    logger.info(f"Reaction probability: {config.behavior.reaction_probability}")
    logger.info(f"Always reply on mention: {config.behavior.always_reply_on_mention}")
    logger.info(f"Message window size: {config.behavior.message_window_size}")
    logger.info(f"Log to file: {config.logging.log_to_file}")

    # Create and run bot
    bot = LLMCompanionBot(config)

    try:
        # Disable discord.py's default logging handler to avoid duplicates
        bot.run(config.discord.token, log_handler=None)
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.exception(f"Bot crashed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
