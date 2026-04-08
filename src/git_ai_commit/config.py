"""Configuration module for git-ai-commit.

Handles environment variables and settings management.
"""

import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv

# Load .env file if present
load_dotenv()


@dataclass
class Config:
    """Configuration settings for git-ai-commit."""

    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-4o-mini"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3"
    ai_provider: str = "ollama"

    @classmethod
    def from_env(cls) -> "Config":
        """Create a Config instance from environment variables.

        Returns:
            Config: A new Config instance with values from environment.

        Examples:
            >>> config = Config.from_env()
            >>> config.ai_provider
            'ollama'
        """
        return cls(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            ollama_model=os.getenv("OLLAMA_MODEL", "llama3"),
            ai_provider=os.getenv("AI_PROVIDER", "ollama"),
        )

    def validate(self) -> tuple[bool, str]:
        """Validate the configuration.

        Returns:
            tuple[bool, str]: A tuple of (is_valid, error_message).
        """
        if self.ai_provider == "openai":
            if not self.openai_api_key:
                return False, "OPENAI_API_KEY is required for OpenAI provider"
            if not self.openai_api_key.startswith("sk-"):
                return False, "Invalid OpenAI API key format"
        elif self.ai_provider == "ollama":
            pass  # Ollama doesn't require an API key
        else:
            return False, f"Unknown AI provider: {self.ai_provider}"

        return True, ""

    def is_ollama_available(self) -> bool:
        """Check if Ollama is configured as the provider.

        Returns:
            bool: True if Ollama is the configured provider.
        """
        return self.ai_provider == "ollama"

    def is_openai_available(self) -> bool:
        """Check if OpenAI is configured as the provider.

        Returns:
            bool: True if OpenAI is the configured provider.
        """
        return self.ai_provider == "openai"
