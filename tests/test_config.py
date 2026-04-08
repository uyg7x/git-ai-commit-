"""Tests for config module."""

import os
import pytest
from unittest.mock import patch

from git_ai_commit.config import Config


class TestConfig:
    """Tests for Config class."""

    def test_default_config(self):
        """Test default configuration values."""
        config = Config()
        assert config.openai_api_key is None
        assert config.openai_model == "gpt-4o-mini"
        assert config.ollama_base_url == "http://localhost:11434"
        assert config.ollama_model == "llama3"
        assert config.ai_provider == "ollama"

    def test_from_env_with_defaults(self):
        """Test creating config from environment with defaults."""
        env_vars = {}

        with patch.dict(os.environ, env_vars, clear=True):
            config = Config.from_env()
            assert config.ai_provider == "ollama"
            assert config.openai_model == "gpt-4o-mini"

    def test_from_env_with_custom_values(self):
        """Test creating config from environment with custom values."""
        env_vars = {
            "OPENAI_API_KEY": "sk-test123",
            "OPENAI_MODEL": "gpt-4o",
            "OLLAMA_BASE_URL": "http://custom:11434",
            "OLLAMA_MODEL": "mistral",
            "AI_PROVIDER": "openai",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            config = Config.from_env()
            assert config.openai_api_key == "sk-test123"
            assert config.openai_model == "gpt-4o"
            assert config.ollama_base_url == "http://custom:11434"
            assert config.ollama_model == "mistral"
            assert config.ai_provider == "openai"

    def test_validate_openai_without_key(self):
        """Test validation fails for OpenAI without API key."""
        config = Config(ai_provider="openai", openai_api_key=None)
        is_valid, error = config.validate()

        assert is_valid is False
        assert "OPENAI_API_KEY is required" in error

    def test_validate_openai_with_invalid_key(self):
        """Test validation fails for OpenAI with invalid key format."""
        config = Config(ai_provider="openai", openai_api_key="invalid-key")
        is_valid, error = config.validate()

        assert is_valid is False
        assert "Invalid OpenAI API key format" in error

    def test_validate_openai_with_valid_key(self):
        """Test validation succeeds for OpenAI with valid key."""
        config = Config(ai_provider="openai", openai_api_key="sk-validkey123")
        is_valid, error = config.validate()

        assert is_valid is True
        assert error == ""

    def test_validate_ollama_always_valid(self):
        """Test validation always succeeds for Ollama."""
        config = Config(ai_provider="ollama")
        is_valid, error = config.validate()

        assert is_valid is True
        assert error == ""

    def test_validate_unknown_provider(self):
        """Test validation fails for unknown provider."""
        config = Config(ai_provider="unknown")
        is_valid, error = config.validate()

        assert is_valid is False
        assert "Unknown AI provider" in error

    def test_is_ollama_available(self):
        """Test Ollama availability check."""
        config = Config(ai_provider="ollama")
        assert config.is_ollama_available() is True
        assert config.is_openai_available() is False

    def test_is_openai_available(self):
        """Test OpenAI availability check."""
        config = Config(ai_provider="openai", openai_api_key="sk-valid")
        assert config.is_openai_available() is True
        assert config.is_ollama_available() is False
