"""Tests for ai_engine module."""

import pytest
from unittest.mock import patch, MagicMock

from git_ai_commit.ai_engine import AIEngine
from git_ai_commit.config import Config


class TestAIEngine:
    """Tests for AIEngine class."""

    def test_init_with_config(self):
        """Test AI engine initialization with config."""
        config = Config()
        engine = AIEngine(config)
        assert engine.config == config

    def test_clean_commit_message_removes_code_blocks(self):
        """Test that code blocks are removed from commit messages."""
        config = Config()
        engine = AIEngine(config)

        raw_message = '```\nfeat: add login feature\n```'
        cleaned = engine._clean_commit_message(raw_message)

        assert cleaned == "feat: add login feature"

    def test_clean_commit_message_removes_quotes(self):
        """Test that quotes are removed from commit messages."""
        config = Config()
        engine = AIEngine(config)

        raw_message = '"feat: add login feature"'
        cleaned = engine._clean_commit_message(raw_message)

        assert cleaned == "feat: add login feature"

    def test_clean_commit_message_removes_newlines(self):
        """Test that newlines are removed from commit messages."""
        config = Config()
        engine = AIEngine(config)

        raw_message = "feat:\n  add login feature"
        cleaned = engine._clean_commit_message(raw_message)

        assert cleaned == "feat: add login feature"


class TestAIEngineOpenAI:
    """Tests for OpenAI integration."""

    def test_generate_with_openai_success(self):
        """Test successful commit message generation with OpenAI."""
        config = Config(ai_provider="openai", openai_api_key="sk-test")
        engine = AIEngine(config)

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "feat(auth): add login"

        with patch("openai.OpenAI") as mock_openai_class:
            mock_client = MagicMock()
            mock_openai_class.return_value = mock_client
            mock_client.chat.completions.create.return_value = mock_response

            # Create a new engine since client is cached
            engine = AIEngine(config)
            engine._openai_client = mock_client

            success, result = engine._generate_with_openai("dummy diff")

            assert success is True
            assert result == "feat(auth): add login"

    def test_generate_with_openai_error(self):
        """Test OpenAI error handling."""
        config = Config(ai_provider="openai", openai_api_key="sk-test")

        with patch("openai.OpenAI") as mock_openai_class:
            mock_client = MagicMock()
            mock_openai_class.return_value = mock_client
            mock_client.chat.completions.create.side_effect = Exception("API error")

            engine = AIEngine(config)
            engine._openai_client = mock_client

            success, result = engine._generate_with_openai("dummy diff")

            assert success is False
            assert "OpenAI API error" in result


class TestAIEngineOllama:
    """Tests for Ollama integration."""

    def test_generate_with_ollama_success(self):
        """Test successful commit message generation with Ollama."""
        config = Config(ai_provider="ollama")
        engine = AIEngine(config)

        mock_response = {"message": {"content": "feat(api): add endpoint"}}

        mock_client_instance = MagicMock()
        mock_client_instance.chat.return_value = mock_response

        with patch("ollama.Client", return_value=mock_client_instance):
            success, result = engine._generate_with_ollama("dummy diff")

            assert success is True
            assert result == "feat(api): add endpoint"

    def test_generate_with_ollama_import_error(self):
        """Test Ollama import error handling."""
        config = Config(ai_provider="ollama")
        engine = AIEngine(config)

        with patch.dict("sys.modules", {"ollama": None}):
            success, result = engine._generate_with_ollama("dummy diff")

            assert success is False
            assert "Ollama package not installed" in result

    def test_generate_with_ollama_api_error(self):
        """Test Ollama API error handling."""
        config = Config(ai_provider="ollama")
        engine = AIEngine(config)

        with patch("ollama.Client") as mock_client_class:
            mock_client_instance = MagicMock()
            mock_client_class.return_value = mock_client_instance
            mock_client_instance.chat.side_effect = Exception("Connection failed")

            success, result = engine._generate_with_ollama("dummy diff")

            assert success is False
            assert "Ollama API error" in result


class TestGenerateCommitMessage:
    """Tests for generate_commit_message dispatcher."""

    def test_dispatches_to_openai(self):
        """Test that OpenAI provider dispatches correctly."""
        config = Config(ai_provider="openai", openai_api_key="sk-test")
        engine = AIEngine(config)

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "feat: test"

        with patch.object(engine, "_generate_with_openai", return_value=(True, "feat: test")) as mock_generate:
            success, result = engine.generate_commit_message("dummy diff")

            mock_generate.assert_called_once_with("dummy diff")

    def test_dispatches_to_ollama(self):
        """Test that Ollama provider dispatches correctly."""
        config = Config(ai_provider="ollama")
        engine = AIEngine(config)

        with patch.object(engine, "_generate_with_ollama", return_value=(True, "feat: test")):
            success, result = engine.generate_commit_message("dummy diff")

            assert success is True

    def test_no_provider_configured(self):
        """Test error when no provider is configured."""
        config = Config(ai_provider="unknown")
        engine = AIEngine(config)

        success, result = engine.generate_commit_message("dummy diff")

        assert success is False
        assert "No AI provider configured" in result
