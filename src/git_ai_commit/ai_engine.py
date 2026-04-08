"""AI Engine for git-ai-commit.

Handles interaction with OpenAI API and local Ollama for
generating commit message suggestions.
"""

import re
from typing import Optional

from openai import OpenAI

from .config import Config


SYSTEM_PROMPT = """You are an expert at writing conventional commit messages.

Follow these rules strictly:
1. Use the Conventional Commits format: <type>(<scope>): <description>
2. Types: feat, fix, docs, style, refactor, test, chore, perf, ci, build, revert
3. Keep the description concise (under 72 characters)
4. Use imperative mood: "add" not "added" or "adds"
5. Focus on WHAT changed, not WHY

Examples:
- feat(auth): add login with OAuth2
- fix(api): handle null response from /users endpoint
- chore(deps): update pytest to 8.0.0
- docs(readme): add installation instructions

Given the git diff, generate a single commit message that follows these rules.
Respond with ONLY the commit message, nothing else."""


class AIEngine:
    """AI engine for generating commit message suggestions."""

    def __init__(self, config: Config):
        """Initialize the AI engine.

        Args:
            config: Configuration object with API settings.
        """
        self.config = config
        self._openai_client: Optional[OpenAI] = None

    @property
    def openai_client(self) -> OpenAI:
        """Get or create the OpenAI client.

        Returns:
            OpenAI: The OpenAI client instance.
        """
        if self._openai_client is None:
            self._openai_client = OpenAI(api_key=self.config.openai_api_key)
        return self._openai_client

    def generate_commit_message(self, diff: str) -> tuple[bool, str]:
        """Generate a commit message based on the git diff.

        Args:
            diff: The git diff content.

        Returns:
            tuple[bool, str]: A tuple of (success, message_or_error).
        """
        if self.config.is_openai_available():
            return self._generate_with_openai(diff)
        elif self.config.is_ollama_available():
            return self._generate_with_ollama(diff)
        else:
            return False, "No AI provider configured"

    def _generate_with_openai(self, diff: str) -> tuple[bool, str]:
        """Generate commit message using OpenAI API.

        Args:
            diff: The git diff content.

        Returns:
            tuple[bool, str]: A tuple of (success, message_or_error).
        """
        try:
            response = self.openai_client.chat.completions.create(
                model=self.config.openai_model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Generate a commit message for this diff:\n\n{diff}"},
                ],
                max_tokens=100,
                temperature=0.3,
            )

            message = response.choices[0].message.content
            if message:
                return True, self._clean_commit_message(message)
            else:
                return False, "Empty response from OpenAI"

        except Exception as e:
            return False, f"OpenAI API error: {str(e)}"

    def _generate_with_ollama(self, diff: str) -> tuple[bool, str]:
        """Generate commit message using local Ollama.

        Args:
            diff: The git diff content.

        Returns:
            tuple[bool, str]: A tuple of (success, message_or_error).
        """
        try:
            from ollama import Client

            client = Client(host=self.config.ollama_base_url)

            response = client.chat(
                model=self.config.ollama_model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Generate a commit message for this diff:\n\n{diff}"},
                ],
                options={"num_predict": 100, "temperature": 0.3},
            )

            message = response["message"]["content"]
            if message:
                return True, self._clean_commit_message(message)
            else:
                return False, "Empty response from Ollama"

        except ImportError:
            return False, "Ollama package not installed. Run: pip install ollama"
        except Exception as e:
            return False, f"Ollama API error: {str(e)}"

    def _clean_commit_message(self, message: str) -> str:
        """Clean and validate a commit message.

        Args:
            message: The raw commit message from AI.

        Returns:
            str: The cleaned commit message.
        """
        # Remove any leading/trailing whitespace
        message = message.strip()

        # Remove code blocks if present
        message = re.sub(r"```[\w]*\n?", "", message)

        # Remove any leading/trailing quotes
        message = message.strip('"\'')

        # Remove newlines
        message = " ".join(message.split())

        return message
