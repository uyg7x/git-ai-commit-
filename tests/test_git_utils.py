"""Tests for git_utils module."""

import pytest
from unittest.mock import patch, MagicMock

from git_ai_commit.git_utils import (
    GitStatus,
    run_git_command,
    is_git_repo,
    get_staged_diff,
    get_current_branch,
    commit_changes,
    get_git_status,
)


class TestRunGitCommand:
    """Tests for run_git_command function."""

    def test_successful_git_command(self):
        """Test successful git command execution."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="test output",
                stderr="",
            )
            returncode, stdout, stderr = run_git_command(["status"])

            mock_run.assert_called_once()
            assert returncode == 0
            assert stdout == "test output"
            assert stderr == ""

    def test_git_command_failure(self):
        """Test git command that returns non-zero exit code."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=1,
                stdout="",
                stderr="fatal: not a git repository",
            )
            returncode, stdout, stderr = run_git_command(["status"])

            assert returncode == 1
            assert stdout == ""
            assert stderr == "fatal: not a git repository"

    def test_git_command_not_found(self):
        """Test when git is not installed."""
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError()
            returncode, stdout, stderr = run_git_command(["status"])

            assert returncode == 1
            assert stderr == "Git command not found. Is git installed?"


class TestIsGitRepo:
    """Tests for is_git_repo function."""

    def test_is_git_repo_returns_true(self):
        """Test when directory is a git repo."""
        with patch("git_ai_commit.git_utils.run_git_command") as mock_run:
            mock_run.return_value = (0, "", "")
            assert is_git_repo() is True

    def test_is_git_repo_returns_false(self):
        """Test when directory is not a git repo."""
        with patch("git_ai_commit.git_utils.run_git_command") as mock_run:
            mock_run.return_value = (128, "", "fatal: not a git repository")
            assert is_git_repo() is False


class TestGetStagedDiff:
    """Tests for get_staged_diff function."""

    def test_has_staged_changes(self):
        """Test when there are staged changes."""
        diff_content = "diff --git a/test.py b/test.py\n--- a/test.py\n+++ b/test.py\n@@ -1 +1 @@\n-old\n+new"
        with patch("git_ai_commit.git_utils.run_git_command") as mock_run:
            mock_run.return_value = (0, diff_content, "")
            has_staged, diff = get_staged_diff()

            assert has_staged is True
            assert diff == diff_content

    def test_no_staged_changes(self):
        """Test when there are no staged changes."""
        with patch("git_ai_commit.git_utils.run_git_command") as mock_run:
            mock_run.return_value = (0, "", "")
            has_staged, diff = get_staged_diff()

            assert has_staged is False
            assert diff == ""


class TestGetCurrentBranch:
    """Tests for get_current_branch function."""

    def test_returns_branch_name(self):
        """Test when branch name is retrieved successfully."""
        with patch("git_ai_commit.git_utils.run_git_command") as mock_run:
            mock_run.return_value = (0, "main", "")
            assert get_current_branch() == "main"

    def test_returns_none_on_failure(self):
        """Test when branch name cannot be retrieved."""
        with patch("git_ai_commit.git_utils.run_git_command") as mock_run:
            mock_run.return_value = (128, "", "error")
            assert get_current_branch() is None


class TestCommitChanges:
    """Tests for commit_changes function."""

    def test_successful_commit(self):
        """Test successful commit."""
        with patch("git_ai_commit.git_utils.run_git_command") as mock_run:
            mock_run.return_value = (0, "[main abc123] test commit", "")
            success, msg = commit_changes("test commit")

            assert success is True
            assert "Successfully committed" in msg

    def test_failed_commit(self):
        """Test failed commit."""
        with patch("git_ai_commit.git_utils.run_git_command") as mock_run:
            mock_run.return_value = (1, "", "error: pathspec 'test' did not match")
            success, msg = commit_changes("test commit")

            assert success is False
            assert "Commit failed" in msg


class TestGetGitStatus:
    """Tests for get_git_status function."""

    def test_not_a_repo(self):
        """Test when directory is not a git repo."""
        with patch("git_ai_commit.git_utils.is_git_repo", return_value=False):
            status = get_git_status()

            assert status.is_repo is False
            assert status.has_staged_changes is False
            assert "Not a git repository" in status.error

    def test_no_staged_changes(self):
        """Test when there are no staged changes."""
        with patch("git_ai_commit.git_utils.is_git_repo", return_value=True):
            with patch("git_ai_commit.git_utils.get_staged_diff", return_value=(False, "")):
                status = get_git_status()

                assert status.is_repo is True
                assert status.has_staged_changes is False
                assert "No staged changes" in status.error

    def test_success_case(self):
        """Test successful status retrieval."""
        diff_content = "diff content"
        with patch("git_ai_commit.git_utils.is_git_repo", return_value=True):
            with patch("git_ai_commit.git_utils.get_current_branch", return_value="main"):
                with patch("git_ai_commit.git_utils.get_staged_diff", return_value=(True, diff_content)):
                    status = get_git_status()

                    assert status.is_repo is True
                    assert status.has_staged_changes is True
                    assert status.staged_diff == diff_content
                    assert status.branch == "main"
                    assert status.error is None
