"""Git utility functions for git-ai-commit.

Handles all git-related operations including repository validation,
staged changes detection, and commit execution.
"""

import subprocess
from dataclasses import dataclass
from typing import Optional


@dataclass
class GitStatus:
    """Represents the current git repository status."""

    is_repo: bool
    has_staged_changes: bool
    staged_diff: str
    branch: Optional[str] = None
    error: Optional[str] = None


def run_git_command(args: list[str], cwd: Optional[str] = None) -> tuple[int, str, str]:
    """Execute a git command and return the result.

    Args:
        args: List of git command arguments (e.g., ["status", "--porcelain"]).
        cwd: Working directory for the command. Defaults to current directory.

    Returns:
        tuple[int, str, str]: A tuple of (return_code, stdout, stderr).
    """
    try:
        result = subprocess.run(
            ["git"] + args,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        return result.returncode, result.stdout, result.stderr
    except FileNotFoundError:
        return 1, "", "Git command not found. Is git installed?"
    except subprocess.TimeoutExpired:
        return 1, "", "Git command timed out"
    except Exception as e:
        return 1, "", str(e)


def is_git_repo(cwd: Optional[str] = None) -> bool:
    """Check if the current directory is a git repository.

    Args:
        cwd: Working directory to check. Defaults to current directory.

    Returns:
        bool: True if the directory is inside a git repository.
    """
    returncode, _, _ = run_git_command(["rev-parse", "--is-inside-work-tree"], cwd)
    return returncode == 0


def get_staged_diff(cwd: Optional[str] = None) -> tuple[bool, str]:
    """Get the diff of staged changes.

    Args:
        cwd: Working directory. Defaults to current directory.

    Returns:
        tuple[bool, str]: A tuple of (has_staged, diff_content).
    """
    returncode, stdout, stderr = run_git_command(["diff", "--cached"], cwd)

    if returncode != 0:
        return False, f"Error getting staged diff: {stderr}"

    has_staged = bool(stdout.strip())
    return has_staged, stdout


def get_current_branch(cwd: Optional[str] = None) -> Optional[str]:
    """Get the name of the current branch.

    Args:
        cwd: Working directory. Defaults to current directory.

    Returns:
        Optional[str]: The current branch name or None if not available.
    """
    returncode, stdout, stderr = run_git_command(
        ["rev-parse", "--abbrev-ref", "HEAD"], cwd
    )

    if returncode != 0:
        return None

    return stdout.strip()


def commit_changes(message: str, cwd: Optional[str] = None) -> tuple[bool, str]:
    """Execute a git commit with the given message.

    Args:
        message: The commit message.
        cwd: Working directory. Defaults to current directory.

    Returns:
        tuple[bool, str]: A tuple of (success, message).
    """
    returncode, stdout, stderr = run_git_command(
        ["commit", "-m", message], cwd
    )

    if returncode == 0:
        return True, f"Successfully committed:\n{stdout}"
    else:
        return False, f"Commit failed:\n{stderr}"


def get_git_status(cwd: Optional[str] = None) -> GitStatus:
    """Get the complete git status for the current repository.

    Args:
        cwd: Working directory. Defaults to current directory.

    Returns:
        GitStatus: Object containing the current git status.
    """
    # Check if it's a git repo
    if not is_git_repo(cwd):
        return GitStatus(
            is_repo=False,
            has_staged_changes=False,
            staged_diff="",
            error="Not a git repository. Run 'git init' first.",
        )

    # Get current branch
    branch = get_current_branch(cwd)

    # Get staged diff
    has_staged, diff = get_staged_diff(cwd)

    if not has_staged:
        return GitStatus(
            is_repo=True,
            has_staged_changes=False,
            staged_diff="",
            branch=branch,
            error="No staged changes. Stage your changes with 'git add' first.",
        )

    return GitStatus(
        is_repo=True,
        has_staged_changes=True,
        staged_diff=diff,
        branch=branch,
    )
