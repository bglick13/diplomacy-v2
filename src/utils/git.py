"""
Git and Graphite utilities for experiment reproducibility.

This module provides functions to ensure code is committed before
launching training runs, enabling exact reproduction of experiments.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from datetime import datetime


@dataclass
class GitState:
    """Current git state for reproducibility tracking."""

    branch: str
    commit: str
    is_dirty: bool
    changed_files: list[str]


def _run_command(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    """Run a shell command and return the result."""
    return subprocess.run(cmd, capture_output=True, text=True, check=check)


def get_git_status() -> tuple[bool, list[str]]:
    """
    Check for uncommitted changes (staged, unstaged, or untracked).

    Returns:
        Tuple of (has_changes, list_of_changed_files)
    """
    # Check for any changes (staged, unstaged, untracked)
    result = _run_command(["git", "status", "--porcelain"], check=False)
    if result.returncode != 0:
        raise RuntimeError(f"git status failed: {result.stderr}")

    lines = [line.strip() for line in result.stdout.strip().split("\n") if line.strip()]
    return len(lines) > 0, lines


def get_current_branch() -> str:
    """Get the current git branch name."""
    result = _run_command(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    return result.stdout.strip()


def get_commit_sha(short: bool = False) -> str:
    """Get the current HEAD commit SHA."""
    cmd = ["git", "rev-parse", "HEAD"]
    if short:
        cmd = ["git", "rev-parse", "--short", "HEAD"]
    result = _run_command(cmd)
    return result.stdout.strip()


def get_git_state() -> GitState:
    """Get the complete current git state."""
    is_dirty, changed_files = get_git_status()
    return GitState(
        branch=get_current_branch(),
        commit=get_commit_sha(),
        is_dirty=is_dirty,
        changed_files=changed_files,
    )


def create_experiment_branch(experiment_name: str) -> str:
    """
    Create a new branch for the experiment using Graphite.

    Args:
        experiment_name: Name of the experiment (used in branch name)

    Returns:
        The created branch name

    Raises:
        RuntimeError: If graphite branch creation fails
    """
    # Generate branch name with date prefix (graphite convention)
    date_prefix = datetime.now().strftime("%m-%d")
    branch_name = f"{date_prefix}-{experiment_name}"

    # Create branch using graphite CLI (no git fallback - keep everything in stack)
    result = _run_command(["gt", "branch", "create", branch_name], check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"Failed to create branch with graphite: {result.stderr}\n"
            "Ensure graphite CLI (gt) is installed and configured."
        )

    return branch_name


def commit_changes(message: str) -> str:
    """
    Stage all changes and commit using Graphite.

    Args:
        message: Commit message

    Returns:
        The new commit SHA

    Raises:
        RuntimeError: If graphite commit fails
    """
    # Stage all changes (gt commit doesn't auto-stage untracked files)
    _run_command(["git", "add", "-A"])

    # Commit using graphite CLI (no git fallback - keep everything in stack)
    result = _run_command(["gt", "commit", "-m", message], check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"Failed to commit with graphite: {result.stderr}\n"
            "Ensure graphite CLI (gt) is installed and configured."
        )

    return get_commit_sha()


def ensure_clean_state_for_experiment(experiment_name: str) -> dict[str, str | bool]:
    """
    Ensure the working directory is clean before launching an experiment.

    If there are uncommitted changes:
    1. Create a new branch named after the experiment
    2. Commit all changes with a WIP message

    Args:
        experiment_name: Name of the experiment (e.g., "sweep-reward-structure")

    Returns:
        Dict with keys:
        - "branch": str - Current branch name
        - "commit": str - Current commit SHA
        - "was_dirty": bool - Whether changes were committed
    """
    is_dirty, changed_files = get_git_status()

    if not is_dirty:
        # Already clean - just return current state
        return {
            "branch": get_current_branch(),
            "commit": get_commit_sha(),
            "was_dirty": False,
        }

    # We have uncommitted changes - create branch and commit
    print(f"\n  Found {len(changed_files)} uncommitted change(s):")
    for f in changed_files[:5]:  # Show first 5
        print(f"    {f}")
    if len(changed_files) > 5:
        print(f"    ... and {len(changed_files) - 5} more")

    print("\n  Creating branch and committing for reproducibility...")

    # Create experiment branch
    branch_name = create_experiment_branch(experiment_name)
    print(f"    Branch: {branch_name}")

    # Commit all changes
    commit_message = (
        f"WIP: {experiment_name}\n\nAuto-committed before experiment launch for reproducibility."
    )
    commit_sha = commit_changes(commit_message)
    print(f"    Commit: {commit_sha[:8]}")

    return {
        "branch": branch_name,
        "commit": commit_sha,
        "was_dirty": True,
    }
