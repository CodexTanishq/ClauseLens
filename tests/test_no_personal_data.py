"""Privacy and hygiene test ensuring no personal data, absolute user paths, or tokens are checked into clauselens."""

from __future__ import annotations

import re
from pathlib import Path

CLAUSELENS_ROOT = Path(__file__).resolve().parent.parent

# Regex patterns assembled without triggering literal matches in test source
USER_PATH_REGEX = re.compile(r"[A-Za-z]:[\\/]" + r"(?:Users|home)" + r"[\\/]", re.IGNORECASE)
LINUX_HOME_REGEX = re.compile(r"^/[h]ome/[A-Za-z0-9_-]+", re.MULTILINE)
EMAIL_REGEX = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
HF_TOKEN_REGEX = re.compile(r"\b" + r"hf_" + r"[A-Za-z0-9]{20,}\b")

SKIP_DIRS = {
    ".git",
    ".cuad_env",
    "venv",
    ".venv",
    "outputs",
    "models",
    "__pycache__",
    ".pytest_cache",
}

TEXT_SUFFIXES = {
    ".py",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".md",
    ".txt",
    ".css",
    ".js",
    ".html",
    ".csv",
}


def test_no_personal_data_or_tokens():
    """Verify absence of personal credentials, real emails, and absolute user paths."""
    forbidden_words = set()
    forbidden_file = CLAUSELENS_ROOT / ".forbidden_words.txt"
    if forbidden_file.exists():
        for line in forbidden_file.read_text(encoding="utf-8").splitlines():
            w = line.strip()
            if w and not w.startswith("#"):
                forbidden_words.add(w)

    violations = []

    for file_path in CLAUSELENS_ROOT.rglob("*"):
        if not file_path.is_file():
            continue

        # Skip ignored directory trees
        if any(part in SKIP_DIRS for part in file_path.parts):
            continue

        if file_path.suffix.lower() not in TEXT_SUFFIXES and file_path.name != ".gitignore":
            continue

        # Do not check the forbidden words file against itself
        if file_path.name.startswith(".forbidden_words"):
            continue

        try:
            content = file_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue

        # 1. Absolute user paths
        if USER_PATH_REGEX.search(content) or LINUX_HOME_REGEX.search(content):
            # Exempt this test file from regex pattern self-match
            if file_path.name != "test_no_personal_data.py":
                violations.append(f"{file_path.relative_to(CLAUSELENS_ROOT)}: contains absolute user directory path")

        # 2. Email addresses
        emails = EMAIL_REGEX.findall(content)
        # Filter out common example domains if any
        real_emails = [e for e in emails if not e.endswith(("@example.com", "@domain.com", "@email.com"))]
        if real_emails and file_path.name != "test_no_personal_data.py":
            violations.append(f"{file_path.relative_to(CLAUSELENS_ROOT)}: contains email address ({real_emails[:2]})")

        # 3. Hugging Face tokens
        hf_matches = HF_TOKEN_REGEX.findall(content)
        if hf_matches and file_path.name != "test_no_personal_data.py":
            violations.append(f"{file_path.relative_to(CLAUSELENS_ROOT)}: contains Hugging Face auth token")

        # 4. Forbidden words
        for fw in forbidden_words:
            if fw in content:
                violations.append(f"{file_path.relative_to(CLAUSELENS_ROOT)}: contains forbidden word '{fw}'")

    assert not violations, "Personal data / secret violations found:\n" + "\n".join(violations)

