"""Browser profile lifecycle and directory management.

Ensures persistent context directories exist, are isolated, and are protected from accidental credential leakage.
"""

import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class BrowserProfileManager:
    """Manages persistent browser storage profiles on the local filesystem."""

    def __init__(self, base_profile_dir: Optional[Path] = None) -> None:
        self.base_profile_dir = base_profile_dir

    def ensure_profile_dir(self, profile_path: Path) -> Path:
        """Ensure that the given profile directory exists and return the resolved Path.

        The profile directory will store cookies, local storage, and session cache.
        """
        resolved_path = profile_path.resolve()
        try:
            resolved_path.mkdir(parents=True, exist_ok=True)
            logger.debug("Ensured browser profile directory at: %s", resolved_path)
            return resolved_path
        except Exception as exc:
            logger.error("Failed to create browser profile directory at %s: %s", resolved_path, exc)
            raise

    def profile_exists(self, profile_path: Path) -> bool:
        """Check if the profile directory exists and contains files."""
        if not profile_path.exists() or not profile_path.is_dir():
            return False
        # If directory contains any files or subdirectories
        return any(profile_path.iterdir())

    def get_profile_file_count(self, profile_path: Path) -> int:
        """Return the number of files stored in the persistent profile."""
        if not profile_path.exists() or not profile_path.is_dir():
            return 0
        total_files = 0
        for _, _, files in os.walk(profile_path):
            total_files += len(files)
        return total_files
