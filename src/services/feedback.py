# src/services/feedback.py
"""
FeedbackManager: Handles user feedback persistence for hate speech predictions.
Thread-safe via append-only file writes.
"""
import csv
import json
from pathlib import Path
from typing import Dict, List, Union


class FeedbackManager:
    """
    Manages user feedback storage for model predictions.

    Feedback is persisted to a CSV file with append-only writes for
    best-effort thread safety. Each feedback entry includes the original
    text, predicted spans (JSON serialized), and user feedback.
    """

    CSV_COLUMNS: List[str] = ["text", "predicted_spans", "user_feedback"]

    def __init__(self, file_path: str | Path) -> None:
        """
        Initialize the FeedbackManager.

        Args:
            file_path: Path to the feedback CSV file.
                       File and parent directories will be created if they don't exist.
        """
        self._file_path = Path(file_path)
        self._ensure_file_exists()

    def _ensure_file_exists(self) -> None:
        """
        Ensure the feedback file exists with proper headers.
        Creates parent directories and file with header if not present.
        """
        # Create parent directories if they don't exist
        self._file_path.parent.mkdir(parents=True, exist_ok=True)

        # Create file with header if it doesn't exist
        if not self._file_path.exists():
            self._write_header()
        elif self._file_path.stat().st_size == 0:
            # File exists but is empty, write header
            self._write_header()

    def _write_header(self) -> None:
        """Write CSV header to the file."""
        with open(self._file_path, mode="w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(self.CSV_COLUMNS)

    def save_feedback(
        self,
        text: str,
        spans: List[Dict[str, Union[int, str]]],
        user_feedback: str
    ) -> None:
        """
        Save user feedback for a prediction.

        Appends feedback to the CSV file. Thread-safe via append-only writes.

        Args:
            text: The original input text that was analyzed.
            spans: List of predicted toxic spans. Each span has 'start', 'end',
                   and 'label' keys. Will be JSON serialized.
            user_feedback: User's feedback text (e.g., "correct", "incorrect",
                          or custom correction notes).

        Raises:
            IOError: If the file cannot be written.
        """
        # Serialize spans to JSON (create a copy to avoid mutation)
        spans_json = json.dumps(
            [dict(span) for span in spans],
            ensure_ascii=False
        )

        # Append to CSV file
        with open(self._file_path, mode="a", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([text, spans_json, user_feedback])

    def get_feedback_count(self) -> int:
        """
        Get the total number of feedback entries.

        Returns:
            Number of feedback rows (excluding header).
        """
        if not self._file_path.exists():
            return 0

        with open(self._file_path, mode="r", encoding="utf-8") as f:
            # Count lines minus header
            return max(0, sum(1 for _ in f) - 1)

    @property
    def file_path(self) -> Path:
        """Return the feedback file path."""
        return self._file_path

