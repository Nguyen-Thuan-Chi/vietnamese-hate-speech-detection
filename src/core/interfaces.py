# src/core/interfaces.py
from abc import ABC, abstractmethod
from typing import List
from src.core.dtos import HateSpeechSample

class IDataLoader(ABC):
    """
    Interface quy định việc đọc dữ liệu.
    """
    @abstractmethod
    def load_data(self, file_path: str) -> List[HateSpeechSample]:
        pass