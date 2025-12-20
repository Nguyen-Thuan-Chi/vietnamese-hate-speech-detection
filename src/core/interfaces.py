# src/core/interfaces.py
from abc import ABC, abstractmethod
from typing import List
from src.core.dtos import HateSpeechSample

class IDataLoader(ABC):
    """
    Contract for dataset loaders returning normalized HateSpeechSample lists from a path.
    Implementations may support multiple dataset formats but must abstract file specifics.
    """
    @abstractmethod
    def load_data(self, file_path: str) -> List[HateSpeechSample]:
        pass