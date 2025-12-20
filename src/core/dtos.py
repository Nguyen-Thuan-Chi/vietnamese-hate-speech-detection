from dataclasses import dataclass

# DTO representing a single text sample with optional label; kept minimal for interchange across layers
@dataclass
class HateSpeechSample:
    text: str
    label: str = None
