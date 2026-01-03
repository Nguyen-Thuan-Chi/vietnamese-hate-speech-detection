# src/services/__init__.py
"""
Services module - Business logic layer.
Contains pure Python services with no UI dependencies.
"""
from src.services.highlighter import ToxicHighlighter, KeywordHighlighter
from src.services.feedback import FeedbackManager
from src.services.explainer import LimeTextExplainerService, create_predict_proba_wrapper

__all__ = [
    "ToxicHighlighter",
    "KeywordHighlighter",
    "FeedbackManager",
    "LimeTextExplainerService",
    "create_predict_proba_wrapper",
]

