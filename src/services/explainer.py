# src/services/explainer.py
"""
Explainable AI service using LIME for sentence-level hate speech classification.
Provides local interpretability for black-box model predictions.
"""
from typing import Callable, List, Dict, Any
import numpy as np
from lime.lime_text import LimeTextExplainer


class LimeTextExplainerService:
    """
    Wraps LIME's LimeTextExplainer to generate word-level importance scores
    for sentence-level hate speech predictions.

    This service treats the model as a black box and requires only a
    predict_proba callable that returns class probabilities.

    Attributes:
        explainer: The underlying LIME text explainer instance.
        class_names: List of class labels (default: ["CLEAN", "TOXIC"]).
    """

    def __init__(
        self,
        class_names: List[str] = None,
        random_state: int = 42
    ) -> None:
        """
        Initialize the LIME explainer service.

        Args:
            class_names: Labels for classification classes.
                         Defaults to ["CLEAN", "TOXIC"].
            random_state: Seed for reproducibility in LIME's sampling.
        """
        self.class_names = class_names or ["CLEAN", "TOXIC"]
        self.explainer = LimeTextExplainer(
            class_names=self.class_names,
            random_state=random_state,
            # Split on whitespace for Vietnamese text (syllable-based)
            split_expression=r'\s+',
            bow=True  # Bag of words mode for sentence-level model
        )

    def explain(
        self,
        text: str,
        predict_proba_fn: Callable[[List[str]], np.ndarray],
        num_features: int = 10,
        num_samples: int = 500,
        label_index: int = 1
    ) -> List[Dict[str, Any]]:
        """
        Generate word-level explanation for a single input text.

        Args:
            text: The input sentence to explain.
            predict_proba_fn: A callable that accepts a list of text strings
                              and returns a numpy array of shape (n_samples, n_classes)
                              containing class probabilities.
            num_features: Maximum number of top words to include in explanation.
                          Words are ranked by absolute weight.
            num_samples: Number of perturbed samples LIME generates for fitting
                         the local linear model. Higher = more stable but slower.
            label_index: The class index to explain (default 1 = TOXIC).

        Returns:
            A list of dictionaries with 'word' and 'weight' keys, sorted by
            absolute weight descending. Positive weights indicate contribution
            toward the explained class; negative weights indicate opposition.

            Example:
            [
                {"word": "láo", "weight": 0.42},
                {"word": "thằng", "weight": 0.31},
                {"word": "này", "weight": -0.05}
            ]
        """
        if not text or not text.strip():
            return []

        # Generate LIME explanation
        explanation = self.explainer.explain_instance(
            text,
            predict_proba_fn,
            num_features=num_features,
            num_samples=num_samples,
            labels=(label_index,)
        )

        # Extract word weights for the specified label
        word_weights = explanation.as_list(label=label_index)

        # Sort by absolute weight (most influential first)
        sorted_weights = sorted(
            word_weights,
            key=lambda x: abs(x[1]),
            reverse=True
        )

        # Convert to UI-friendly format
        result = [
            {"word": word, "weight": round(weight, 4)}
            for word, weight in sorted_weights[:num_features]
        ]

        return result

    def explain_with_details(
        self,
        text: str,
        predict_proba_fn: Callable[[List[str]], np.ndarray],
        num_features: int = 10,
        num_samples: int = 500,
        label_index: int = 1
    ) -> Dict[str, Any]:
        """
        Generate explanation with additional metadata.

        Args:
            text: The input sentence to explain.
            predict_proba_fn: Probability prediction callable.
            num_features: Maximum number of top words.
            num_samples: Number of LIME samples.
            label_index: Class index to explain.

        Returns:
            Dictionary containing:
            - 'words': List of word-weight dicts (same as explain())
            - 'explained_class': The class name being explained
            - 'text': Original input text
        """
        words = self.explain(
            text=text,
            predict_proba_fn=predict_proba_fn,
            num_features=num_features,
            num_samples=num_samples,
            label_index=label_index
        )

        return {
            "words": words,
            "explained_class": self.class_names[label_index],
            "text": text
        }


def create_predict_proba_wrapper(predictor: Any) -> Callable[[List[str]], np.ndarray]:
    """
    Factory function to create a LIME-compatible predict_proba callable
    from a HateSpeechPredictor instance.

    This wrapper delegates to the predictor's predict_proba method
    which handles batch predictions and returns probabilities
    in the format expected by LIME.

    Args:
        predictor: A HateSpeechPredictor instance with a predict_proba() method.

    Returns:
        A callable that accepts List[str] and returns np.ndarray of shape
        (n_samples, 2) with probabilities for [CLEAN, TOXIC].
    """
    def predict_proba(texts: List[str]) -> np.ndarray:
        """
        Batch prediction returning class probabilities.

        Args:
            texts: List of text strings to classify.

        Returns:
            numpy array of shape (len(texts), 2) with probabilities.
        """
        return predictor.predict_proba(texts)

    return predict_proba

