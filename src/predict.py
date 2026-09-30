from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np

from .preprocess import clean_text


class NewsClassificationPipeline:
    """Inference wrapper for the fitted TF-IDF + Linear SVM pipeline.

    The interpretability output is feature-level evidence from the SVM
    coefficients, not a probability or a model-agnostic explanation.
    """

    CLASS_NAMES = {
        0: "World",
        1: "Sports",
        2: "Business",
        3: "Sci/Tech",
    }

    def __init__(self, vectorizer_path: str | Path, model_path: str | Path):
        self.vectorizer = joblib.load(vectorizer_path)
        self.model = joblib.load(model_path)

        if not hasattr(self.vectorizer, "transform"):
            raise TypeError("vectorizer_path must contain a fitted vectorizer.")
        if not hasattr(self.model, "predict") or not hasattr(self.model, "coef_"):
            raise TypeError(
                "model_path must contain a linear classifier with predict() and coef_."
            )
        if not hasattr(self.model, "classes_"):
            raise TypeError("The classifier must expose classes_ for label mapping.")

        self.feature_names = np.asarray(
            self.vectorizer.get_feature_names_out()
        )
        self.classes_ = np.asarray(self.model.classes_)

        missing_labels = set(self.CLASS_NAMES) - set(self.classes_.tolist())
        if missing_labels:
            raise ValueError(
                "The classifier does not contain all expected AG News labels: "
                f"{sorted(missing_labels)}"
            )

    def predict(self, raw_texts: Sequence[str]) -> list[dict[str, Any]]:
        """Predict AG News categories and return positive feature evidence."""

        cleaned_texts = [clean_text(text) for text in raw_texts]
        vecs = self.vectorizer.transform(cleaned_texts)
        predictions = self.model.predict(vecs)

        results = []
        for row_index, raw_text in enumerate(raw_texts):
            predicted_label = predictions[row_index]

            class_matches = np.flatnonzero(self.classes_ == predicted_label)
            if len(class_matches) != 1:
                raise ValueError(
                    f"Unexpected predicted label {predicted_label!r}."
                )
            coefficient_row = int(class_matches[0])

            if int(predicted_label) not in self.CLASS_NAMES:
                raise ValueError(
                    f"Unexpected AG News label {predicted_label!r}."
                )

            row_vec = vecs[row_index]
            present_indices = row_vec.indices
            present_values = row_vec.data

            contributions = []
            for feature_index, tfidf_value in zip(
                present_indices, present_values
            ):
                score = float(
                    self.model.coef_[coefficient_row, feature_index]
                    * tfidf_value
                )
                if score > 0:
                    contributions.append(
                        (self.feature_names[feature_index], score)
                    )

            contributions.sort(key=lambda item: item[1], reverse=True)

            results.append(
                {
                    "Text": raw_text,
                    "Predicted_Category": self.CLASS_NAMES[int(predicted_label)],
                    "Top_Evidence_Keywords": [
                        keyword for keyword, _ in contributions[:4]
                    ],
                }
            )

        return results
