import numpy as np
import joblib
from .preprocess import clean_text

class NewsClassificationPipeline:
    def __init__(self, vectorizer_path: str, model_path: str):
        self.vectorizer = joblib.load(vectorizer_path)
        self.model = joblib.load(model_path)
        self.class_names = ['World', 'Sports', 'Business', 'Sci/Tech']
        self.feature_names = np.array(self.vectorizer.get_feature_names_out())

    def predict(self, raw_texts: list):
        cleaned_texts = [clean_text(t) for t in raw_texts]
        vecs = self.vectorizer.transform(cleaned_texts)
        predictions = self.model.predict(vecs)
        results = []
        for i, text in enumerate(raw_texts):
            pred_class_idx = predictions[i]
            pred_class_name = self.class_names[pred_class_idx]
            row_vec = vecs[i].toarray().flatten()
            present_indices = np.where(row_vec > 0)[0]
            contributions = []
            for idx in present_indices:
                word = self.feature_names[idx]
                weight = self.model.coef_[pred_class_idx, idx] * row_vec[idx]
                contributions.append((word, weight))
            contributions = sorted(contributions, key=lambda x: x[1], reverse=True)[:4]
            results.append({
                'Text': text,
                'Predicted_Category': pred_class_name,
                'Top_Evidence_Keywords': [k[0] for k in contributions]
            })
        return results
